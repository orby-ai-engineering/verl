# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

"""
TODO(Rishu)
- Add sequence parallel support
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    Qwen2_5_VLForConditionalGeneration,
)
from torchdata.stateful_dataloader import StatefulDataLoader

import verl.utils.hdfs_io as hdfs_io
from orby.utils.dataset.sft_dataset import collate_fn
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.torch_functional import (
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.device import (
    get_device_name,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from omegaconf import OmegaConf

if is_cuda_available:
    from flash_attn.bert_padding import (
        pad_input,
        unpad_input,
        rearrange,
        index_first_axis,
    )
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import (
        pad_input,
        unpad_input,
        rearrange,
        index_first_axis,
    )

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def create_sft_multimodal_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from orby.utils.dataset.sft_dataset import SFTDataset

    if (
        "custom_cls" in data_config
        and data_config.custom_cls.get("path", None) is not None
    ):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(
            data_config.custom_cls.path, data_config.custom_cls.name
        )
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        dataset_cls = SFTDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        processor,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        self.processor = processor
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(
            self.config, "ulysses_sequence_parallel_size", 1
        )
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(
                f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}"
            )
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = (
            self.device_mesh.size(0)
            if not self.ulysses_device_mesh
            else self.ulysses_device_mesh.size(0)
        )
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert (
            self.config.data.train_batch_size % dp_size == 0
        ), f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert (
            self.config.data.train_batch_size
            % self.config.data.micro_batch_size_per_gpu
            == 0
        )

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(
                    f"Using SP rank {rank} and size {world_size} for data distribution"
                )
                print(
                    "Each SP rank gets different data, but the same data WITHIN the same rank"
                )
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )

        # For multimodal inputs, use StatefulDataLoader
        # which supports multimodal batching and includes a collate_fn for properly handling images

        if self.config.data.get("image_key", None) is not None:
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=config.data.train_batch_size,
                num_workers=8,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=self.train_sampler,
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=config.data.micro_batch_size_per_gpu,
                num_workers=8,
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=self.val_sampler,
            )
        else:
            self.train_dataloader = DataLoader(
                dataset=self.train_dataset,
                batch_size=config.data.train_batch_size,
                sampler=self.train_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )

            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=config.data.micro_batch_size_per_gpu,
                sampler=self.val_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(
            src=self.config.model.partial_pretrain, verbose=True
        )

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(
            local_model_path, trust_remote_code=trust_remote_code
        )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert (
                self.use_remove_padding
            ), "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            model = None

            if self.config.data.image_key is not None:
                kwargs = {}
                if self.config.model.get("qwen_attention_dropout", None):
                    kwargs.update(
                        {
                            "attention_dropout": self.config.model.qwen_attention_dropout,
                        }
                    )
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    # device_map=torch.cuda.current_device(), doesn't work with meta tensors
                    attn_implementation="sdpa",
                    **kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    config=config,
                    torch_dtype=torch.float32,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=trust_remote_code,
                )
            self.model: PreTrainedModel = model

            if (
                self.use_remove_padding
                or self.config.ulysses_sequence_parallel_size > 1
            ):
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=self.model,
                    ulysses_sp_size=self.config.ulysses_sequence_parallel_size,
                )

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(
                offload_params=self.config.model.fsdp_config.offload_params
            )

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=get_torch_device().current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if (
            not hasattr(self.config.optim, "lr_scheduler")
            or self.config.optim.lr_scheduler == "cosine"
        ):
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = (
            self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        )

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        raw_prompt_ids = batch["raw_prompt_ids"]
        multi_modal_inputs = batch.get("multi_modal_inputs", {})

        if position_ids.dim() == 3:
            # When processing multimodal data (text + images), Qwen2.5-VL uses 3D position embeddings
            # where each token gets 3 coordinates: [t, h, w] representing temporal, height, width dimensions.
            #
            # The get_rope_index() function returns position_ids as (3, seq_len) where:
            # - Dimension 0: temporal coordinates for all tokens
            # - Dimension 1: height coordinates for all tokens
            # - Dimension 2: width coordinates for all tokens
            #
            # However, the Qwen2.5-VL attention mechanism expects (seq_len, 3) where:
            # - Each row represents one token's [t, h, w] coordinates  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)

        if "loss_mask" in batch:
            loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        else:
            batch_size, seq_len = input_ids.shape

            # Start with attention mask as base
            loss_mask = attention_mask.clone()  # (batch_size, seq_len)

            # Create a mask for prompt tokens to exclude from loss
            # Locate where assistant responses begin by finding the "<|im_start|>assistant" token sequence
            # This ensures we only train on actual response tokens, not prompt or image tokens
            assistant_token_ids = torch.tensor(
                self.tokenizer.encode(
                    "<|im_start|>assistant", add_special_tokens=False
                ),
                device=attention_mask.device,
            )

            assistant_len = len(assistant_token_ids)
            prompt_end_position = torch.zeros(
                batch_size, device=attention_mask.device, dtype=torch.long
            )

            # Slide a window across each position to find where assistant tokens start
            # We only need to check positions where the full assistant sequence could fit
            # TODO: (RISHU) Find a more efficient way to do this
            for i in range(seq_len - assistant_len + 1):
                # Compare assistant token sequence against all batch sequences simultaneously
                # This leverages GPU parallelization for efficient batch processing
                matches = torch.all(
                    input_ids[:, i : i + assistant_len]
                    == assistant_token_ids.unsqueeze(0),
                    dim=1,
                )
                # Record the end position of assistant prefix for sequences where we found the first match
                # The mask ensures we only capture the first occurrence per sequence
                mask = matches & (prompt_end_position == 0)
                prompt_end_position[mask] = i + assistant_len

            position_indices = (
                torch.arange(seq_len, device=attention_mask.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Mask out prompt tokens (everything before the prompt_end_position)
            prompt_mask = position_indices < prompt_end_position.unsqueeze(1)
            loss_mask = loss_mask.masked_fill(prompt_mask, 0)
            # Mask out the last token of each sequence
            # Find the last valid token position for each sequence
            last_token_positions = attention_mask.sum(dim=1) - 1  # (batch_size,)

            # Create mask for last tokens
            last_token_mask = position_indices == last_token_positions.unsqueeze(1)
            loss_mask = loss_mask.masked_fill(last_token_mask, 0)
            # Remove last column and flatten
            loss_mask = loss_mask[:, :-1].reshape(-1).to(self.device_name)
            debug_loss_mask_2d = (
                loss_mask.view(batch_size, -1)
                if self.device_mesh.get_rank() == 0
                else None
            )
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(
            device_type=self.device_name, dtype=torch.bfloat16
        ):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                model_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "use_cache": False,
                }
                multimodal_kwargs = {}
                if len(multi_modal_inputs) > 0 and hasattr(multi_modal_inputs, "data"):
                    for key in multi_modal_inputs.data[0].keys():
                        multimodal_kwargs[key] = torch.cat(
                            [inputs[key] for inputs in multi_modal_inputs.data], dim=0
                        ).cuda()
                    model_kwargs.update(multimodal_kwargs)
                output = self.fsdp_model(**model_kwargs)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                ## Enhanced Debug prints - Show model predictions
                # if self.device_mesh.get_rank() == 0 and debug_loss_mask_2d is not None:
                # print("\n=== Debug Info ===")
                # print("Full prompt:")
                # print(self.tokenizer.decode(input_ids[0]))

                ## Ground truth tokens under loss
                # ground_truth_tokens = input_ids[0, 1:][debug_loss_mask_2d[0].bool()]
                # print("\nGround truth (tokens under loss):")
                # print(self.tokenizer.decode(ground_truth_tokens))
                # print(f"\nTotal tokens under loss: {loss_mask.sum().item()}")

                ## Model predictions
                # with torch.no_grad():
                #    predicted_token_ids = torch.argmax(shift_logits.view(batch_size, -1, self.model.config.vocab_size), dim=-1)
                #    predicted_tokens_masked = predicted_token_ids[0][debug_loss_mask_2d[0].bool()]
                #    print("\nModel predictions (tokens under loss):")
                #    print(self.tokenizer.decode(predicted_tokens_masked))
                #
                #    # Token-by-token comparison
                #    print("\n=== Token-by-Token Comparison ===")
                #    ground_truth_list = ground_truth_tokens.tolist()
                #    predicted_list = predicted_tokens_masked.tolist()
                #
                #    min_len = min(len(ground_truth_list), len(predicted_list))
                #    for i in range(min_len):
                #        gt_id = ground_truth_list[i]
                #        pred_id = predicted_list[i]
                #        gt_token = self.tokenizer.decode([gt_id])
                #        pred_token = self.tokenizer.decode([pred_id])
                #        match = "✓" if gt_id == pred_id else "✗"
                #        print(f"{i:3d}: GT='{gt_token}' | PRED='{pred_token}' {match}")
                #
                # print("================\n")
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices,
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = (
                    ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=get_ulysses_sequence_parallel_world_size(),
                    )
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                )  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled,
                    None,
                    get_ulysses_sequence_parallel_world_size(),
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                    0
                )  # ((total_nnz / sp) + pad)
                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(
                    loss, gather_dim=0, unpad_dim=0, padding_size=pad_size
                )

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = (
                    self.ulysses_device_mesh.size("dp")
                    if use_sp
                    else torch.distributed.get_world_size()
                )
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = self._split_batch_with_indices(
            batch, self.config.data.micro_batch_size_per_gpu
        )
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch) / n_micro_batches
            step_loss += loss.item()

        grad_norm = self.fsdp_model.clip_grad_norm_(
            max_norm=self.config.optim.clip_grad
        )

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.ulysses_device_mesh.size(0)
        return {"train/loss": step_loss.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def _split_batch_with_indices(self, batch: TensorDict, micro_batch_size: int):
        """
        Split batch into micro-batches for gradient accumulation while preserving text-image correspondence.
        by pairing each text sample with only its corresponding images.
        """
        batch_size = batch.batch_size[0]
        micro_batches = []

        for start_idx in range(0, batch_size, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, batch_size)
            indices = list(range(start_idx, end_idx))

            # Create micro-batch by slicing
            micro_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batch[key] = value[start_idx:end_idx]
                else:
                    # Handle non-tensor data (like multi_modal_inputs)
                    if hasattr(value, "data") and isinstance(value.data, list):
                        # For multimodal inputs, slice the data list
                        sliced_data = value.data[start_idx:end_idx]
                        micro_batch[key] = type(value)(data=sliced_data)
                    else:
                        micro_batch[key] = (
                            value[start_idx:end_idx]
                            if hasattr(value, "__getitem__")
                            else value
                        )

            # Create TensorDict for the micro-batch
            micro_batch_td = TensorDict(micro_batch, batch_size=(end_idx - start_idx,))
            # Store the original indices for reference
            micro_batch_td._original_indices = indices
            micro_batches.append(micro_batch_td)

        return micro_batches

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.ulysses_device_mesh.size(0)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{step}"
        )

        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            if self.config.trainer.default_local_dir.startswith("s3://"):
                # For S3, use temporary directory
                import tempfile, shutil

                with tempfile.TemporaryDirectory() as temp_dir:
                    local_path = os.path.join(temp_dir, f"global_step_{step}")
                    os.makedirs(local_path, exist_ok=True)
                    self.model.save_pretrained(local_path, state_dict=state_dict)
                    self.tokenizer.save_pretrained(local_path)
                    if hasattr(self, "processor") and self.processor is not None:
                        self.processor.save_pretrained(local_path)
                    self._upload_directory_to_s3(local_path, path)
            else:
                # Original logic for local/HDFS
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
                if hasattr(self, "processor") and self.processor is not None:
                    self.processor.save_pretrained(path)
                if self.config.trainer.default_hdfs_dir:
                    hdfs_io.makedirs(
                        self.config.trainer.default_hdfs_dir, exist_ok=True
                    )
                    hdfs_io.copy(
                        src=path,
                        dst=self.config.trainer.default_hdfs_dir,
                        dirs_exist_ok=True,
                    )
        torch.distributed.barrier()

    def _upload_directory_to_s3(self, local_dir: str, s3_path: str):
        """Upload local directory to S3"""
        print(f"Uploading checkpoint from {local_dir} to {s3_path}")
        from verl.utils.s3_io import file_upload, parse_uri

        bucket, prefix, _ = parse_uri(s3_path, is_dir=True)

        for root, _, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, local_dir)
                s3_key = f"{prefix}{relative_path}"
                file_upload(bucket, file_path, s3_key)

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            ):
                global_step += 1
                data = TensorDict(
                    data, batch_size=self.config.data.train_batch_size
                ).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                # Validation after every val_interval steps (default: 25)
                if global_step % self.config.trainer.val_interval == 0:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        batch_size = (
                            len(val_data["input_ids"])
                            if "input_ids" in val_data
                            else len(next(iter(val_data.values())))
                        )
                        val_data = TensorDict(val_data, batch_size=batch_size).cuda()
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric_val = {"val/loss": avg_val_loss.detach().item()}
                        tracking.log(data=metric_val, step=global_step)
                    torch.distributed.barrier()

                # Save checkpoint after every save_interval steps (default: 50)
                if global_step % self.config.trainer.save_interval == 0:
                    self.save_checkpoint(step=global_step)
                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        # mostly for last batch: If validation set size isn't divisible by micro_batch_size
                        batch_size = (
                            len(val_data["input_ids"])
                            if "input_ids" in val_data
                            else len(next(iter(val_data.values())))
                        )
                        val_data = TensorDict(val_data, batch_size=batch_size).cuda()
                        # val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": avg_val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                    torch.distributed.barrier()

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

            # validation
            val_losses = []
            for data in self.val_dataloader:
                batch_size = (
                    len(data["input_ids"])
                    if "input_ids" in data
                    else len(next(iter(data.values())))
                )
                data = TensorDict(data, batch_size=batch_size).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)


@hydra.main(
    config_path="../../verl/trainer/config/",
    config_name="sft_trainer",
    version_base=None,
)
def main(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(
        local_model_path, trust_remote_code=config.model.trust_remote_code
    )
    processor = hf_processor(local_model_path, **config.get("processor", {}))
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, processor
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, processor
    )

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()


def create_sft_dataset(data_paths, data_config, tokenizer, processor=None):

    # if multi-modal dataset, use the modified RLHF dataset from VERL for SFT that has support for multi-modal inputs
    if data_config.get("image_key", None) is not None:
        return create_sft_multimodal_dataset(
            data_paths, data_config, tokenizer, processor
        )
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(
            data_config.custom_cls.path, data_config.custom_cls.name
        )
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        from verl.utils.dataset import SFTDataset as VERL_SFTDataset

        dataset_cls = VERL_SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config
    )
    return dataset


if __name__ == "__main__":
    main()
