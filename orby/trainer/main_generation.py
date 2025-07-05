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
Generate responses given a dataset of prompts. Multimodal input is supported.
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import DataLoader

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
from verl.trainer.main_ppo import create_rl_dataset


def _create_dataloader(path, config, tokenizer, processor):
    """
    Creates the dataloader.
    """
    dataset = create_rl_dataset(path, config.data, tokenizer, processor)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        shuffle=False,
        drop_last=False,
        collate_fn=default_collate_fn,
    )

    assert len(dataloader) >= 1, "Dataloader is empty!"
    print(f"Size of dataloader: {len(dataloader)}")
    return dataloader


@hydra.main(
    config_path="../../verl/trainer/config", config_name="generation", version_base=None
)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            },
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=64)
def main_task(config):
    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(
        local_path, use_fast=True,
    )  # used for multimodal LLM, could be none

    paths = config.data.path.split(",")
    output_paths = config.data.output_path.split(",")

    assert len(paths) == len(
        output_paths
    ), "Number of paths and output paths must be the same"

    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout"
    )
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes
    )
    wg = RayWorkerGroup(
        resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init
    )
    wg.init_model()

    for path, output_path in zip(paths, output_paths):
        print(f"Processing {path}...")
        dataset = _create_dataloader(path, config, tokenizer, processor)

        if config.rollout.temperature == 0.0:
            assert (
                config.data.n_samples == 1
            ), "When temperature=0, n_samples must be 1."
        assert config.data.n_samples >= 1, "n_samples should always >= 1"

        output_lst = []
        for batch_idx, batch_dict in enumerate(dataset):
            print(f"[{batch_idx + 1}] Start to process.")
            data = DataProto.from_single_dict(batch_dict)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in data.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(
                    ["multi_modal_data", "multi_modal_inputs"]
                )
            if "raw_prompt" in data.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in data.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            data = data.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}] Start to generate.")
            output_padded = wg.generate_sequences(data_padded)
            output_padded.batch = output_padded.batch.reshape((-1, config.data.n_samples))
            # Only keep the first batch size dim.
            output_padded.batch.batch_size = output_padded.batch.batch_size[:1]
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            output_texts = []
            for i, data_item in enumerate(output):
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][
                     :, prompt_length:
                ].sum(axis=-1)
                responses = data_item.batch["responses"]
                responses = [r[:l] for r, l in zip(responses, valid_response_length)]
                response_str = tokenizer.batch_decode(
                    responses, skip_special_tokens=True
                )
                output_texts.append([response_str])
            output_lst.append(np.concatenate(output_texts))

        # output_lst shape: (n_data, n_sampels)
        output_lst = np.concatenate(output_lst)

        # add to the data frame
        dataset = pd.read_parquet(path)
        dataset["responses"] = output_lst

        # write to a new parquet
        output_dir = os.path.dirname(output_path)
        if output_dir != "":
            makedirs(output_dir, exist_ok=True)
        dataset.to_parquet(output_path, row_group_size=512)


if __name__ == "__main__":
    main()
