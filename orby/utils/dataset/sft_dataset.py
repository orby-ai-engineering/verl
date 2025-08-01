# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import io
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from datasets import Sequence, Image

logger = logging.getLogger(__name__)


def clean_dataset_for_training(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Remove unnecessary fields from dataset to keep only those used during training.
    Also standardizes image format and schema across different datasets.

    Required fields for training:
    - prompt: Used to build messages for chat template
    - response: Used to extend messages for SFT training
    - images: Used for multimodal processing
    - data_source: Used for logging

    Args:
        dataset: The dataset to clean

    Returns:
        Cleaned dataset with only necessary fields and standardized formats
    """
    # Define fields to keep
    fields_to_keep = {"prompt", "response", "images", "videos", "data_source"}

    # Get current features
    current_features = set(dataset.features.keys())

    # Find fields to remove
    fields_to_remove = current_features - fields_to_keep

    if fields_to_remove:
        logger.info(f"Removing unnecessary fields: {fields_to_remove}")
        dataset = dataset.remove_columns(list(fields_to_remove))

    # TODO: Reconcile the differences in image format and response schema between subtask_direct_distill and uground,os_atlas at the dataset level
    # Standardize image format if needed
    if "images" in dataset.features:
        # Check if images are in Dataset 2 format (list of dicts with bytes/path)
        # Dataset subtask_direct_distill format: [{'bytes': binary, 'path': int32}]
        # Dataset uground,os_atlas format: Sequence(feature=Image(mode=None, decode=True))
        if type(dataset.features["images"]) == list:
            logger.info(
                "Converting image format from [{'bytes': binary, 'path': int32}] to Sequence[Image]"
            )
            # Cast the images column to the new Sequence type
            dataset = dataset.cast_column(
                "images", Sequence(feature=Image(decode=True), length=-1)
            )

    # In all datasets, the response field is a list of dicts with role and content keys
    # The order of keys is "role" and "content" in subtask_direct_distill dataset, and "content" and "role" in uground,os_atlas dataset
    # This leads to a mismatch in the response schema between the two datasets, which results in a failure to concatenate the two datasets
    # Change response message key order for subtask_direct_distill dataset
    if (
        "data_source" in dataset.features
        and dataset["data_source"][0] == "subtask_direct_distill"
    ):
        logger.info("Standardizing response message key order")

        # Explicitly cast the response column to ensure schema compatibility
        response_features = [
            {"content": datasets.Value("string"), "role": datasets.Value("string")}
        ]

        # Create new features with the correct order
        new_features = dataset.features.copy()
        new_features["response"] = response_features

        # Cast the dataset to the new schema
        dataset = dataset.cast(new_features)

    logger.info(f"Dataset features: {dataset.features}")
    return dataset


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class SFTDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information

    Note that this dataset is a "copy" or RLHFDataset from verl, and has been modified for SFT
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(
            config.get("cache_dir", "~/.cache/verl/rlhf")
        )
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "extra_info")
        self.response_dict_key = config.get("response_dict_keys", None)
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 5000)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.clean_dataset = config.get("clean_dataset", True)

        self.num_workers = config.get(
            "filter_overlong_prompts_workers", max(1, os.cpu_count() // 4)
        )
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = (
            self.data_files if not use_origin_parquet else self.original_data_files
        )
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(
                src=parquet_file, cache_dir=self.cache_dir
            )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)[
                "train"
            ]
            # Clean the dataset to remove unnecessary fields
            logger.info(f"Dataset features for {parquet_file}")
            if self.clean_dataset:
                dataframe = clean_dataset_for_training(dataframe)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(
                    tokenizer.apply_chat_template(
                        doc[prompt_key], add_generation_prompt=True
                    )
                )
                <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(
                use_origin_parquet=True
            )  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(
                r"old dataloader ckpt file is used, please train from scratch for better ckpt performance"
            )

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        # SFT Training will always assume that the response_key is 'response' (which is different from verl which assumes that it is in the extra_info dict under the key 'answer')
        if (self.response_key != "response") or (self.response_key not in example):
            raise ValueError(
                f"response_key '{self.response_key}' is not set to 'response' or is not present in example. Available keys: {list(example.keys())}"
            )

        response_messages = ""

        response_messages = example[self.response_key]
        if response_messages:
            messages.extend(response_messages)
        else:
            raise ValueError("No response messages found in example")

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [
                    process_image(image) for image in row_dict.pop(self.image_key)
                ]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [
                    process_video(video) for video in row_dict.pop(self.video_key)
                ]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        ###### ORBY CHANGES START
        # Added changes to handle fast processor for Qwen2VLImageProcessor
        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            ### ORBY CHANGES END
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get(
            "need_tools_kwargs", self.need_tools_kwargs
        )
        if need_tools_kwargs and not tools_kwargs:
            logger.warning(
                "tools_kwargs is empty for index {}, data source: {}",
                index,
                row_dict["data_source"],
            )
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
