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
# Download the OS Atlas dataset
huggingface-cli download OS-Copilot/OS-Atlas-data \
    --repo-type dataset \
    --local-dir=$HOME/data/os_atlas \
    --include="desktop_domain/linux_splited.json"
huggingface-cli download OS-Copilot/OS-Atlas-data \
    --repo-type dataset \
    --local-dir=$HOME/data/os_atlas \
    --include="desktop_domain/linux_images.zip"
# Unzip the images
cd $HOME/data/os_atlas/desktop_domain/
unzip linux_images.zip
cd -
# Convert the dataset to parquet format
python orby/data/convert_os_atlas.py \
    --json_file ~/data/os_atlas/desktop_domain/linux_splited.json \
    --image_dir ~/data/os_atlas/desktop_domain/ \
    --local_dir ~/data/os_atlas/desktop_domain \
    --output_filename linux \
    --prompt_format=sft
"""



"""
Preprocess the Uground dataset to parquet format
"""

import argparse
import io
import json
import os
import math

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from orby.utils.dataset.qwen_agent_function_call import ComputerUse
from orby.data.prompts import get_subtask_messages


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")

def get_resized_hw(image, max_pixels=None):
    """
    Get the resized width and height of the image.
    """

     # if max_pixels is not set, use the max pixels of the image processor
    if not max_pixels:
        print("Max pixels not set, using the max pixels of the image processor", flush=True)
        max_pixels = PROCESSOR.image_processor.max_pixels
    
    resized_height, resized_width = smart_resize(
        height=image.height,
        width=image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=max_pixels,
    )

    return resized_height, resized_width


def save_in_chunks(
    all_data, output_dir, prefix, start_file_counter=0
):
    """Save processed data in multiple parquet files"""
    os.makedirs(output_dir, exist_ok=True)

    file_counter = start_file_counter

    # If all_data is a single dataset, convert to list
    if not isinstance(all_data, list):
        all_data = [all_data]

    # Process each dataset chunk immediately
    for dataset_chunk in all_data:
        if len(dataset_chunk) == 0:
            continue
        
         # Remove width and height columns if they exist
        columns_to_remove = []
        if "width" in dataset_chunk.column_names:
            columns_to_remove.append("width")
        if "height" in dataset_chunk.column_names:
            columns_to_remove.append("height")

        if columns_to_remove:
            dataset_chunk = dataset_chunk.remove_columns(columns_to_remove)
            print(f"Removed columns: {columns_to_remove}", flush=True)

        # Save the chunk as-is (remove the splitting logic)
        output_file = os.path.join(
            output_dir, f"{prefix}_part_{file_counter:04d}.parquet"
        )
        dataset_chunk.to_parquet(output_file)
        print(f"✓ Saved {len(dataset_chunk)} examples to {output_file}", flush=True)
        file_counter += 1

    return file_counter


def load_os_atlas_data(json_path: str, image_dir: str, max_examples: int = None):
    """Load and process OS Atlas data from JSON and image directory."""
    with open(json_path, "r") as f:
        data = json.load(f)

    examples_yielded = 0

    for item in data:
        if max_examples is not None and examples_yielded >= max_examples:
            break

        img_filename = item["img_filename"]
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue

        # Load and process image
        try:
            image = Image.open(img_path)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or "PNG")
            img_byte_arr = img_byte_arr.getvalue()
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

        # Process each element in the elements list
        for element in item["elements"]:
            instruction = element["instruction"]
            bbox = element["bbox"]

            # TODO: whether this works for from_generator() below.
            yield (
                {
                    "image": img_byte_arr,
                    "instruction": instruction,
                    "bbox": bbox,
                    "img_filename": img_filename,
                }
            )
            examples_yielded += 1
    print(f"Yielded {examples_yielded} examples")
    

def process_in_chunks(dataset, chunk_size):
    """Process dataset in chunks with immediate saving capability"""
    chunk = []
    total_processed = 0


    for i, example in enumerate(dataset):
        if (
            hasattr(process_in_chunks, "max_examples")
            and total_processed >= process_in_chunks.max_examples
        ):
            break

        chunk.append(example)

        if len(chunk) >= chunk_size:
            print(
                f"Processing chunk {total_processed//chunk_size + 1}, examples {total_processed}-{total_processed + len(chunk)}",
                flush=True,
            )

            # Convert chunk to Dataset for processing
            chunk_dataset = Dataset.from_list(chunk)

            # Process the chunk
            processed_chunk = chunk_dataset.map(
                function=process_in_chunks.map_fn,
                with_indices=True,
                num_proc=4,  # Reduced from 16 to manage memory
            )
            processed_chunk = processed_chunk.cast_column(
                "images", Sequence(ImageData())
            )

            yield processed_chunk, total_processed

            total_processed += len(chunk)
            chunk = []

    # Process remaining examples
    if chunk:
        print(
            f"Processing final chunk, examples {total_processed}-{total_processed + len(chunk)}",
            flush=True,
        )
        chunk_dataset = Dataset.from_list(chunk)
        processed_chunk = chunk_dataset.map(
            function=process_in_chunks.map_fn, with_indices=True, num_proc=4
        )
        processed_chunk = processed_chunk.cast_column("images", Sequence(ImageData()))
        yield processed_chunk, total_processed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/os_atlas/desktop_domain/os_atlas_converted/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--prompt_format",
        choices=["qwen", "thinking", "subtask", "sft"],
        default="subtask",
        help="Select prompt format: 'qwen' or 'thinking' or 'subtask' or 'sft'",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of examples per chunk",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100000,
        help="Maximum number of examples to process (for testing)",
    )

    parser.add_argument(
        "--json_file",
        default="/root/data/os_atlas/desktop_domain/filtered_data_clean.json",
        help="Path to the JSON file containing OS Atlas data",
    )
    parser.add_argument(
        "--image_dir", default="/root/data/os_atlas/desktop_domain/merged_images/", help="Path to the directory containing images"
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum number of pixels in the image",
    )


    args = parser.parse_args()

    if args.max_examples and args.max_examples > 5000:
        print(f"⚠️  WARNING: You've set max_examples to {args.max_examples:,}, which is quite large.")
        print("   This will process a lot of data and may take a long time.")
        response = input("   Are you sure you want to continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("   Exiting...")
            exit(0)
        print("   Continuing with processing...")


    # Load in streaming mode
    dataset = Dataset.from_generator( lambda: load_os_atlas_data(args.json_file, args.image_dir, args.max_examples)
)

    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop("image")
            instruction = example.pop("instruction")
            print('instruction', instruction)
            bbox = example.pop("bbox")
            assert len(bbox) == 4, f"Expected 4 coordinates, got {len(bbox)}"

            # Get image and resize ratios
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            # Convert image to RGB if it's RGBA
            image = to_rgb(image)
            # Get the resized width and height of the image.
            resized_height, resized_width = get_resized_hw(image, args.max_pixels)

            
            bbox = [
                bbox[0] * resized_width,
                bbox[1] * resized_height,
                bbox[2] * resized_width,
                bbox[3] * resized_height,
            ]

            ground_truth = {
                "bbox": bbox,
            }

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            answer = [
                {"role": "assistant", "content": f"<answer>click({center_x:.0f}, {center_y:.0f})</answer>"}
            ]
            

            data = {
                "data_source": "uground",
                "images": [image],
                "ability": "vision-grounding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": instruction,
                    "bounding_box": bbox,
                    "max_pixels": args.max_pixels,
                },
                "response": answer
            }

            # Create prompt based on selected format

            if args.prompt_format == "thinking":
                data["prompt"] = [
                    {
                        "role": "user",
                        "content": (
                            "Map the user instruction to the coordinates in the UI image. "
                            "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                            "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                            "<image> Instruction: " + instruction
                        ),
                    },
                ]
            elif args.prompt_format == "sft":
                data["prompt"] = [
                    {
                        "role": "user",
                        "content": ("<image> Instruction: " + instruction),
                    },
                ]
            elif args.prompt_format == "subtask":
                prompt = get_subtask_messages(instruction)
                data["prompt"] = prompt
            elif args.prompt_format == "qwen":
                prompt = NousFnCallPrompt().preprocess_fncall_messages(
                    messages=[
                        Message(
                            role="system",
                            content=[ContentItem(text="You are a helpful assistant.")],
                        ),
                        Message(
                            role="user",
                            content=[
                                ContentItem(text=instruction + "<image>"),
                            ],
                        ),
                    ],
                    functions=[
                        ComputerUse(
                            cfg={
                                "display_width_px": resized_width,
                                "display_height_px": resized_height,
                            }
                        ).function
                    ],
                    lang=None,
                )

                prompt = [msg.model_dump() for msg in prompt]
                for message in prompt:
                    # Replace the list of content to a string.
                    content = "".join(m["text"] for m in message["content"])
                    message["content"] = content

                data["prompt"] = prompt
            


            return data

        return process_fn

    local_dir = os.path.expanduser(args.local_dir)
    local_dir = os.path.join(local_dir, args.prompt_format)
    if args.max_examples % 1000 == 0:
        folder_name = f"{args.max_examples//1000}k"
    else:
        folder_name = f"{args.max_examples/1000:.2f}k"
    local_dir = os.path.join(local_dir, folder_name)
    print(f"Saving to {local_dir}...", flush=True)
    os.makedirs(local_dir, exist_ok=True)

    # Initialize counters and directories
    train_file_counter = 0
    test_file_counter = 0
    total_processed = 0

    train_dir = os.path.join(local_dir, "train")
    test_dir = os.path.join(local_dir, "test")

    # Set up the map function for process_in_chunks
    process_in_chunks.map_fn = make_map_fn("train")
    process_in_chunks.max_examples = args.max_examples

    for chunk_dataset, chunk_start in process_in_chunks(dataset, args.chunk_size):
        # Split each chunk into train/test
        chunk_split = chunk_dataset.train_test_split(train_size=0.95, seed=42)
        train_chunk = chunk_split["train"]
        test_chunk = chunk_split["test"]

        # Save train data
        train_file_counter = save_in_chunks(
            [train_chunk],
            train_dir,
            "train",
            train_file_counter,
        )

        # Save test data
        test_file_counter = save_in_chunks(
            [test_chunk],
            test_dir,
            "test",
            test_file_counter,
        )

        total_processed += len(chunk_dataset)

    print(f"Processing completed! {total_processed} examples processed")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
