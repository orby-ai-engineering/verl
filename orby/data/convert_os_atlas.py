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
    desktop_domain/linux_splited.json

huggingface-cli download OS-Copilot/OS-Atlas-data \
    --repo-type dataset \
    --local-dir=$HOME/data/os_atlas \
    desktop_domain/linux_images.zip

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

import argparse
import json
import os

from datasets import Sequence, Image as ImageData
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


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


def get_resized_wh(image):
    """
    Get the resized width and height of the image.
    """
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=PROCESSOR.image_processor.max_pixels,
    )

    return resized_height, resized_width


def save_in_chunks(
    all_data, output_dir, prefix, max_examples_per_file=12500, start_file_counter=0
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

        # If this chunk is larger than max_examples_per_file, split it
        if len(dataset_chunk) > max_examples_per_file:
            for start_idx in range(0, len(dataset_chunk), max_examples_per_file):
                end_idx = min(start_idx + max_examples_per_file, len(dataset_chunk))
                sub_chunk = dataset_chunk.select(range(start_idx, end_idx))

                output_file = os.path.join(
                    output_dir, f"{prefix}_part_{file_counter:04d}.parquet"
                )
                sub_chunk.to_parquet(output_file)
                print(f"✓ Saved {len(sub_chunk)} examples to {output_file}", flush=True)
                file_counter += 1
        else:
            # Save the chunk as-is
            output_file = os.path.join(
                output_dir, f"{prefix}_part_{file_counter:04d}.parquet"
            )
            dataset_chunk.to_parquet(output_file)
            print(f"✓ Saved {len(dataset_chunk)} examples to {output_file}", flush=True)
            file_counter += 1

    return file_counter


def process_in_chunks(streaming_dataset, chunk_size=1000):
    """Process streaming dataset in chunks with immediate saving capability"""
    chunk = []
    total_processed = 0

    for i, example in enumerate(streaming_dataset):
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
                num_proc=4,
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


def load_os_atlas_data(json_path: str, image_dir: str):
    """Load and process OS Atlas data from JSON and image directory."""
    with open(json_path, "r") as f:
        data = json.load(f)

    for item in data:
        img_filename = item["img_filename"]
        img_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            continue

        # Load and process image
        image = Image.open(img_path)

        # Process each element in the elements list
        for element in item["elements"]:
            instruction = element["instruction"]
            bbox = element["bbox"]

            # TODO: whether this works for from_generator() below.
            yield (
                {
                    "image": image,
                    "instruction": instruction,
                    "bbox": bbox,
                    "img_filename": img_filename,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        required=True,
        help="Path to the JSON file containing OS Atlas data",
    )
    parser.add_argument(
        "--image_dir", required=True, help="Path to the directory containing images"
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/os_atlas",
        help="Local directory to save processed data",
    )
    parser.add_argument(
        "--hdfs_dir", default=None, help="HDFS directory to save processed data"
    )
    parser.add_argument(
        "--output_filename", default="train", help="Output filename prefix"
    )
    parser.add_argument(
        "--prompt_format",
        choices=["qwen", "thinking", "sft"],
        default="sft",
        help="Select prompt format: 'qwen' or 'thinking' or 'sft'",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of examples to process at once",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)",
    )
    parser.add_argument(
        "--max_examples_per_file",
        type=int,
        default=12500,
        help="Maximum examples per output parquet file",
    )

    args = parser.parse_args()

    # Load and process data
    data = load_os_atlas_data(args.json_file, args.image_dir)
    dataset = Dataset.from_generator(data)

    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop("image")
            instruction = example.pop("instruction")
            bbox = example.pop("bbox")
            img_filename = example.pop("img_filename")

            # Get image and resize ratios
            resized_height, resized_width = get_resized_wh(image)

            # Adjust bbox based on resize ratios
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

            data = {
                "data_source": "os_atlas",
                "images": [image],
                "ability": "vision",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": instruction,
                    "bounding_box": bbox,
                    "img_filename": img_filename,
                },
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
                data["response"] = [
                    {
                        "role": "assistant",
                        "content": f"<answer>{center_x:.0f} {center_y:.0f}</answer>",
                    }
                ]
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
    if args.prompt_format == "sft":
        local_dir += "_sft"

    print(f"Saving to {local_dir}...", flush=True)
    os.makedirs(local_dir, exist_ok=True)

    if args.prompt_format == "sft":
        # Set up progress tracking
        progress_file = os.path.join(local_dir, "processing_progress.json")

        # Check for existing progress
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
            if progress.get("status") == "completed":
                print(
                    f"✓ Processing already completed! Found {progress['total_processed']} examples",
                    flush=True,
                )
                exit(0)
            else:
                print(
                    f"Found previous progress: {progress['total_processed']} examples processed",
                    flush=True,
                )

        # Initialize counters and directories
        train_file_counter = 0
        test_file_counter = 0
        total_processed = 0

        train_dir = os.path.join(local_dir, "train")
        test_dir = os.path.join(local_dir, "test")

        # Accumulators for batching
        train_accumulator = []
        test_accumulator = []
        train_accumulator_size = 0
        test_accumulator_size = 0

        # Calculate save frequency
        save_frequency_chunks = max(
            1, args.max_examples_per_file // (args.chunk_size * 4)
        )
        chunks_processed = 0

        # Set up the map function for process_in_chunks
        process_in_chunks.map_fn = make_map_fn("train")
        process_in_chunks.max_examples = args.max_examples

        try:
            for chunk_dataset, chunk_start in process_in_chunks(
                dataset, args.chunk_size
            ):
                # Split each chunk into train/test
                chunk_split = chunk_dataset.train_test_split(train_size=0.8, seed=42)
                train_chunk = chunk_split["train"]
                test_chunk = chunk_split["test"]

                # Add to accumulators
                train_accumulator.append(train_chunk)
                test_accumulator.append(test_chunk)
                train_accumulator_size += len(train_chunk)
                test_accumulator_size += len(test_chunk)

                total_processed += len(chunk_dataset)
                chunks_processed += 1

                # Save conditions
                should_save_train = (
                    train_accumulator_size >= args.max_examples_per_file
                    or chunks_processed >= save_frequency_chunks
                )
                should_save_test = (
                    test_accumulator_size >= (args.max_examples_per_file // 4)
                    or chunks_processed >= save_frequency_chunks
                )

                # Save train data
                if should_save_train and train_accumulator:
                    train_files_created = save_in_chunks(
                        train_accumulator,
                        train_dir,
                        "train",
                        args.max_examples_per_file,
                        train_file_counter,
                    )
                    train_file_counter = train_files_created
                    train_accumulator = []
                    train_accumulator_size = 0

                # Save test data
                if should_save_test and test_accumulator:
                    test_files_created = save_in_chunks(
                        test_accumulator,
                        test_dir,
                        "test",
                        args.max_examples_per_file // 4,
                        test_file_counter,
                    )
                    test_file_counter += test_files_created
                    test_accumulator = []
                    test_accumulator_size = 0

                # Reset chunk counter if we saved
                if should_save_train or should_save_test:
                    chunks_processed = 0

                # Update progress
                if total_processed % (args.chunk_size * 2) == 0:
                    progress_info = {
                        "total_processed": total_processed,
                        "train_files_created": train_file_counter,
                        "test_files_created": test_file_counter,
                        "status": "in_progress",
                    }
                    with open(progress_file, "w") as f:
                        json.dump(progress_info, f, indent=2)
                    print(
                        f"📊 Progress: {total_processed} examples processed, {train_file_counter} train files, {test_file_counter} test files",
                        flush=True,
                    )

            # Save any remaining data
            if train_accumulator:
                train_files_created = save_in_chunks(
                    train_accumulator,
                    train_dir,
                    "train",
                    args.max_examples_per_file,
                    train_file_counter,
                )
                train_file_counter = train_files_created
                train_accumulator = []
                train_accumulator_size = 0

            if test_accumulator:
                test_files_created = save_in_chunks(
                    test_accumulator,
                    test_dir,
                    "test",
                    args.max_examples_per_file // 4,
                    test_file_counter,
                )
                test_file_counter += test_files_created
                test_accumulator = []
                test_accumulator_size = 0

            # Mark completion
            final_progress = {
                "total_processed": total_processed,
                "train_files_created": train_file_counter,
                "test_files_created": test_file_counter,
                "status": "completed",
            }
            with open(progress_file, "w") as f:
                json.dump(final_progress, f, indent=2)

            print(
                f"✅ Processing completed! {total_processed} examples in {train_file_counter} train files and {test_file_counter} test files",
                flush=True,
            )

        except Exception as e:
            print(f"❌ Error occurred: {e}", flush=True)
            # Save any accumulated data before crashing
            if train_accumulator:
                train_files_created = save_in_chunks(
                    train_accumulator,
                    train_dir,
                    "train",
                    args.max_examples_per_file,
                    train_file_counter,
                )
                train_file_counter = train_files_created
                train_accumulator = []
                train_accumulator_size = 0

            if test_accumulator:
                test_files_created = save_in_chunks(
                    test_accumulator,
                    test_dir,
                    "test",
                    args.max_examples_per_file // 4,
                    test_file_counter,
                )
                test_file_counter += test_files_created
                test_accumulator = []
                test_accumulator_size = 0

            # Save progress before crashing
            error_progress = {
                "total_processed": total_processed,
                "train_files_created": train_file_counter,
                "test_files_created": test_file_counter,
                "status": "error",
                "error_message": str(e),
            }
            with open(progress_file, "w") as f:
                json.dump(error_progress, f, indent=2)
            print(
                f"📊 Progress saved: {total_processed} examples processed", flush=True
            )
            raise
    else:
        dataset = dataset.map(
            function=make_map_fn("train"), with_indices=True, num_proc=16
        )
        dataset = dataset.cast_column("images", Sequence(ImageData()))

        local_dir = os.path.expanduser(args.local_dir)
        os.makedirs(local_dir, exist_ok=True)

        dataset.to_parquet(os.path.join(local_dir, f"{args.output_filename}.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
