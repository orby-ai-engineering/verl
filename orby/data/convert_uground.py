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

    file_counter = start_file_counter  # Start from provided counter instead of 0

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

    return file_counter  # Return the next counter value


def process_in_chunks(streaming_dataset, chunk_size=5000):
    """Process streaming dataset in chunks with immediate saving capability"""
    chunk = []
    total_processed = 0

    # Add progress tracking variables
    progress_file = None

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
    parser.add_argument("--local_dir", default="~/data/uground")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--data_files", default="shard_*.parquet")
    parser.add_argument("--output_filename", default="train")
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

    data_source = "osunlp/UGround-V1-Data-Box"
    print(
        f"Loading the {data_source} dataset from huggingface in streaming mode...",
        flush=True,
    )

    # Load in streaming mode
    dataset = datasets.load_dataset(
        data_source, data_files=args.data_files, streaming=True
    )
    dataset = dataset["train"]

    def make_map_fn(split):
        def process_fn(example, idx):
            image = example.pop("image")
            conversation = example.pop("conversations").strip()
            # Use the first message for now. Uground has multiple grounding
            # commands / groundtruths in the conversation.
            command, label = json.loads(conversation)[:2]
            assert command["from"] == "human" and label["from"] == "gpt"
            instruction = command["value"]
            label_text = label["value"]

            # Parse the label text as "(x1, y1, x2, y2)" format
            label_text = label_text.strip("()")
            bbox = [int(x.strip()) for x in label_text.split(",")]
            assert len(bbox) == 4, f"Expected 4 coordinates, got {len(bbox)}"

            # Get image and resize ratios
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            resized_height, resized_width = get_resized_wh(image)

            # Adjust bbox based on resize ratios. Uground labels range from
            # [0, 999]
            bbox = [
                bbox[0] * resized_width / 999.0,
                bbox[1] * resized_height / 999.0,
                bbox[2] * resized_width / 999.0,
                bbox[3] * resized_height / 999.0,
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
                    "answer": answer
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
                        "content": ("<image> Instruction: " + example["instruction"]),
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


    print(f"Saving to {local_dir}...", flush=True)
    os.makedirs(local_dir, exist_ok=True)

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

    # Calculate save frequency - save every few chunks or when reaching file size limit
    save_frequency_chunks = max(
        1, args.max_examples_per_file // (args.chunk_size * 4)
    )  # Save every N chunks
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

            # Save conditions: either reached file size limit OR processed enough chunks for fault tolerance
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
                train_file_counter = train_files_created  # Update counter for next call
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
                test_file_counter = test_files_created
                test_accumulator = []
                test_accumulator_size = 0

            # Reset chunk counter if we saved
            if should_save_train or should_save_test:
                chunks_processed = 0

            # Update progress every few chunks
            if (
                total_processed % (args.chunk_size * 2) == 0
            ):  # Every 2 chunks instead of 5
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

        # Save any remaining data (this will definitely run for your 2000 example test)
        if train_accumulator:
            train_files_created = save_in_chunks(
                train_accumulator,
                train_dir,
                "train",
                args.max_examples_per_file,
                train_file_counter,
            )
            train_file_counter = train_files_created  # Update counter for next call
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
            test_file_counter = test_files_created
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
            train_file_counter = train_files_created  # Update counter for next call
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
            test_file_counter = test_files_created
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

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
