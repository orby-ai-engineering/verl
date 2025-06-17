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


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)


ACTION_HINTS = """

click(x: float, y: float, button: Literal['left', 'right'] = 'left', double: bool = False)
    Move the mouse to a location and click a mouse button.
    Can be used to click a button, select a checkbox, focus on a input field, etc.
    Args:
        x (float): The x coordinate of the location to click.
        y (float): The y coordinate of the location to click.
        button (Literal["left", "right"]): The button to click.
        double (bool): Whether to double click.
    Examples:
        click(324.5, 12)
        click(119, 34, button="right")
        click(34.1, 720, double=True)
        click(230, 100, button="left", double=False)

complete(answer: str = '', infeasible_reason: str = '')
    Complete the task and optionally provide the user some feedback.
    Fill in the answer if the completion of the task requires providing a response to the user.
    Fill in the infeasible_reason if the task is infeasible.
    DO NOT fill in both answer and infeasible_reason at the same time.
    Args:
        answer (str): The answer to the task, if any.
        infeasible_reason (str): The reason the task is infeasible, if any.
    Examples:
        complete(answer="To request a refund, you need to call the customer service at 123-456-7890.")
        complete(infeasible_reason="The task is infeasible because the user has not provided a valid email address.")
        complete()
        complete(answer="{\n  "name": "John",\n  "age": 30,\n  "city": "New York"\n}")

drag_and_release(x1: float, y1: float, x2: float, y2: float)
    Press down the left mouse button at a location, drag the mouse to another location, and release the mouse button.
    Can be used for selecting a section of text, dragging a slider, etc.
    Args:
        x1 (float): The x coordinate of the location to press down the left mouse button.
        y1 (float): The y coordinate of the location to press down the left mouse button.
        x2 (float): The x coordinate of the location to release the left mouse button.
        y2 (float): The y coordinate of the location to release the left mouse button.
    Examples:
        drag_and_release(10.5, 200, 10.5, 230)

hover(x: float, y: float)
    Move the mouse to a location and stay there.
    Can be used to focus on a location, pop up a tooltip, navigate a dropdown menu, etc.
    Args:
        x (float): The x coordinate of the location to hover over.
        y (float): The y coordinate of the location to hover over.
    Examples:
        hover(102, 720)

key_press(keys: list[str])
    Press one or a combination of keys at the same time on the keyboard.
    Can be used
    - As various shortcuts of the current environment (e.g. ["Control", "a"], ["Control", "f"]).
    - To complete a search with ["Enter"].
    - And any other common actions that can be performed with a keyboard in the relevant application.
    This should NOT be used to type a string of text. Use the type action for that.
    The list of allowed keys are:
    - F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    - 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    - a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
    - Backspace, Tab, Enter, Shift, Control, Alt, Delete
    - ArrowUp, ArrowDown, ArrowLeft, ArrowRight
    - Home, End, PageUp, PageDown
    Args:
        keys (list[str]): The list of keys to press.
    Examples:
        key_press(["Control", "a"]) # Select all
        key_press(["Control", "f"]) # Open the search bar
        key_press(["Enter"]) # Submit a form
        key_press(["F12"]) # Open the developer tools in a browser

scroll(x: float, y: float, delta_x: float = 0, delta_y: float = 100)
    Move the mouse to a location and scroll the mouse wheel in the x and y directions.
    Can be used to scroll a webpage, scroll a dropdown menu, etc.
    Args:
        x (float): The x coordinate of the location to scroll over.
        y (float): The y coordinate of the location to scroll over.
        delta_x (float): The amount to scroll horizontally.
        delta_y (float): The amount to scroll vertically.
    Examples:
        scroll(102, 320)
        scroll(102, 320, 0, 200)
        scroll(90, 32.7, 0, -300)
        scroll(620, 105, 68, 49.6)

type(x: float, y: float, text: str)
    Focus on a location and type a string of text in it.
    Can be used to type in a text field, search bar, edit a document, etc.
    Args:
        x (float): The x coordinate of the location to type text in.
        y (float): The y coordinate of the location to type text in.
        text (str): The text to type.
    Examples:
        type(102, 70.6, "\nThank you for the coffee!\n")
        type(44, 120, "Best sellers")

wait(ms: int = 1000)
    Wait for a specified amount of time.
    Can be used to wait for a webpage to load, a long form to display, etc.
    Args:
        ms (int): The amount of time to wait in milliseconds.
    Examples:
        wait()
        wait(1000)
        wait(500)
"""


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


def process_in_chunks(streaming_dataset, chunk_size=1000):
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
                        "content": (
                            "You are a powerful and precise web agent, helping a user with their web-related tasks. "
                            "Here is the set of actions you can take on a webpage, which you can call as python functions. "
                            "It is a documentation of all the functions, and it's important that you read it well and carefully: "
                            f"{ACTION_HINTS} "
                            "The user will provide the screenshot of the webpage, and you will need to use the actions to complete the task. "
                            "Your outputs should be single python function calls, without any additional text, and should follow the correct formatting given above. "
                            "Refer to the documentation to determine appropriate args. "
                            "DO NOT PROVIDE MORE THAN ONE FUNCTION CALL AT EACH TURN! "
                            "You will also be given the history of your previous thoughts and actions, use this to correct your trajectory intelligently. "
                            "Human:Please help me! "
                            f"We are trying to complete the following tasks: {instruction} "
                            "If a previous action failed or has been repeated multiple times without a positive outcome, please avoid repeating the same mistakes. "
                            "Try to use a completely different approach. "
                            "The previous action can fail either because the action itself is inappropriate, or because the coordinates of the elements are not correct. "
                            "Adjust your next action accordingly. "
                            "Here is the current screenshot of the webpage, which you can interact with using the actions and which you should remember coordinates for elements: "
                            "<image:current_screenshot> "
                            "Pixel coordinates originate from the top left corner of the image, where the first coordinate refers to the horizontal/width axis and the second refers to the vertical/height axis. "
                            "Important: explore, explore, explore! The screenshot is not the entire webpage and you need to scroll to determine if a task is completable or whether you have gathered all the information you need. "
                            f"Again, our goal is: {instruction} "
                            "<image> Instruction: " + instruction
                        ),
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
                    train_file_counter = (
                        train_files_created  # Update counter for next call
                    )
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
