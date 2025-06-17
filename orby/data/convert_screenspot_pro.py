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
Preprocess the Screenspot Pro dataset to parquet format
"""

import argparse
import io
import json
import os
import logging
import glob

import datasets
from datasets import Sequence
from datasets import Image as ImageData
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import smart_resize
import pyarrow.parquet as pq
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from orby.utils.dataset.qwen_agent_function_call import ComputerUse

from verl.utils.hdfs_io import copy, makedirs

MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)

_PLATFORM_MAP = {
    "ios": "mobile",
    "android": "mobile",
    "windows": "desktop",
    "macos": "desktop",
    "linux": "desktop",
    "web": "web",
}

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


def get_resized_ratio(image):
    """
    Get the resize ratios for width and height of the image.

    Returns:
        Tuple of (height_ratio, width_ratio) where each ratio is the resized dimension
        divided by the original dimension.
    """
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=PROCESSOR.image_processor.patch_size
        * PROCESSOR.image_processor.merge_size,
        min_pixels=PROCESSOR.image_processor.min_pixels,
        max_pixels=PROCESSOR.image_processor.max_pixels,
    )

    height_ratio = resized_height / image.height
    width_ratio = resized_width / image.width

    return height_ratio, width_ratio


def process_json_file(json_path, image_dir, split, prompt_format="thinking"):
    """
    Process a single JSON file and return a list of processed examples.

    Args:
        json_path: Path to the JSON file
        image_dir: Directory containing the images
        split: Dataset split name (e.g., "train", "test")
        prompt_format: Format of the prompt ("thinking" or "qwen")

    Returns:
        List of processed examples
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    processed_examples = []
    for idx, example in enumerate(data):
        # Load image from file
        img_path = os.path.join(image_dir, example["img_filename"])
        try:
            image = Image.open(img_path)
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or "PNG")
            img_byte_arr = img_byte_arr.getvalue()
        except Exception as e:
            logging.warning(f"Failed to load image {img_path}: {e}")
            continue

        # Get image resize ratios
        height_ratio, width_ratio = get_resized_ratio(image)

        # Adjust bbox based on resize ratios
        bbox = example["bbox"]
        bbox = [
            bbox[0] * width_ratio,
            bbox[1] * height_ratio,
            bbox[2] * width_ratio,
            bbox[3] * height_ratio,
        ]

        device = _PLATFORM_MAP.get(example["platform"], "web")

        ground_truth = {
            "bbox": bbox,
            "data_type": example["ui_type"],
            "device": device,
            "application": example["application"],
        }
        data = {
            "data_source": "screenspot_pro",
            "images": [img_byte_arr],
            "ability": "vision",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "question": example["instruction"],
                "bounding_box": bbox,
                "application": example["application"],
                "platform": example["platform"],
                "ui_type": example["ui_type"],
            },
        }

        # Create prompt based on selected format
        if prompt_format == "thinking":
            data["prompt"] = [
                {
                    "role": "user",
                    "content": (
                        "Map the user instruction to the coordinates in the UI image. "
                        "Think step by step before you answer. The reasoning process MUST BE enclosed within <think> </think> tags. "
                        "The coordinate x and y MUST BE put in <answer> </answer> tags, separeted by space. "
                        "<image> Instruction: " + example["instruction"]
                    ),
                },
            ]
        elif prompt_format == "sft":
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
                        f"We are trying to complete the following tasks: {example['instruction']} "
                        "If a previous action failed or has been repeated multiple times without a positive outcome, please avoid repeating the same mistakes. "
                        "Try to use a completely different approach. "
                        "The previous action can fail either because the action itself is inappropriate, or because the coordinates of the elements are not correct. "
                        "Adjust your next action accordingly. "
                        "Here is the current screenshot of the webpage, which you can interact with using the actions and which you should remember coordinates for elements: "
                        "<image:current_screenshot> "
                        "Pixel coordinates originate from the top left corner of the image, where the first coordinate refers to the horizontal/width axis and the second refers to the vertical/height axis. "
                        "Important: explore, explore, explore! The screenshot is not the entire webpage and you need to scroll to determine if a task is completable or whether you have gathered all the information you need. "
                        f"Again, our goal is: {example['instruction']} "
                        "<image> Instruction: " + example["instruction"]
                    ),
                },
            ]
        elif prompt_format == "qwen":  # qwen format
            prompt = NousFnCallPrompt().preprocess_fncall_messages(
                messages=[
                    Message(
                        role="system",
                        content=[ContentItem(text="You are a helpful assistant.")],
                    ),
                    Message(
                        role="user",
                        content=[
                            ContentItem(text=example["instruction"] + "<image>"),
                        ],
                    ),
                ],
                functions=[
                    ComputerUse(
                        cfg={
                            "display_width_px": int(image.width * width_ratio),
                            "display_height_px": int(image.height * height_ratio),
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

        processed_examples.append(data)

    return processed_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/screenspot_pro")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--image_dir",
        default="~/data/screenspot_pro/images",
        help="Directory containing the images",
    )
    parser.add_argument(
        "--annotations_dir",
        default="~/data/screenspot_pro/annotations",
        help="Directory containing the annotation JSON files",
    )
    parser.add_argument(
        "--prompt_format",
        choices=["thinking", "qwen", "sft"],
        default="thinking",
        help="Select prompt format: ['thinking', 'qwen', 'sft']",
    )

    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    image_dir = os.path.expanduser(args.image_dir)
    annotations_dir = os.path.expanduser(args.annotations_dir)
    os.makedirs(local_dir, exist_ok=True)

    # Find all JSON files in the annotations directory
    json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in {annotations_dir}")

    all_examples = []
    shard = 0
    for json_file in json_files:
        logging.info(f"Processing {json_file}")
        examples = process_json_file(json_file, image_dir, "test", args.prompt_format)
        all_examples.extend(examples)
        if len(all_examples) > 750:
            # Convert to dataset
            dataset = datasets.Dataset.from_list(all_examples)
            dataset = dataset.cast_column("images", Sequence(ImageData()))

            # Save to parquet
            dataset.to_parquet(os.path.join(local_dir, f"test-{shard}.parquet"))
            shard += 1
            all_examples = []

    # Convert the final shard.
    if len(all_examples):
        # Convert to dataset
        dataset = datasets.Dataset.from_list(all_examples)
        dataset = dataset.cast_column("images", Sequence(ImageData()))

        # Save to parquet
        dataset.to_parquet(os.path.join(local_dir, f"test-{shard}.parquet"))

    files = [os.path.join(local_dir, f"test-{shard}.parquet") for shard in range(shard)]

    schema = pq.ParquetFile(files[0]).schema_arrow
    with pq.ParquetWriter(
        os.path.join(local_dir, "test.parquet"), schema=schema
    ) as writer:
        for file in files:
            writer.write_table(pq.read_table(file, schema=schema))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
