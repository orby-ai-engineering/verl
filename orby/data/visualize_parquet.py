"""
Visualize parquet file as HTML table.

It will read the parquet file from s3 with only the max_rows rows.

Usage:
python visualize_parquet.py \
    --parquet_file /path/to/parquet_file_on_s3 \
    --output_file /path/to/local_output_file.html \
    --max_rows 10 \
    --image_column images
"""

import os
from io import BytesIO
import pandas as pd
import argparse
import base64
import json
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
import re
from transformers import AutoProcessor


MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
IMAGE_LOCATOR = "<|vision_start|><|image_pad|><|vision_end|>"


def image_to_base64(image_data):
    """
    Convert image data to base64 string for HTML display

    Args:
        image_data: Image data (could be bytes, PIL Image, or file path)

    Returns:
        str: Base64 encoded image string
    """
    try:
        if isinstance(image_data, bytes):
            # If it's already bytes, encode directly
            return [
                f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"
            ]
        elif isinstance(image_data, str):
            # If it's a file path, read and encode
            if os.path.exists(image_data):
                with open(image_data, "rb") as f:
                    return [
                        f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                    ]
            else:
                return None
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image_bytes = BytesIO(image_data["bytes"]).getvalue()
            return [
                f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            ]
        elif isinstance(image_data, list) or isinstance(image_data, np.ndarray):
            return [image_to_base64(img)[0] for img in image_data]
        else:
            print(type(image_data))
            raise ValueError("Invalid image data")
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def convert_ndarray_to_list(value):
    for k, v in value.items():
        if isinstance(v, np.ndarray):
            value[k] = v.tolist()
        elif isinstance(v, dict):
            value[k] = convert_ndarray_to_list(v)
        elif isinstance(v, list):
            value[k] = [convert_ndarray_to_list(item) for item in v]
    return value


def format_field_value(value):
    """
    Format a field value for HTML display

    Args:
        value: The value to format

    Returns:
        str: Formatted HTML string
    """
    if value is None:
        return "<em>None</em>"

    # Convert to JSON for better readability
    if isinstance(value, (dict, list)):
        if isinstance(value, dict):
            value = convert_ndarray_to_list(value)
        value = json.dumps(value, indent=2)
    elif isinstance(value, np.ndarray):
        value = value.tolist()
        value = json.dumps(value, indent=2)
    else:
        value = str(value)

    value = value.replace("<", "&lt;")
    value = value.replace(">", "&gt;")
    return value


def read_first_n_rows_pyarrow(filepath, n_rows):
    """
    Reads the first 'n_rows' from a Parquet file using pyarrow.
    """
    parquet_file = pq.ParquetFile(filepath)
    # Read the first 'n_rows' by iterating through batches
    # and stopping once 'n_rows' are collected.
    data = []
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=8):
        batch_df = batch.to_pandas()
        rows_to_add = min(n_rows - rows_read, len(batch_df))
        data.append(batch_df.head(rows_to_add))
        rows_read += rows_to_add
        if rows_read >= n_rows:
            break
    return pd.concat(data) if data else pd.DataFrame()


def generate_html_table(df, output_file, max_rows=50, image_column=None):
    """
    Generate an HTML table from parquet data

    Args:
        df: Pandas DataFrame
        output_file: Path to output HTML file
        max_rows: Maximum number of rows to display
        image_column: Name of column containing image data
    """
    # Limit rows for performance
    df_display = df.head(max_rows)

    # Start HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parquet Data Visualization</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            img {{ max-width: 600px; object-fit: contain; }}
            pre {{ white-space: pre-wrap; word-wrap: break-word; }}
            .prompt-cell {{ width: 50%; }}
            .other-cell {{ width: 50%; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>Parquet Data Visualization</h1>
        <p>Showing {rows} rows from {total_rows} total rows</p>
        <table>
            <thead>
                <tr>
                    <th class="prompt-cell">Input (Prompt)</th>
                    <th class="other-cell">Other Fields</th>
                </tr>
            </thead>
            <tbody>
    """.format(
        rows=len(df_display), total_rows=len(df)
    )

    # Add table rows
    for _, row in df_display.iterrows():
        html_content += "<tr>"

        # Process prompt and images
        messages = row["prompt"]
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

        prompt = PROCESSOR.apply_chat_template(messages, tokenize=False)

        # Replace image placeholders with actual images
        if image_column and image_column in row:
            image_data = row[image_column]
            image_base64 = image_to_base64(image_data)

            elements = prompt.split(IMAGE_LOCATOR)
            elements = ["<pre>" + element + "</pre>" for element in elements]
            prompt = IMAGE_LOCATOR.join(elements)
            if image_base64:
                for image_base64_item in image_base64:
                    image_base64_item = '<image src="' + image_base64_item + '">'
                    prompt = prompt.replace(IMAGE_LOCATOR, image_base64_item, 1)

        # Prompt column with images.
        html_content += '<td class="prompt-cell">'
        html_content += prompt
        html_content += "</td>"

        html_content += '<td class="other-cell">'

        # Show these fields first for easier reading.
        for col in [
            "data_source",
            "ability",
            "reward_score",
            "response",
            "predictions",
        ]:
            if col not in row:
                continue
            value = row[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            formatted_value = format_field_value(value)
            html_content += (
                f"<strong>{col}:</strong> <pre>{formatted_value}</pre><br><br>"
            )

        used_fields = [
            "data_source",
            "ability",
            "reward_score",
            "response",
            "predictions",
            "prompt",
            image_column,
        ]

        # Other other fields if any are left.
        for col in df.columns:
            if col in used_fields:
                continue
            value = row[col]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            formatted_value = format_field_value(value)
            html_content += (
                f"<strong>{col}:</strong> <pre>{formatted_value}</pre><br><br>"
            )
        html_content += "</td>"

        html_content += "</tr>"

    html_content += """
        </tbody>
    </table>
    </body>
    </html>
    """

    # Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML table saved to: {output_file}")


def visualize_parquet(parquet_file, output_file=None, max_rows=50, image_column=None):
    """
    Visualize parquet file as HTML table

    Args:
        parquet_file: Path to parquet file
        output_file: Path to output HTML file (optional)
        max_rows: Maximum number of rows to display
        image_column: Name of column containing image data
    """
    # Load the parquet file
    print(f"Loading parquet file: {parquet_file}")
    df = read_first_n_rows_pyarrow(parquet_file, max_rows)

    print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

    # Auto-detect image column if not specified
    if image_column is None:
        image_columns = [
            col for col in df.columns if "image" in col.lower() or "img" in col.lower()
        ]
        if image_columns:
            image_column = image_columns[0]
            print(f"Auto-detected image column: {image_column}")
        else:
            print(
                "No image column detected. Will display all columns in regular table format."
            )

    # Generate output filename if not provided
    if output_file is None:
        parquet_path = Path(parquet_file)
        output_file = parquet_path.with_suffix(".html")

    # Generate HTML table
    generate_html_table(df, output_file, max_rows, image_column)

    # Print summary
    print(f"\nSummary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Displayed rows: {min(len(df), max_rows)}")
    print(f"- Image column: {image_column}")
    print(f"- Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize parquet file as HTML table")
    parser.add_argument(
        "--parquet_file", required=True, help="Path to parquet file to visualize"
    )
    parser.add_argument(
        "--output_file",
        default="/tmp/vis.html",
        help="Path to output HTML file (optional)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10,
        help="Maximum number of rows to display (default: 50)",
    )
    parser.add_argument(
        "--image_column",
        default="images",
        help="Name of column containing image data (auto-detected if not specified)",
    )

    args = parser.parse_args()

    visualize_parquet(
        args.parquet_file, args.output_file, args.max_rows, args.image_column
    )
