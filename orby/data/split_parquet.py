import math
import argparse
import os
import pandas as pd


def split_parquet(input_file, output_dir, num_to_divide=1000):
    """
    Split a parquet file into smaller chunks.

    Args:
        input_file (str): Path to the input parquet file
        output_dir (str): Directory to save the split files
        num_to_divide (int): Number of files to split into
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the parquet file
    df = pd.read_parquet(input_file)

    num_rows = df.shape[0]
    split_size = math.ceil(num_rows / num_to_divide)

    print(f"Splitting {input_file} with {num_rows} rows into {num_to_divide} files")
    print(f"Each file will contain approximately {split_size} rows")

    for i in range(0, num_rows, split_size):
        df_subset = df.iloc[i : i + split_size]
        output_file = os.path.join(output_dir, f"split_{i//split_size:04d}.parquet")
        df_subset.to_parquet(output_file, row_group_size=512, engine="pyarrow")
        print(f"Saved {len(df_subset)} rows to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a parquet file into smaller chunks"
    )
    parser.add_argument(
        "--input_file", required=True, help="Path to the input parquet file"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the split files"
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=10,
        help="Number of files to split into (default: 1000)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        exit(1)

    split_parquet(args.input_file, args.output_dir, args.num_splits)
