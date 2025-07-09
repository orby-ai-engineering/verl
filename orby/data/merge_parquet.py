import argparse
import os
import glob
import pandas as pd


def merge_parquet_files(input_dir, output_file, pattern="*.parquet"):
    """
    Merge multiple parquet files into a single parquet file.

    Args:
        input_dir (str): Directory containing the parquet files to merge
        output_file (str): Path to the output merged parquet file
        pattern (str): File pattern to match parquet files (default: "*.parquet")
    """
    # Find all parquet files in the input directory
    search_pattern = os.path.join(input_dir, pattern)
    parquet_files = glob.glob(search_pattern)

    if not parquet_files:
        print(f"No parquet files found in {input_dir} matching pattern '{pattern}'")
        return

    # Sort files to ensure consistent ordering
    parquet_files.sort()

    print(f"Found {len(parquet_files)} parquet files to merge:")
    for file in parquet_files:
        print(f"  - {os.path.basename(file)}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read and concatenate all parquet files
    print("Reading and merging parquet files...")
    dataframes = []
    total_rows = 0

    for i, file in enumerate(parquet_files):
        print(f"Reading file {i+1}/{len(parquet_files)}: {os.path.basename(file)}")
        df = pd.read_parquet(file)
        dataframes.append(df)
        total_rows += len(df)
        print(f"  - Added {len(df)} rows")

    # Concatenate all dataframes
    print("Concatenating dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Expected total rows: {total_rows}")

    # Save the merged dataframe
    print(f"Saving merged file to: {output_file}")
    merged_df.to_parquet(output_file, row_group_size=512, engine="pyarrow")
    print(f"Successfully merged {len(parquet_files)} files into {output_file}")


def merge_parquet_files_by_list(input_files, output_file):
    """
    Merge specific parquet files into a single parquet file.

    Args:
        input_files (list): List of paths to parquet files to merge
        output_file (str): Path to the output merged parquet file
    """
    if not input_files:
        print("No input files specified")
        return

    # Check if all input files exist
    for file in input_files:
        if not os.path.exists(file):
            print(f"Error: Input file '{file}' does not exist")
            return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(input_files)} parquet files to merge:")
    for file in input_files:
        print(f"  - {file}")

    # Read and concatenate all parquet files
    print("Reading and merging parquet files...")
    dataframes = []
    total_rows = 0

    for i, file in enumerate(input_files):
        print(f"Reading file {i+1}/{len(input_files)}: {os.path.basename(file)}")
        df = pd.read_parquet(file)
        dataframes.append(df)
        total_rows += len(df)
        print(f"  - Added {len(df)} rows")

    # Concatenate all dataframes
    print("Concatenating dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Expected total rows: {total_rows}")

    # Save the merged dataframe
    print(f"Saving merged file to: {output_file}")
    merged_df.to_parquet(output_file, row_group_size=512, engine="pyarrow")
    print(f"Successfully merged {len(input_files)} files into {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple parquet files into a single file"
    )

    # Create mutually exclusive group for input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir", help="Directory containing parquet files to merge"
    )
    input_group.add_argument(
        "--input_files", nargs="+", help="List of specific parquet files to merge"
    )

    parser.add_argument(
        "--output_file", required=True, help="Path to the output merged parquet file"
    )
    parser.add_argument(
        "--pattern",
        default="*.parquet",
        help="File pattern to match parquet files when using --input_dir (default: '*.parquet')",
    )

    args = parser.parse_args()

    if args.input_dir:
        # Merge files from directory
        merge_parquet_files(args.input_dir, args.output_file, args.pattern)
    elif args.input_files:
        # Merge specific files
        merge_parquet_files_by_list(args.input_files, args.output_file)
