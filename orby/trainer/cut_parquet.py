import os
import hydra
import pandas as pd
from verl.utils.fs import copy_to_local


def split_parquet_file(input_path, output_path_1, output_path_2, split_ratio=0.5):
    """
    Split a parquet file into two parts based on the given ratio.
    
    Args:
        input_path: Path to input parquet file
        output_path_1: Path for first output file
        output_path_2: Path for second output file  
        split_ratio: Ratio for first file (0.5 means 50% goes to first file)
    """
    # Validate paths to prevent data loss
    input_abs = os.path.abspath(input_path)
    output1_abs = os.path.abspath(output_path_1)
    output2_abs = os.path.abspath(output_path_2)
    
    # Check for the dangerous case where all paths are identical
    if input_abs == output1_abs == output2_abs:
        print(f"ERROR: All paths are identical. This will result in data loss!")
        print(f"Path: {input_abs}")
        print("Final result will contain only part 2 of the data.")
        raise ValueError("Input and both output paths cannot be the same. This would cause data loss.")
    
    # Warn about intentional overwrites (but allow them)
    if input_abs == output1_abs:
        print(f"INFO: Input file will be replaced with part 1 data ({split_ratio*100:.1f}% of original)")
        print(f"Path: {input_abs}")
    
    if input_abs == output2_abs:
        print(f"INFO: Input file will be replaced with part 2 data ({(1-split_ratio)*100:.1f}% of original)")
        print(f"Path: {input_abs}")
    
    print(f"Reading parquet file: {input_path}")
    
    # Read the parquet file
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    
    print(f"Total rows in input file: {total_rows}")
    
    # Shuffle the dataframe to ensure random distribution
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split point
    split_point = int(total_rows * split_ratio)
    
    # Split the data
    df_part1 = df_shuffled[:split_point]
    df_part2 = df_shuffled[split_point:]
    
    print(f"Split ratio: {split_ratio}")
    print(f"Part 1 rows: {len(df_part1)} (will be saved to {output_path_1})")
    print(f"Part 2 rows: {len(df_part2)} (will be saved to {output_path_2})")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_path_1), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_2), exist_ok=True)
    
    # Save the split parts
    df_part1.to_parquet(output_path_1, index=False)
    df_part2.to_parquet(output_path_2, index=False)
    
    print(f"Successfully split parquet file:")
    print(f"  {output_path_1}: {len(df_part1)} rows")
    print(f"  {output_path_2}: {len(df_part2)} rows")


@hydra.main(
    config_path="config", config_name="cut_parquet", version_base=None
)
def main(config):
    # Get local copy of input file
    local_input_path = copy_to_local(config.data.path)
    
    # Get configuration parameters
    output_path_1 = config.data.output_path_1
    output_path_2 = config.data.output_path_2
    split_ratio = config.data.split_ratio
    
    # Perform the split
    split_parquet_file(
        local_input_path,
        output_path_1, 
        output_path_2,
        split_ratio
    )


if __name__ == "__main__":
    main()
