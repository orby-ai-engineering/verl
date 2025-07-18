import ast
import os
import hydra
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from verl.utils.fs import copy_to_local


def extract_should_end_values(df, should_end_column):
    """Extract should_end values from nested dictionary structure."""
    should_end_values = []
    
    for _, row in df.iterrows():
        if '.' in should_end_column:
            # Handle nested access like "reward_model.ground_truth.should_end"
            parts = should_end_column.split('.')
            value = row[parts[0]]  # Get the first column value (should be a dict)
            
            # Navigate through the remaining nested structure
            for part in parts[1:]:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            should_end_values.append(value)
        else:
            # Direct column access
            should_end_values.append(row.get(should_end_column))
    
    return should_end_values


def filter_parquet_chunks(
    input_path,
    output_path,
    filter_bounds,
    balance_should_end=True,
    should_end_column="reward_model.ground_truth.should_end",
    reward_score_column="reward_score",
    chunk_size=1000
):
    """Filter parquet file in chunks to handle large files efficiently."""
    # Read parquet file info
    parquet_file = pq.ParquetFile(input_path)
    
    # Initialize collections for all data
    all_filtered_data = []
    total_rows = 0
    
    # For balancing logic
    should_end_true_count = 0
    should_end_false_count = 0
    balancing_data = []

    # First pass: Filter data and collect balancing statistics
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        # Convert to pandas for easier filtering
        df_chunk = batch.to_pandas()
        
        if total_rows == 0:
            # Check for reward_score column
            if reward_score_column not in df_chunk.columns:
                raise ValueError(f"No '{reward_score_column}' column found in dataset. Make sure main_eval was run with save_scores=True")
            
            # Check for nested should_end column if balancing is enabled
            if balance_should_end and should_end_column:
                # For nested columns, check if the root column exists
                root_column = should_end_column.split('.')[0]
                if root_column not in df_chunk.columns:
                    print(f"Warning: root column '{root_column}' for should_end not found. Skipping balancing.")
                    balance_should_end = False
            
            print(f"Filtering based on column: {reward_score_column}")
            if balance_should_end:
                print(f"Balancing enabled using nested column: {should_end_column}")
        
        # Filter chunk
        lower_bound, upper_bound = filter_bounds
        mask = (df_chunk[reward_score_column] >= lower_bound) & (df_chunk[reward_score_column] <= upper_bound)
        filtered_chunk = df_chunk[mask]
        
        # Count should_end statistics for balancing
        if balance_should_end and should_end_column:
            # Extract should_end values from nested structure
            filtered_should_end_values = extract_should_end_values(filtered_chunk, should_end_column)
            should_end_true_in_chunk = sum(1 for val in filtered_should_end_values if val == "true")
            should_end_false_in_chunk = sum(1 for val in filtered_should_end_values if val == "false")
            should_end_true_count += should_end_true_in_chunk
            should_end_false_count += should_end_false_in_chunk
            
            # Collect potential balancing data (should_end == false that didn't meet filter criteria)
            chunk_should_end_values = extract_should_end_values(df_chunk, should_end_column)
            balancing_mask = []
            for i, (meets_filter, should_end_val) in enumerate(zip(mask, chunk_should_end_values)):
                balancing_mask.append(should_end_val == "false" and not meets_filter)
            
            balancing_chunk = df_chunk[balancing_mask]
            if len(balancing_chunk) > 0:
                balancing_data.append(balancing_chunk)
        
        # Collect filtered data instead of writing immediately
        if len(filtered_chunk) > 0:
            all_filtered_data.append(filtered_chunk.reset_index(drop=True))
        
        total_rows += len(df_chunk)
        
        if total_rows % 5000 == 0:
            print(f"Processed {total_rows} rows, collected {sum(len(df) for df in all_filtered_data)} filtered rows")
    
    # Prepare final dataset with balancing
    final_datasets = []
    
    # Add all filtered data
    if all_filtered_data:
        final_datasets.extend(all_filtered_data)
    
    # Add balancing data if needed
    if balance_should_end and should_end_column and balancing_data:
        needed_false_count = should_end_true_count - should_end_false_count
        if needed_false_count > 0:
            print(f"Balancing dataset: should_end true={should_end_true_count}, false={should_end_false_count}")
            print(f"Adding {needed_false_count} should_end=false rows for balance")
            
            # Concatenate all balancing data
            all_balancing_data = pd.concat(balancing_data, ignore_index=True)
            
            # Sample the needed amount (or all if we don't have enough)
            if len(all_balancing_data) >= needed_false_count:
                balancing_sample = all_balancing_data.sample(n=needed_false_count, random_state=42)
            else:
                balancing_sample = all_balancing_data
                print(f"Warning: Only {len(all_balancing_data)} balancing rows available, using all")
            
            # Add balancing data to final dataset
            if len(balancing_sample) > 0:
                final_datasets.append(balancing_sample.reset_index(drop=True))
        else:
            print(f"No balancing needed: should_end true={should_end_true_count}, false={should_end_false_count}")
    
    # Combine all data and shuffle
    combined_data = pd.concat(final_datasets, ignore_index=True)
    
    # Shuffle the combined dataset
    shuffled_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Write shuffled data to output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shuffled_data.to_parquet(output_path, index=False, row_group_size=512)
    filtered_rows = len(shuffled_data)
    print(f"Filtering complete: {filtered_rows}/{total_rows} rows kept")


@hydra.main(
    config_path="config", config_name="reward_filter", version_base=None
)
def main(config):
    # Get local copy of input file
    local_input_path = copy_to_local(config.data.path)

    # Parse filter bounds
    medium_bounds = (
        config.data.medium_difficulty_filter_lower_bound,
        config.data.medium_difficulty_filter_upper_bound
    )
    hard_bounds = (
        config.data.hard_difficulty_filter_lower_bound,
        config.data.hard_difficulty_filter_upper_bound
    )

    # Get balancing configuration
    balance_should_end = config.data.get("balance_should_end", True)
    should_end_column = config.data.get("should_end_column", "reward_model.ground_truth.should_end")
    reward_score_column = config.data.get("reward_score_column", "reward_score")
    
    # Filter for medium difficulty
    medium_output = config.data.medium_difficulty_output_path
    print(f"Filtering medium difficulty data with bounds {medium_bounds}")
    filter_parquet_chunks(
        local_input_path,
        medium_output,
        medium_bounds,
        balance_should_end,
        should_end_column,
        reward_score_column
    )
    
    # Filter for hard difficulty  
    hard_output = config.data.hard_difficulty_output_path
    print(f"Filtering hard difficulty data with bounds {hard_bounds}")
    filter_parquet_chunks(
        local_input_path,
        hard_output,
        hard_bounds,
        balance_should_end,
        should_end_column,
        reward_score_column
    )


if __name__ == "__main__":
    main()
