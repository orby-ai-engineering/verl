import argparse
import os
import pandas as pd
import numpy as np
import random
import re

random.seed(42)


def extract_goal_from_prompt(prompt):
    """
    Extract goal from prompt in format "I am trying to complete the following task: goal"

    Args:
        prompt (str): The prompt string

    Returns:
        str: The extracted goal, or None if not found
    """
    pattern = r"I am trying to complete the following task:\s*(.+)"
    match = re.search(pattern, prompt)
    if match:
        return match.group(1).strip()
    return None


def split_parquet_by_goal(goals_file, data_file, output_dir):
    """
    Split a parquet file into train/test sets based on goals.

    Args:
        goals_file (str): Path to the parquet file containing goals
        data_file (str): Path to the parquet file containing the main dataset
        output_dir (str): Directory to save the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load goals from the goals file
    print(f"Loading goals from {goals_file}")
    goals_df = pd.read_parquet(goals_file)

    if "goal" not in goals_df.columns:
        raise ValueError("Goals file must contain a 'goal' column")

    # Get unique goals
    unique_goals = goals_df["goal"].unique()
    print(f"Found {len(unique_goals)} unique goals")
    unique_goals = list(unique_goals)
    random.shuffle(unique_goals)
    goal_split_1 = unique_goals[: len(unique_goals) // 2]
    goal_split_2 = unique_goals[len(unique_goals) // 2 :]

    print(f"Split goals: {len(goal_split_1)} split_1, {len(goal_split_2)} split_2")

    # Load the main dataset
    print(f"Loading main dataset from {data_file}")
    data_df = pd.read_parquet(data_file)

    if "prompt" not in data_df.columns:
        raise ValueError("Data file must contain a 'prompt' column")

    print(f"Main dataset has {len(data_df)} rows")

    # Extract goals from prompts
    print("Extracting goals from prompts...")
    data_df["extracted_goal"] = data_df["prompt"].apply(extract_goal_from_prompt)

    # Count how many goals were successfully extracted
    extracted_count = data_df["extracted_goal"].notna().sum()
    print(
        f"Successfully extracted goals from {extracted_count} out of {len(data_df)} rows"
    )

    # Filter data based on goal sets
    split_1_data = data_df[data_df["extracted_goal"].isin(goal_split_1)]
    split_2_data = data_df[data_df["extracted_goal"].isin(goal_split_2)]

    print(
        f"Filtered data: {len(split_1_data)} split_1 rows, {len(split_2_data)} split_2 rows"
    )

    # Save the split datasets
    split_1_file = os.path.join(output_dir, "split_1.parquet")
    split_2_file = os.path.join(output_dir, "split_2.parquet")

    split_1_data.to_parquet(split_1_file, row_group_size=512, engine="pyarrow")
    split_2_data.to_parquet(split_2_file, row_group_size=512, engine="pyarrow")

    print(f"Saved split_1 data to {split_1_file}")
    print(f"Saved split_2 data to {split_2_file}")

    # Save goal sets for reference
    split_1_goals_file = os.path.join(output_dir, "split_1_goals.txt")
    split_2_goals_file = os.path.join(output_dir, "split_2_goals.txt")

    with open(split_1_goals_file, "w") as f:
        for goal in sorted(goal_split_1):
            f.write(f"{goal}\n")

    with open(split_2_goals_file, "w") as f:
        for goal in sorted(goal_split_2):
            f.write(f"{goal}\n")

    print(f"Saved split_1 goals to {split_1_goals_file}")
    print(f"Saved split_2 goals to {split_2_goals_file}")

    return split_1_file, split_2_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a parquet file into train/test sets based on goals"
    )
    parser.add_argument(
        "--goals_file", required=True, help="Path to the parquet file containing goals"
    )
    parser.add_argument(
        "--data_file",
        required=True,
        help="Path to the parquet file containing the main dataset with prompts",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save the split files"
    )

    args = parser.parse_args()

    # Check if input files exist
    if not os.path.exists(args.goals_file):
        print(f"Error: Goals file '{args.goals_file}' does not exist")
        exit(1)

    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' does not exist")
        exit(1)

    try:
        split_parquet_by_goal(args.goals_file, args.data_file, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
