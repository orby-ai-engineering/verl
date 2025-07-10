import os
import pandas as pd
import argparse
from PIL import Image, ImageDraw
import io

# Set pandas display options to show full content
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_rows', None)

def visualize_parquet(parquet_file):
    # Load the parquet file
    df = pd.read_parquet(parquet_file)
    print(f"\nParquet file contents ({len(df)} rows):")
    print("\nColumns:", df.columns.tolist())
    
    # Print first few rows
    print("\nFirst few rows:")
    print(df.head())
    
     # Print detailed prompt content for first few examples
    print("\n" + "="*80)
    print("DETAILED PROMPT CONTENT")
    print("="*80)
    
    idx = 0

    for idx in range(min(len(df), 20)):
        # Show first examples
        print(f"\n{'='*20} EXAMPLE {idx+1} {'='*20}")
        
        # Get the prompt
        prompt = df.iloc[idx]['prompt']


        print(f"\nPROMPT MESSAGES ({len(prompt)} messages):")

        for i, message in enumerate(prompt):
            print(f"\n--- MESSAGE {i+1} ---")
            print(message)
            print("-"*80)
        
        print("\n" + "="*60)
        # Show response if available
        if 'response' in df.iloc[idx]:
            print("\n--- RESPONSE ---")
            print(df.iloc[idx]['response'])
            print("-"*80)
        if 'reward_model' in df.iloc[idx]:
            print("\n--- REWARD MODEL ---")
            print(df.iloc[idx]['reward_model'])
            print("-"*80) 

    # Show extra info
    extra_info = df.iloc[idx]['extra_info']
    print(f"\n--- EXTRA INFO ---")
    print(f"Question: {extra_info['question']}")
    print(f"Bounding Box: {extra_info['bounding_box']}")
    print(f"Max Pixels: {extra_info['max_pixels']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_file", 
    default="~/data/uground/subtask/0.50k/train/train_part_0000.parquet",
    help="Path to parquet file to visualize")
    args = parser.parse_args()
    
    visualize_parquet(args.parquet_file)
