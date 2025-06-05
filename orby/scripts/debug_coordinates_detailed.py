# Save this as debug_coordinates_detailed.py
import pandas as pd
import re

# Load the output file
df = pd.read_parquet('/root/data/screenspot/result-test-output-1.parquet')

print("=== DETAILED COORDINATE EXTRACTION DEBUG ===\n")

# Recreate the exact patterns from the reward function
thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
coordinate_pattern = re.compile(r"(\d*\.?\d+)\s+(\d*\.?\d+)")

def debug_coordinate_extraction(prediction):
    """Debug the exact same logic as the reward function"""
    print(f"Input prediction: {repr(prediction[:200])}...")
    
    # Step 1: Check thinking
    has_thinking = bool(thinking_pattern.search(prediction))
    print(f"Has thinking: {has_thinking}")
    
    # Step 2: Extract answer
    answer_match = answer_pattern.search(prediction)
    has_answer = bool(answer_match)
    print(f"Has answer: {has_answer}")
    
    if answer_match:
        answer_str = answer_match.group(1).strip()
        print(f"Answer string: '{answer_str}'")
        
        # Step 3: Extract coordinates - THIS IS THE CRITICAL PART
        coord_match = coordinate_pattern.match(answer_str.strip())  # Note: .match() not .search()!
        print(f"Coordinate pattern match: {coord_match}")
        
        if coord_match:
            x, y = coord_match.groups()
            print(f"Extracted coordinates: x={x}, y={y}")
            return float(x), float(y)
        else:
            print("❌ COORDINATE PATTERN FAILED!")
            print(f"Trying .search() instead of .match(): {coordinate_pattern.search(answer_str.strip())}")
            return None, None
    else:
        print("❌ NO ANSWER TAG FOUND!")
        return None, None

# Test on first few examples
print("=== TESTING ON ACTUAL DATA ===")
for i in range(min(3, len(df))):
    print(f"\n--- Example {i+1} ---")
    response_list = df.iloc[i]['responses']
    if isinstance(response_list, list) and len(response_list) > 0:
        prediction = response_list[0]
    else:
        prediction = str(response_list)
    
    coords = debug_coordinate_extraction(prediction)
    print(f"Final result: {coords}")
    
    # Show ground truth
    if 'reward_model' in df.columns:
        gt = df.iloc[i]['reward_model']
        if isinstance(gt, dict) and 'bbox' in gt:
            print(f"Ground truth bbox: {gt['bbox']}")

print("\n=== THE ISSUE ===")
print("The reward function uses .match() which only matches at the START of the string.")
print("If there's any extra text before the coordinates, it will fail!")
print("Example: 'The coordinates are 512 512' will fail because it doesn't start with digits.")
