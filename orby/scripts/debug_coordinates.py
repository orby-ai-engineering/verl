# Save this as debug_coordinates.py
import pandas as pd
import re

# Load the output file
df = pd.read_parquet('/root/data/screenspot/result-test-output-1.parquet')

print("=== COORDINATE EXTRACTION DEBUG ===\n")

def extract_coordinates_debug(response_list):
    """Extract coordinates with detailed debugging"""
    if isinstance(response_list, list) and len(response_list) > 0:
        response = response_list[0]  # Get the first (and likely only) response
    else:
        response = str(response_list)
    
    print(f"Processing response: {repr(response[:200])}...")
    
    # Look for <answer> tags
    answer_matches = re.findall(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    print(f"Found answer tags: {answer_matches}")
    
    if answer_matches:
        answer_text = answer_matches[0].strip()
        print(f"Answer text: '{answer_text}'")
        
        # Try different coordinate patterns
        patterns = [
            r'(\d+)\s+(\d+)',  # "123 456"
            r'(\d+),\s*(\d+)',  # "123, 456"
            r'x:\s*(\d+).*?y:\s*(\d+)',  # "x: 123, y: 456"
            r'\((\d+),\s*(\d+)\)',  # "(123, 456)"
        ]
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, answer_text, re.IGNORECASE)
            print(f"Pattern {i+1} '{pattern}': {matches}")
            if matches:
                return matches[0]  # Return first match as tuple
    
    return None

# Test on first few examples
print("=== TESTING COORDINATE EXTRACTION ===")
for i in range(min(5, len(df))):
    print(f"\n--- Example {i+1} ---")
    response = df.iloc[i]['responses']
    coords = extract_coordinates_debug(response)
    print(f"Final extracted coordinates: {coords}")
    
    # Check ground truth
    if 'reward_model' in df.columns:
        reward_data = df.iloc[i]['reward_model']
        print(f"Reward model data type: {type(reward_data)}")
        if isinstance(reward_data, dict):
            print(f"Ground truth bbox: {reward_data.get('bbox', 'Not found')}")
        else:
            print(f"Reward model data: {repr(reward_data)}")

# Check what the reward function expects
print("\n=== CHECKING REWARD FUNCTION ===")
