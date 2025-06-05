# Save this as debug_responses.py
import pandas as pd
import re

# Load the output file
df = pd.read_parquet('/root/data/screenspot/result-test-output-1.parquet')

print("=== DEBUGGING COORDINATE SCORES ===\n")

# Function to extract coordinates from response
def extract_coordinates(response):
    # Convert bytes to string if needed
    if isinstance(response, bytes):
        response = response.decode('utf-8')
    elif isinstance(response, list) and len(response) > 0:
        response = response[0]
        if isinstance(response, bytes):
            response = response.decode('utf-8')
    
    # Look for <answer>x y</answer> pattern
    answer_match = re.search(r'<answer>\s*([^<]+)\s*</answer>', response, re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Try to extract two numbers
        coord_match = re.findall(r'\d+', answer_text)
        if len(coord_match) >= 2:
            return f"{coord_match[0]} {coord_match[1]}"
        return answer_text
    return "No answer found"

def decode_response(response):
    """Helper to decode response to string"""
    if isinstance(response, bytes):
        return response.decode('utf-8')
    elif isinstance(response, list) and len(response) > 0:
        resp = response[0]
        if isinstance(resp, bytes):
            return resp.decode('utf-8')
        return str(resp)
    return str(response)

# Check a few examples
print("=== SAMPLE RESPONSES ===")
for i in range(5):
    print(f"\n--- Example {i+1} ---")
    raw_response = df.iloc[i]['responses']
    response = decode_response(raw_response)
    
    print(f"Raw type: {type(raw_response)}")
    print(f"Full Response: {repr(response[:300])}...")
    
    # Extract coordinates
    coords = extract_coordinates(raw_response)
    print(f"Extracted Coordinates: {coords}")
    
    # Check if it has proper format
    has_thinking = '<think>' in response.lower()
    has_answer = '<answer>' in response.lower()
    print(f"Has <think>: {has_thinking}")
    print(f"Has <answer>: {has_answer}")
    
    # Check ground truth if available
    if 'reward_model' in df.columns:
        gt = df.iloc[i]['reward_model']
        if isinstance(gt, dict) and 'bbox' in gt:
            print(f"Ground Truth: {gt['bbox']}")

print("\n=== COORDINATE FORMAT ANALYSIS ===")
# Analyze coordinate patterns
coord_patterns = []
for i in range(min(20, len(df))):
    response = df.iloc[i]['responses']
    coords = extract_coordinates(response)
    coord_patterns.append(coords)

print("Coordinate patterns found:")
for pattern in set(coord_patterns):
    count = coord_patterns.count(pattern)
    print(f"  '{pattern}': {count} times")

print("\n=== EXPECTED FORMAT ===")
print("The model should generate responses like:")
print("<think>I need to find the element and click on it...</think>")
print("<answer>123 456</answer>")
print("\nWhere 123 456 are the x,y coordinates separated by a space.")
