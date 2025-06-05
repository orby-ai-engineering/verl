# Save this as debug_outputs.py
import pandas as pd

# Load the output file
df = pd.read_parquet('/root/data/screenspot/result-test-output-1.parquet')

print("=== EXAMINING MODEL OUTPUTS ===\n")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()

print("=== SAMPLE OUTPUTS ===")
for i in range(min(10, len(df))):
    print(f"\n--- Example {i+1} ---")
    
    # Get the response
    response = df.iloc[i]['responses']
    print(f"Response type: {type(response)}")
    
    if isinstance(response, list):
        print(f"Response list length: {len(response)}")
        if len(response) > 0:
            actual_response = response[0]
            print(f"Actual response type: {type(actual_response)}")
            print(f"Actual response: {repr(actual_response)}")
        else:
            print("Empty response list!")
    else:
        print(f"Response: {repr(response)}")
    
    # Check ground truth
    if 'reward_model' in df.columns:
        gt = df.iloc[i]['reward_model']
        if isinstance(gt, dict) and 'ground_truth' in gt:
            bbox = gt['ground_truth'].get('bbox', 'Not found')
            print(f"Ground truth bbox: {bbox}")
            if bbox != 'Not found':
                # Calculate center coordinates
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                print(f"Expected center coordinates: {center_x} {center_y}")

print("\n=== RESPONSE PATTERNS ===")
# Analyze what types of responses we're getting
response_patterns = {}
for i in range(min(50, len(df))):
    response = df.iloc[i]['responses']
    if isinstance(response, list) and len(response) > 0:
        resp_str = str(response[0])[:100]  # First 100 chars
    else:
        resp_str = str(response)[:100]
    
    # Categorize responses
    if not resp_str or resp_str.strip() == '':
        category = "EMPTY"
    elif resp_str.startswith('['):
        category = "LIST_FORMAT"
    elif 'assistant' in resp_str.lower():
        category = "ASSISTANT_REPETITION"
    elif '<think>' in resp_str.lower():
        category = "THINK_TAG_FORMAT"
    elif '<answer>' in resp_str.lower():
        category = "ANSWER_TAG_FORMAT"
    elif any(char.isdigit() for char in resp_str):
        category = "CONTAINS_NUMBERS"
    else:
        category = "OTHER"
    
    response_patterns[category] = response_patterns.get(category, 0) + 1

print("Response pattern distribution:")
for pattern, count in response_patterns.items():
    print(f"  {pattern}: {count}")
