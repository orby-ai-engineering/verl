import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the parquet file
df = pd.read_parquet("~/data/screenspot/result-test-output-1.parquet")

print(df.head(10))

print("=== EVALUATION RESULTS ANALYSIS ===")

# Extract ground truth coordinates from extra_info
ground_truth_coords = []
questions = []


for i, row in df.iterrows():
    extra_info = row['extra_info']
    if isinstance(extra_info, dict) and 'bounding_box' in extra_info:
        bbox = extra_info['bounding_box']
        # Convert bounding box to center coordinates
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        ground_truth_coords.append([center_x, center_y])
        questions.append(extra_info.get('question', ''))

ground_truth_coords = np.array(ground_truth_coords)

print(f"Total samples: {len(df)}")
print(f"Ground truth coordinates extracted: {len(ground_truth_coords)}")
print(f"\n=== EXAMPLE RESPONSES ===")
for i in range(min(15, len(df))):
    question = df.iloc[i]['extra_info']['question']
    response = df.iloc[i]['responses'] if len(df.iloc[i]['responses']) > 0 else "empty"
    gt_coords = ground_truth_coords[i]
    print(f"Q: '{question}' | GT: ({gt_coords[0]:.1f}, {gt_coords[1]:.1f}) | Response: '{response}'")
