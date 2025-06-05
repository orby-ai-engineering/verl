# Save this as explore_dataset.py
import pandas as pd
import numpy as np

# Load the output file
df = pd.read_parquet('/root/data/screenspot/result-test-output-1.parquet')

print("=== DATASET STRUCTURE ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print()

print("=== COLUMN TYPES ===")
print(df.dtypes)
print()

print("=== FIRST ROW (RAW) ===")
for col in df.columns:
    value = df.iloc[0][col]
    print(f"{col}:")
    print(f"  Type: {type(value)}")
    print(f"  Value: {repr(value)}")
    if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
        try:
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First element type: {type(value[0])}")
                print(f"  First element: {repr(value[0])}")
        except:
            pass
    print()

print("=== SAMPLE OF RESPONSES COLUMN ===")
responses_col = df['responses']
print(f"Responses column type: {type(responses_col)}")
print(f"First 3 responses:")
for i in range(min(3, len(df))):
    resp = responses_col.iloc[i]
    print(f"  [{i}] Type: {type(resp)}, Value: {repr(resp)}")
    print()
