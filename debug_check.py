import pandas as pd
import os

metadata = pd.read_csv("metadata.csv")
print("COLUMNS FOUND:", metadata.columns.tolist())
print("\nFIRST ROW:\n", metadata.iloc[0])
print("\nfilepath dtype/values:")
print(metadata["filepath"] if "filepath" in metadata.columns else "NO 'filepath' COLUMN FOUND")

print("\npose_landmarker.task exists:", os.path.exists("pose_landmarker.task"))
print("trimmed/ folder exists:", os.path.exists("trimmed"))
if os.path.exists("trimmed"):
    for root, dirs, files in os.walk("trimmed"):
        for f in files:
            print("  found trimmed file:", os.path.join(root, f))