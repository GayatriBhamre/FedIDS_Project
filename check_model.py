import numpy as np

# Load the NPZ file
data = np.load("artifacts/global_final.npz", allow_pickle=True)

# Show the keys stored inside
print("Keys in NPZ file:", data.files)

# Optionally, inspect each key's content
for key in data.files:
    print(f"\nKey: {key}")
    print("Type:", type(data[key]))
    print("Content preview:", data[key])
