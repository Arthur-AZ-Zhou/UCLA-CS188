import numpy as np
import sys
from collections import defaultdict

def reconstruct_from_npz(npz_path):
    try:
        flat_data = np.load(npz_path)
        demos = defaultdict(lambda: {})  # demo_id -> { field_path: data }
        for key in flat_data.files:
            print(key)
            parts = key.split('_', 2)  # Expect format: demo_0_fieldname
            if len(parts) < 3:
                print(f"Skipping malformed key: {key}")
                continue
            demo_id = f"{parts[0]}_{parts[1]}"  # e.g., demo_0
            field_name = parts[2]  # reconstruct path
            demos[demo_id][field_name] = flat_data[key]

        print(f"Reconstructed {len(demos)} demos:")
        for demo_id, fields in demos.items():
            print(f"  {demo_id}: {len(fields)} fields")
            for field in fields:
                print(field, demos[demo_id][field].shape)
        return demos
    except Exception as e:
        print(f"Error reconstructing data: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reconstruct_npz.py <file.npz>")
    else:
        reconstruct_from_npz(sys.argv[1])
