"""
Download Dunnhumby Complete Journey dataset from Kaggle
"""
import kagglehub
import shutil
import os

print("Downloading Dunnhumby Complete Journey dataset...")
path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")
print(f"Downloaded to: {path}")

# Move files to data/ folder
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

for file in os.listdir(path):
    if file.endswith('.csv'):
        src = os.path.join(path, file)
        dst = os.path.join(data_dir, file)
        shutil.copy2(src, dst)
        print(f"Copied {file} to {data_dir}/")

print("\n✅ Dataset ready in data/ folder!")
