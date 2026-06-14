import os
import urllib.request

data_dir = "/home/jovyan/work/data"
os.makedirs(f"{data_dir}/vatex", exist_ok=True)
os.makedirs(f"{data_dir}/msvd", exist_ok=True)

# 1. Download VATEX (Direct from official hosting)
print("Downloading VATEX...")
vatex_url = "https://eric-xw.github.io/vatex-website/data/vatex_training_v1.0.json"
urllib.request.urlretrieve(vatex_url, f"{data_dir}/vatex/vatex_train.json")
print("VATEX downloaded.")

# 2. Download MSVD (Direct from commonly used computer vision mirrors)
# Note: MSVD is often packaged as a CSV or varying JSONs. 
# We will pull the raw caption CSV to ensure we get everything, and the agent 
# can write a quick parser to convert it to JSON format if needed.
print("Downloading MSVD annotations...")
msvd_url = "https://raw.githubusercontent.com/xudejing/video-clip-order-prediction/master/data/msvd/raw-captions.csv"
urllib.request.urlretrieve(msvd_url, f"{data_dir}/msvd/msvd_train.csv")
print("MSVD downloaded.")

print("Annotations downloaded successfully!")