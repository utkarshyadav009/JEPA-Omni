#!/bin/bash
# Script to download VATEX parts 2 and 3 from Kaggle.
# Names corrected based on kaggle search.

mkdir -p /home/jovyan/work/data/vatex/part2
mkdir -p /home/jovyan/work/data/vatex/part3

export KAGGLE_USERNAME=utkishu
export KAGGLE_KEY=KGAT_233ae7d0a05f46e1f805b0575f4f3beb

echo "Downloading VATEX Part 2 (khaledatef1/vatex01101)..."
kaggle datasets download -d khaledatef1/vatex01101 --path /home/jovyan/work/data/vatex/part2 --unzip

echo "Downloading VATEX Part 3 (khaledatef1/vatex011011)..."
kaggle datasets download -d khaledatef1/vatex011011 --path /home/jovyan/work/data/vatex/part3 --unzip

echo "Downloads complete."
