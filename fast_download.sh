#!/bin/bash

# Exit on any error
set -e

echo "=========================================================="
echo " High-Speed HuggingFace Dataset Downloader (Git-LFS)"
echo "=========================================================="

echo "[1/4] Downloading standalone Git-LFS binary..."
# Download the pre-compiled git-lfs binary for Linux
wget -q -nc https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz

echo "[2/4] Extracting and configuring Git-LFS..."
tar -xzf git-lfs-linux-amd64-v3.4.0.tar.gz

# Add the local git-lfs binary to the system PATH for this script's execution
export PATH=$PATH:$(pwd)/git-lfs-3.4.0

# Initialize git-lfs locally without needing sudo
git lfs install

# MAXIMIZE SPEED: Tell Git-LFS to use 64 concurrent threads instead of default 8
git config --global lfs.concurrenttransfers 64

echo "[3/4] Cloning the Nemotron-AIQ-Agentic-Safety-Dataset repository..."
# This clones the folder structure and the tiny pointer files instantly
git clone https://huggingface.co/datasets/meet-the-1337/Nemotron-AIQ-Agentic-Safety-Dataset-1.0

echo "[4/4] Pulling 3.3GB of OpenTelemetry JSON files via concurrent streams..."
cd Nemotron-AIQ-Agentic-Safety-Dataset-1.0

# This command actually fetches the heavy 3.3 GB files using 64 parallel threads
git lfs pull

echo "=========================================================="
echo " Download Complete! The dataset is ready for processing."
echo "=========================================================="
