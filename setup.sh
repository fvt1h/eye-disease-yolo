#!/bin/bash
set -e

# Upgrade pip dan setup tools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install numpy==1.23.5
pip install -r requirements.txt

# Print versi untuk debugging
python --version
pip list