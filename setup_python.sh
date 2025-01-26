#!/bin/bash

set -e

echo "Running: sudo apt-get update"
if ! command -v add-apt-repository &> /dev/null; then
    sudo apt-get update
    echo "Running: sudo apt-get install -y software-properties-common"
    sudo apt-get install -y software-properties-common
fi

echo "Running: sudo add-apt-repository -y ppa:deadsnakes/ppa"
sudo add-apt-repository -y ppa:deadsnakes/ppa
echo "Running: sudo apt-get update"
sudo apt-get update

echo "Running: sudo apt-get install python packages..."
sudo apt-get install --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip
