#!/bin/bash

echo "🔧 Memulai instalasi dependensi..."

# Instal PyTorch dengan CUDA support (wajib untuk GPU)
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118

# Instal transformers dan dependensi lain
pip install transformers==4.38.0 accelerate==0.27.2 gradio scipy

echo "✅ Instalasi selesai!"
