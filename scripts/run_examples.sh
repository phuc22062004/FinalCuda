#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Example commands:"
echo "  CPU training:  ${PROJECT_ROOT}/build_cpu/autoencoder_cpu"
echo "  CUDA basic:    ${PROJECT_ROOT}/build_cuda/autoencoder_cuda_basic"
echo "  CUDA opt v1:   ${PROJECT_ROOT}/build_cuda/autoencoder_cuda_opt_v1"
echo "  CUDA opt v2:   ${PROJECT_ROOT}/build_cuda/autoencoder_cuda_opt_v2"
echo
echo "  ${PROJECT_ROOT}/cifar-10-binary/cifar-10-batches-bin"


