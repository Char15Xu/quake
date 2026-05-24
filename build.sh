#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate quake-env
source /opt/intel/oneapi/setvars.sh || true
pip install -e .
