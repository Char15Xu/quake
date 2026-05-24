#!/bin/bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate quake-env
source /opt/intel/oneapi/setvars.sh || true
export QUAKE_USE_S3=1
export QUAKE_S3_ENDPOINT=http://localhost:9000
export QUAKE_S3_BUCKET=my-quake-bucket
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export LD_PRELOAD=/opt/intel/oneapi/mkl/2025.1/lib/libmkl_rt.so.2
python cache_experiment.py --skip-build
