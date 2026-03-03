#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging (print each command before execution)

echo "Starting QUAKE Artifact and Baselines Setup Script."
echo "This script should be run with sudo (e.g., 'sudo ./install.sh')."
echo "Targeting Ubuntu 22.04 (Jammy) based system."

# Resolve the directory containing this script (the local quake repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Environment Variables
# -----------------------------
export CONDA_DIR_PATH="/opt/miniconda"
# Ensure Miniconda's bin is in PATH for the root user running the script
export PATH="${CONDA_DIR_PATH}/bin:${PATH}"
export DEBIAN_FRONTEND=noninteractive

# -----------------------------
# MinIO Configuration
# -----------------------------
MINIO_DATA_DIR="/opt/minio-data"
MINIO_USER="minioadmin"
MINIO_PASSWORD="minioadmin"
MINIO_PORT="9000"
MINIO_CONSOLE_PORT="9001"
MINIO_BUCKET="my-quake-bucket"

# -----------------------------
# Update System and Install Base APT Packages
# -----------------------------
echo ">>> Updating system and installing base APT packages..."
apt-get update

apt-get install -y --no-install-recommends \
    wget \
    curl \
    build-essential \
    ca-certificates \
    swig \
    git \
    libomp-dev \
    graphviz \
    numactl \
    libnuma-dev \
    htop \
    software-properties-common

# -----------------------------
# Install/Ensure GCC 11 and G++ 11
# -----------------------------
echo ">>> Ensuring GCC 11 and G++ 11 are set up correctly..."
if gcc --version 2>/dev/null | grep -q "gcc (.*) 11\."; then
    echo "GCC 11 already active. Skipping installation and update-alternatives."
else
    add-apt-repository -y ppa:ubuntu-toolchain-r/test
    apt-get update
    apt-get install -y gcc-11 g++-11

    echo ">>> Setting up GCC and G++ alternatives to point to version 11..."
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
                             --slave /usr/bin/gcov gcc-gcov /usr/bin/gcov-11 \
                             --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
                             --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 \
                             --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
    update-alternatives --set gcc /usr/bin/gcc-11
    update-alternatives --set g++ /usr/bin/g++-11
fi

echo "Verifying GCC version after update-alternatives:"
gcc --version
echo "Verifying G++ version after update-alternatives:"
g++ --version

# -----------------------------
# Install CMake 3.24.2
# -----------------------------
CMAKE_VERSION_EXPECTED="3.24.2"
CMAKE_INSTALL_PATH="/usr/local/bin/cmake"
echo ">>> Installing CMake ${CMAKE_VERSION_EXPECTED}..."
if command -v cmake &> /dev/null && [[ "$(cmake --version)" == *"cmake version ${CMAKE_VERSION_EXPECTED}"* ]]; then
    echo "CMake version ${CMAKE_VERSION_EXPECTED} already installed at $(command -v cmake). Skipping."
else
    echo "Installing CMake ${CMAKE_VERSION_EXPECTED}..."
    cd /tmp
    wget -qO cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION_EXPECTED}/cmake-${CMAKE_VERSION_EXPECTED}-linux-x86_64.sh
    chmod +x cmake.sh
    ./cmake.sh --skip-license --prefix=/usr/local # Runs as root
    rm cmake.sh
    cd -
fi
echo "Verifying CMake version:"
cmake --version

# -----------------------------
# Install Miniconda
# -----------------------------
echo ">>> Installing Miniconda to ${CONDA_DIR_PATH}..."
if [ -x "${CONDA_DIR_PATH}/bin/conda" ]; then
    echo "Miniconda already found in ${CONDA_DIR_PATH}. Skipping installation."
else
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "${CONDA_DIR_PATH}" # Runs as root, installs to /opt/miniconda
    rm miniconda.sh
    cd -
    echo ">>> Initializing Conda for bash (modifies /root/.bashrc as script runs with sudo)..."
    "${CONDA_DIR_PATH}/bin/conda" init bash
fi

# Source conda's profile script to make conda command available in this script session
echo ">>> Sourcing Conda profile script for current root session..."
if [ -f "${CONDA_DIR_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_DIR_PATH}/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda profile script not found at ${CONDA_DIR_PATH}/etc/profile.d/conda.sh after installation attempt."
    echo "Conda commands might not be available. Exiting."
    exit 1
fi

# -----------------------------
# Create Conda Environment 'quake-env' and Install Dependencies
# -----------------------------
CONDA_ENV_NAME="quake-env"
echo ">>> Setting up conda environment '${CONDA_ENV_NAME}'..."

# Helper: check if a package (and optional version) is installed in the conda env
pkg_installed() {
    local pkg="$1" ver="$2"
    if [ -n "${ver}" ]; then
        conda run -n "${CONDA_ENV_NAME}" pip show "${pkg}" 2>/dev/null | grep -q "Version: ${ver}"
    else
        conda run -n "${CONDA_ENV_NAME}" pip show "${pkg}" &>/dev/null
    fi
}

if conda env list | grep -qE "^${CONDA_ENV_NAME}\s+"; then
    echo "Conda environment '${CONDA_ENV_NAME}' already exists. Skipping creation."
else
    echo ">>> Creating conda.yaml file at /tmp/conda.yaml..."
    rm -f /tmp/conda.yaml
    cat > /tmp/conda.yaml << EOF
name: ${CONDA_ENV_NAME}
channels:
  - pytorch
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas<3
  - matplotlib
  - pytest
  - pip
  - pip:
    - sphinx
    - sphinx_rtd_theme
    - sphinxcontrib-mermaid
    - graphviz
    - pyyaml
EOF

    echo ">>> Creating conda environment '${CONDA_ENV_NAME}' from /tmp/conda.yaml..."
    conda env create -f /tmp/conda.yaml
    rm /tmp/conda.yaml # Clean up
    conda clean -afy
fi

echo ">>> Installing specific PyTorch (CPU) and other Python packages into '${CONDA_ENV_NAME}'..."
if pkg_installed torch; then
    echo "PyTorch already installed. Skipping."
else
    conda run -n "${CONDA_ENV_NAME}" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Downgrade faiss-cpu to 1.7.4 for numpy 1.25.x compatibility
# (faiss-cpu 1.12.0+ requires numpy 2.0+ with numpy._core module)
if pkg_installed faiss-cpu 1.7.4; then
    echo "faiss-cpu 1.7.4 already installed. Skipping."
else
    conda run -n "${CONDA_ENV_NAME}" pip install faiss-cpu==1.7.4 --force-reinstall
fi

if pkg_installed boto3; then
    echo "boto3 already installed. Skipping."
else
    conda run -n "${CONDA_ENV_NAME}" pip install boto3
fi

# aws-sdk-cpp is required for S3 support in quake (QUAKE_USE_S3)
if conda list -n "${CONDA_ENV_NAME}" | grep -q "^aws-sdk-cpp"; then
    echo "aws-sdk-cpp already installed. Skipping."
else
    conda install -n "${CONDA_ENV_NAME}" -c conda-forge aws-sdk-cpp -y
fi

echo ">>> Verifying '${CONDA_ENV_NAME}'..."
conda env list
conda run -n "${CONDA_ENV_NAME}" python -c "import sys; print(f'OK in {sys.prefix}; python:', sys.executable); import torch; print('PyTorch version:', torch.__version__); import numpy; print('Numpy version:', numpy.__version__)"

# -----------------------------
# Install Intel oneAPI Base Toolkit
# -----------------------------
ONEAPI_INSTALL_PATH="/opt/intel/oneapi"
echo ">>> Installing Intel oneAPI Base Toolkit to ${ONEAPI_INSTALL_PATH}..."
if [ -f "${ONEAPI_INSTALL_PATH}/setvars.sh" ]; then
    echo "Intel oneAPI already found in ${ONEAPI_INSTALL_PATH}. Skipping installation."
else
    cd /tmp
    wget -O intel-oneapi-base-toolkit-offline.sh https://registrationcenter-download.intel.com/akdlm/IRC_NAS/6bfca885-4156-491e-849b-1cd7da9cc760/intel-oneapi-base-toolkit-2025.1.1.36_offline.sh
    chmod +x intel-oneapi-base-toolkit-offline.sh
    ./intel-oneapi-base-toolkit-offline.sh -a --silent --cli --eula accept --install-dir "${ONEAPI_INSTALL_PATH}" # sudo implied
    rm intel-oneapi-base-toolkit-offline.sh
    cd -
fi
# Source oneAPI environment variables for the current session (MKL is needed at quake build time).
echo ">>> Sourcing Intel oneAPI setvars.sh for current root session..."
if [ -f "${ONEAPI_INSTALL_PATH}/setvars.sh" ]; then
    source "${ONEAPI_INSTALL_PATH}/setvars.sh" || true  # ignore non-zero exit when already sourced
else
    echo "WARNING: oneAPI setvars.sh not found at ${ONEAPI_INSTALL_PATH}/setvars.sh. Quake build may fail."
fi

# -----------------------------
# Build and Install QUAKE from local directory
# -----------------------------
QUAKE_FULL_PATH="${SCRIPT_DIR}"

echo ">>> Building and installing QUAKE from local directory ${QUAKE_FULL_PATH}..."
cd "${QUAKE_FULL_PATH}"

git config --global --add safe.directory "${QUAKE_FULL_PATH}" # For root's global config
SUBMODULE_PATHS=("src/cpp/third_party/concurrentqueue" "src/cpp/third_party/faiss" "src/cpp/third_party/pybind11")
for SUBMODULE_PATH in "${SUBMODULE_PATHS[@]}"; do
    git config --global --add safe.directory "${QUAKE_FULL_PATH}/${SUBMODULE_PATH}"
done
git submodule update --init --recursive

echo ">>> Fixing ABI compatibility: match PyTorch's _GLIBCXX_USE_CXX11_ABI setting..."
# PyTorch 2.10+ uses _GLIBCXX_USE_CXX11_ABI=1 (new ABI), but quake's CMakeLists.txt
# hardcodes _GLIBCXX_USE_CXX11_ABI=0 (old ABI). Fix to match PyTorch.
TORCH_CXX11_ABI=$(conda run -n "${CONDA_ENV_NAME}" python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
echo "Detected PyTorch _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}"
sed -i "s/add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)/add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})/" "${QUAKE_FULL_PATH}/CMakeLists.txt"

echo ">>> Building and installing QUAKE Python package..."
conda run -n "${CONDA_ENV_NAME}" pip install --no-build-isolation --no-deps .

echo ">>> Copying freshly built bindings to ensure ABI compatibility..."
# Copy the newly built bindings from src/python/ to site-packages to avoid ABI compatibility issues
SITE_PACKAGES_PATH=$(conda run -n "${CONDA_ENV_NAME}" python -c "import site; print(site.getsitepackages()[0])")
if [ -f "${QUAKE_FULL_PATH}/src/python/_bindings.cpython-311-x86_64-linux-gnu.so" ]; then
    cp "${QUAKE_FULL_PATH}/src/python/_bindings.cpython-311-x86_64-linux-gnu.so" "${SITE_PACKAGES_PATH}/quake/"
    echo "Copied _bindings.cpython-311-x86_64-linux-gnu.so"
fi
if [ -f "${QUAKE_FULL_PATH}/src/python/libquake_c.so" ]; then
    cp "${QUAKE_FULL_PATH}/src/python/libquake_c.so" "${SITE_PACKAGES_PATH}/quake/"
    echo "Copied libquake_c.so"
fi

cd /

# -----------------------------
# Install and Configure MinIO
# -----------------------------
echo ">>> Setting up MinIO S3-compatible object storage..."

# Download MinIO server binary
if [ -x "/usr/local/bin/minio" ]; then
    echo "MinIO binary already present. Skipping download."
else
    echo "Downloading MinIO server..."
    wget -qO /usr/local/bin/minio https://dl.min.io/server/minio/release/linux-amd64/minio
    chmod +x /usr/local/bin/minio
fi

# Download MinIO client (mc)
if [ -x "/usr/local/bin/mc" ]; then
    echo "MinIO client (mc) already present. Skipping download."
else
    echo "Downloading MinIO client (mc)..."
    wget -qO /usr/local/bin/mc https://dl.min.io/client/mc/release/linux-amd64/mc
    chmod +x /usr/local/bin/mc
fi

# Create data directory
mkdir -p "${MINIO_DATA_DIR}"

# Write systemd service file
cat > /etc/systemd/system/minio.service << EOF
[Unit]
Description=MinIO S3-compatible Object Storage
After=network.target

[Service]
Type=simple
User=root
Environment="MINIO_ROOT_USER=${MINIO_USER}"
Environment="MINIO_ROOT_PASSWORD=${MINIO_PASSWORD}"
ExecStart=/usr/local/bin/minio server ${MINIO_DATA_DIR} --address ":${MINIO_PORT}" --console-address ":${MINIO_CONSOLE_PORT}"
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable minio
systemctl restart minio

# Wait for MinIO to become ready (up to 30 seconds)
echo "Waiting for MinIO to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${MINIO_PORT}/minio/health/ready" > /dev/null 2>&1; then
        echo "MinIO is ready."
        break
    fi
    if [ "${i}" -eq 30 ]; then
        echo "WARNING: MinIO did not become ready within 30 seconds. Check 'systemctl status minio'."
    fi
    sleep 1
done

# Create default bucket using mc
mc alias set local "http://localhost:${MINIO_PORT}" "${MINIO_USER}" "${MINIO_PASSWORD}" > /dev/null
mc mb --ignore-existing "local/${MINIO_BUCKET}"
echo "MinIO bucket '${MINIO_BUCKET}' is ready."

echo "--------------------------------------------------------------------"
echo "Setup script finished successfully."
echo "To activate the conda environment (in a new shell, as root or user depending on .bashrc): conda activate ${CONDA_ENV_NAME}"
echo "QUAKE built from local directory: ${QUAKE_FULL_PATH}"
echo "Intel oneAPI (if installed) is in ${ONEAPI_INSTALL_PATH}. Source with: source ${ONEAPI_INSTALL_PATH}/setvars.sh"
echo ""
echo "MinIO S3-compatible endpoint:  http://localhost:${MINIO_PORT}"
echo "MinIO console (web UI):        http://localhost:${MINIO_CONSOLE_PORT}"
echo "MinIO credentials:             ${MINIO_USER} / ${MINIO_PASSWORD}"
echo "Default bucket:                ${MINIO_BUCKET}"
echo ""
echo "To use MinIO with Quake, set these environment variables:"
echo "  export QUAKE_S3_ENDPOINT=http://localhost:${MINIO_PORT}"
echo "  export QUAKE_S3_BUCKET=${MINIO_BUCKET}"
echo "  export AWS_ACCESS_KEY_ID=${MINIO_USER}"
echo "  export AWS_SECRET_ACCESS_KEY=${MINIO_PASSWORD}"
echo "--------------------------------------------------------------------"
