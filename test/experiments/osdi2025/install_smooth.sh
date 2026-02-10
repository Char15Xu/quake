#!/bin/bash
set -euo pipefail

echo "Starting QUAKE smooth install script (Ubuntu 22.04)."

if [ "${EUID}" -ne 0 ]; then
  echo "ERROR: Please run with sudo (e.g., 'sudo ./install_smooth.sh')."
  exit 1
fi

TARGET_USER="${SUDO_USER:-root}"
TARGET_UID="$(id -u "${TARGET_USER}")"
TARGET_GID="$(id -g "${TARGET_USER}")"

export CONDA_DIR_PATH="/opt/miniconda"
export PATH="${CONDA_DIR_PATH}/bin:${PATH}"
export DEBIAN_FRONTEND=noninteractive

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

echo ">>> Ensuring GCC 11 and G++ 11 are set up correctly..."
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install -y gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
  --slave /usr/bin/gcov gcc-gcov /usr/bin/gcov-11 \
  --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
  --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-11 \
  --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
update-alternatives --set gcc /usr/bin/gcc-11
update-alternatives --set g++ /usr/bin/g++-11

echo ">>> Installing CMake 3.24.2..."
CMAKE_VERSION_EXPECTED="3.24.2"
if command -v cmake &> /dev/null && [[ "$(cmake --version)" == *"cmake version ${CMAKE_VERSION_EXPECTED}"* ]]; then
  echo "CMake version ${CMAKE_VERSION_EXPECTED} already installed. Skipping."
else
  cd /tmp
  wget -qO cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION_EXPECTED}/cmake-${CMAKE_VERSION_EXPECTED}-linux-x86_64.sh
  chmod +x cmake.sh
  ./cmake.sh --skip-license --prefix=/usr/local
  rm cmake.sh
  cd -
fi

echo ">>> Installing Miniconda to ${CONDA_DIR_PATH}..."
if [ -x "${CONDA_DIR_PATH}/bin/conda" ]; then
  echo "Miniconda already found. Skipping installation."
else
  cd /tmp
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "${CONDA_DIR_PATH}"
  rm miniconda.sh
  cd -
  "${CONDA_DIR_PATH}/bin/conda" init bash
fi

echo ">>> Sourcing Conda profile script..."
source "${CONDA_DIR_PATH}/etc/profile.d/conda.sh"

CONDA_ENV_NAME="quake-env"
echo ">>> Setting up conda environment '${CONDA_ENV_NAME}'..."

if conda env list | grep -E "^${CONDA_ENV_NAME}\s+"; then
  echo "Conda environment '${CONDA_ENV_NAME}' already exists. Removing for a clean setup..."
  conda env remove -n "${CONDA_ENV_NAME}" --all -y
fi

echo ">>> Creating conda environment '${CONDA_ENV_NAME}'..."
conda create -y -n "${CONDA_ENV_NAME}" python=3.11 pip

echo ">>> Ensuring writable conda package cache..."
mkdir -p /opt/miniconda/pkgs
chown -R "${TARGET_UID}:${TARGET_GID}" /opt/miniconda/pkgs
chown -R "${TARGET_UID}:${TARGET_GID}" /opt/miniconda/envs/${CONDA_ENV_NAME}

echo ">>> Installing Python dependencies..."
conda run -n "${CONDA_ENV_NAME}" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
conda run -n "${CONDA_ENV_NAME}" pip install \
  matplotlib \
  pytest \
  graphviz \
  pyyaml \
  lightgbm \
  scikit-learn \
  tabulate

echo ">>> Pinning numpy/pandas/scipy to compatible versions..."
conda run -n "${CONDA_ENV_NAME}" pip install "numpy==1.25.0" "pandas==2.2.3" "scipy==1.11.4"

echo ">>> Installing tensorflow (required by scann)..."
conda run -n "${CONDA_ENV_NAME}" pip install "tensorflow==2.16.2"

echo ">>> Installing ANN dependencies with pinned numpy..."
conda run -n "${CONDA_ENV_NAME}" pip install --force-reinstall "diskannpy==0.7.0"
conda run -n "${CONDA_ENV_NAME}" pip install --force-reinstall --no-deps "scann==1.3.2"
conda run -n "${CONDA_ENV_NAME}" pip install --force-reinstall "faiss-cpu==1.7.4"

echo ">>> Locating QUAKE repo..."
if [ -d "/users/charlesx/quake" ]; then
  QUAKE_FULL_PATH="/users/charlesx/quake"
elif [ -d "/opt/quake" ]; then
  QUAKE_FULL_PATH="/opt/quake"
else
  echo "ERROR: Could not find QUAKE repo. Set QUAKE_FULL_PATH manually or clone it."
  exit 1
fi

echo ">>> Building and installing QUAKE Python package..."
ABI_FLAG=$(conda run -n "${CONDA_ENV_NAME}" python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
export CMAKE_PREFIX_PATH="/opt/miniconda/envs/${CONDA_ENV_NAME}/lib/python3.11/site-packages/torch/share/cmake"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI_FLAG}"
cd "${QUAKE_FULL_PATH}"
conda run -n "${CONDA_ENV_NAME}" pip install . --no-build-isolation --no-cache-dir

echo "--------------------------------------------------------------------"
echo "Setup finished."
echo "Activate env: conda activate ${CONDA_ENV_NAME}"
echo "If you see Torch ABI errors at runtime, set:"
echo "  export LD_LIBRARY_PATH=${CONDA_DIR_PATH}/envs/${CONDA_ENV_NAME}/lib:${CONDA_DIR_PATH}/envs/${CONDA_ENV_NAME}/lib/python3.11/site-packages/torch/lib:\$LD_LIBRARY_PATH"
echo "--------------------------------------------------------------------"
