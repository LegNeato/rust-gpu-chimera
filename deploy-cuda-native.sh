#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
REMOTE_DIR="/root/rust-gpu-chimera-demo"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "🚀 Deploying Rust GPU Chimera Demo to vast.ai (Native CUDA Setup)"
echo "=============================================================="

# Create remote directory
echo "Creating remote directory..."
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Sync files
echo "Syncing project files..."
rsync -avz --delete \
    --exclude 'target/' \
    --exclude '.git/' \
    --exclude '*.swp' \
    --exclude '.DS_Store' \
    --exclude 'deploy-*.sh' \
    -e "ssh $SSH_OPTS -p $REMOTE_PORT" \
    ./ $REMOTE_HOST:$REMOTE_DIR/

echo "Files synced successfully!"

# SSH into the machine and set up environment
echo -e "\n📦 Setting up CUDA development environment..."
ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
# Clear MOTD
echo "" > /etc/motd

cd /root/rust-gpu-chimera-demo

# Check GPU
echo -e "\n🎮 GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Check CUDA version
echo -e "\n🚀 CUDA Version:"
nvcc --version 2>/dev/null || echo "NVCC not found, checking nvidia-smi..."
nvidia-smi | grep "CUDA Version" || echo "CUDA info not found"

# Install build dependencies if not present
echo -e "\n📦 Installing build dependencies..."
if ! command -v clang &> /dev/null; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        build-essential \
        clang \
        curl \
        libssl-dev \
        libtinfo-dev \
        pkg-config \
        xz-utils \
        zlib1g-dev
fi

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo -e "\n🦀 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Show Rust version
echo -e "\n🦀 Rust version:"
cargo --version
rustc --version

# Check if LLVM 7 is needed and not present
if ! command -v llvm-config-7 &> /dev/null && ! command -v llvm-config &> /dev/null; then
    echo -e "\n⚠️  LLVM 7 not found. Installing LLVM for CUDA support..."
    
    # Install LLVM dependencies
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        libffi-dev \
        libedit-dev \
        libncurses5-dev \
        libxml2-dev \
        python3 \
        ninja-build
    
    # Download and build LLVM 7.1.0
    mkdir -p /tmp/llvm7
    cd /tmp/llvm7
    
    echo "Downloading LLVM 7.1.0..."
    curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz
    tar -xf llvm-7.1.0.src.tar.xz
    cd llvm-7.1.0.src
    mkdir build && cd build
    
    echo "Building LLVM 7.1.0 (this will take a while)..."
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
        -DLLVM_BUILD_LLVM_DYLIB=ON \
        -DLLVM_LINK_LLVM_DYLIB=ON \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF \
        -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF \
        -DLLVM_ENABLE_ZLIB=ON \
        -DLLVM_ENABLE_TERMINFO=ON \
        -DCMAKE_INSTALL_PREFIX=/usr \
        ..
    
    ninja -j$(nproc)
    ninja install
    
    # Create symlink
    ln -s /usr/bin/llvm-config /usr/bin/llvm-config-7
    
    # Cleanup
    cd /
    rm -rf /tmp/llvm7
    
    echo "LLVM 7 installed successfully!"
fi

# Set up environment variables
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
export LLVM_LINK_STATIC=1
export RUST_LOG=info
export PATH="$HOME/.cargo/bin:${PATH}"

# Go back to project directory
cd /root/rust-gpu-chimera-demo

# Show toolchain info
echo -e "\n🔧 Rust toolchain:"
rustup show

# Build and run
echo -e "\n🔨 Building project..."

# Test CPU backend first
echo -e "\n📊 Testing CPU backend..."
cargo test --release
cargo run --release

# Test CUDA backend
echo -e "\n🎮 Building and testing CUDA backend..."
if cargo build --release --features cuda; then
    echo "CUDA build successful!"
    echo -e "\n🚀 Running CUDA demo..."
    cargo test --release --features cuda
    cargo run --release --features cuda
else
    echo "CUDA build failed. Checking environment..."
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    echo "CUDA NVVM libraries:"
    ls -la /usr/local/cuda/nvvm/lib64/ 2>/dev/null || echo "NVVM lib64 not found"
    echo "LLVM config:"
    llvm-config --version 2>/dev/null || echo "llvm-config not found"
fi

echo -e "\n✅ Setup complete!"
EOF

echo -e "\n✅ Deployment complete!"
