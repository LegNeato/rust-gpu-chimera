#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
REMOTE_DIR="/root/rust-gpu-chimera-demo"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "🌋 Fixing Vulkan setup for NVIDIA GPU"
echo "===================================="

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
# Clear MOTD
echo "" > /etc/motd

cd /root/rust-gpu-chimera-demo

echo "🔍 Finding NVIDIA Vulkan library:"
find /usr -name "libnvidia-glcore.so*" -o -name "libGLX_nvidia.so*" 2>/dev/null | head -10

echo -e "\n📦 Installing NVIDIA Vulkan components:"
# The container should have NVIDIA drivers, we just need to configure Vulkan
apt-get update -qq
apt-get install -y -qq libnvidia-gl-570 || echo "Specific version not found, trying generic..."

echo -e "\n🔍 Looking for NVIDIA libraries again:"
ldconfig
ldconfig -p | grep nvidia | grep -E "(GL|vulkan)" | head -10

echo -e "\n🔧 Creating proper NVIDIA Vulkan ICD:"
# Remove incorrect ICD files
rm -f /usr/share/vulkan/icd.d/nvidia_icd.json

# Find the actual NVIDIA Vulkan driver
NVIDIA_VULKAN_SO=$(find /usr/lib -name "libnvidia-vulkan-producer.so*" -o -name "libGLX_nvidia.so.0" 2>/dev/null | head -1)
if [ -z "$NVIDIA_VULKAN_SO" ]; then
    echo "NVIDIA Vulkan library not found, checking standard locations..."
    if [ -f "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0" ]; then
        NVIDIA_VULKAN_SO="/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0"
    elif [ -f "/usr/lib/x86_64-linux-gnu/nvidia/current/libGLX_nvidia.so.0" ]; then
        NVIDIA_VULKAN_SO="/usr/lib/x86_64-linux-gnu/nvidia/current/libGLX_nvidia.so.0"
    fi
fi

if [ -n "$NVIDIA_VULKAN_SO" ]; then
    echo "Found NVIDIA library: $NVIDIA_VULKAN_SO"
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json << JSON
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "$NVIDIA_VULKAN_SO",
        "api_version" : "1.3.275"
    }
}
JSON
    echo "Created NVIDIA ICD file with path: $NVIDIA_VULKAN_SO"
else
    echo "ERROR: Could not find NVIDIA Vulkan library!"
fi

echo -e "\n🌋 Testing Vulkan with NVIDIA GPU:"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
vulkaninfo --summary 2>&1 | grep -E "(GPU|deviceName|driverVersion)" | head -20

echo -e "\n🔨 Testing wgpu with fixed Vulkan:"
export LD_LIBRARY_PATH="/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"
export LLVM_LINK_STATIC=1
export RUST_LOG=wgpu_hal::vulkan=debug,wgpu_core=debug
export PATH="$HOME/.cargo/bin:${PATH}"

cargo run --release --features wgpu

echo -e "\n🔥 Testing ash with fixed Vulkan:"
cargo run --release --features ash

echo -e "\n✅ Vulkan fix complete!"
EOF