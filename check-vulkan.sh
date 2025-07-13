#!/bin/bash
set -e

# Remote connection details
REMOTE_HOST="root@ssh1.vast.ai"
REMOTE_PORT="10839"
SSH_OPTS="-o StrictHostKeyChecking=no"

echo "🌋 Checking Vulkan setup on vast.ai"
echo "==================================="

ssh $SSH_OPTS -p $REMOTE_PORT $REMOTE_HOST << 'EOF'
# Clear MOTD
echo "" > /etc/motd

echo "🎮 GPU Information:"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo -e "\n📦 Checking Vulkan packages:"
dpkg -l | grep -i vulkan || echo "No Vulkan packages found"

echo -e "\n🔍 Looking for Vulkan ICD files:"
find /usr -name "*.json" -path "*/vulkan/icd.d/*" 2>/dev/null || echo "No ICD files found"
find /etc -name "*.json" -path "*/vulkan/icd.d/*" 2>/dev/null || true

echo -e "\n🔍 Looking for Vulkan libraries:"
find /usr -name "libvulkan.so*" 2>/dev/null | head -10
ldconfig -p | grep vulkan || echo "No Vulkan in ldconfig"

echo -e "\n🔍 Checking for NVIDIA Vulkan driver:"
find /usr -name "*nvidia*vulkan*" -o -name "*vulkan*nvidia*" 2>/dev/null | head -10

echo -e "\n📋 Environment variables:"
env | grep -i vulkan || echo "No Vulkan env vars set"

echo -e "\n🔧 Installing Vulkan loader and NVIDIA driver if needed:"
apt-get update -qq
apt-get install -y -qq libvulkan1 mesa-vulkan-drivers

echo -e "\n🔍 Checking again after install:"
find /usr -name "*.json" -path "*/vulkan/icd.d/*" 2>/dev/null || echo "Still no ICD files"

echo -e "\n🌋 Running vulkaninfo:"
if command -v vulkaninfo &> /dev/null; then
    VK_LOADER_DEBUG=all vulkaninfo 2>&1 | head -50
else
    echo "vulkaninfo not available"
fi

echo -e "\n💡 Checking if we need NVIDIA's Vulkan ICD:"
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    echo "NVIDIA ICD found:"
    cat /usr/share/vulkan/icd.d/nvidia_icd.json
else
    echo "NVIDIA ICD not found. Creating one..."
    mkdir -p /usr/share/vulkan/icd.d/
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'JSON'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.281"
    }
}
JSON
    echo "Created NVIDIA ICD file"
fi

echo -e "\n🔍 Final Vulkan check:"
vulkaninfo --summary 2>&1 | grep -E "(GPU|deviceName)" || echo "Vulkan still not working"
EOF