#!/bin/bash
set -e

# Parse command line arguments
SKIP_CONTAINER=false
CLEAN_CACHE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-container|--skip-docker)
            SKIP_CONTAINER=true
            ;;
        --clean-cache)
            CLEAN_CACHE=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --skip-container  Skip Docker container tests"
            echo "  --clean-cache     Remove stateful Docker containers"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
    shift
done

# Handle cache cleanup if requested
if [ "$CLEAN_CACHE" = true ]; then
    echo "🧹 Cleaning stateful Docker containers..."
    docker rm -f rust-gpu-demo-test rust-cuda-test 2>/dev/null || true
    echo "✅ Stateful containers removed"
    exit 0
fi

echo "🧪 Rust GPU Chimera Demo - Comprehensive Test Suite"
echo "=================================================="

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
else
    echo "❌ Unsupported platform: $OSTYPE"
    exit 1
fi

echo "📍 Platform: $PLATFORM"
if [ "$SKIP_CONTAINER" = true ]; then
    echo "📌 Container tests: DISABLED"
fi
echo ""

# Function to run a test
run_test() {
    local name="$1"
    local cmd="$2"
    echo -e "\n📋 Test: $name"
    echo "   Command: $cmd"
    echo "   -------------------"
    if eval "$cmd"; then
        echo "   ✅ PASSED"
    else
        echo "   ❌ FAILED"
    fi
}

if [[ "$PLATFORM" == "macOS" ]]; then
    echo "Running macOS tests..."
    
    # Test 1: CPU only
    run_test "CPU execution" "cargo run --release --quiet"
    
    # Test 2: wgpu with Metal
    run_test "wgpu with Metal backend" "cargo run --release --features wgpu --quiet"
    
    # Test 3: wgpu with MoltenVK (dynamically find VulkanSDK)
    VULKAN_SDK=""
    MOLTENVK_ICD=""
    MOLTENVK_LIB=""
    
    # Search for VulkanSDK in common locations
    for sdk_path in "$HOME/VulkanSDK"/*/ "/usr/local/VulkanSDK"/*/ ; do
        if [ -d "$sdk_path" ]; then
            icd_path="${sdk_path}macOS/share/vulkan/icd.d/MoltenVK_icd.json"
            lib_path="${sdk_path}macOS/lib"
            if [ -f "$icd_path" ] && [ -d "$lib_path" ]; then
                VULKAN_SDK="$sdk_path"
                MOLTENVK_ICD="$icd_path"
                MOLTENVK_LIB="$lib_path"
                break
            fi
        fi
    done
    
    if [ -n "$MOLTENVK_ICD" ]; then
        echo "   Found VulkanSDK at: $VULKAN_SDK"
        run_test "wgpu with MoltenVK" \
            "VK_ICD_FILENAMES=$MOLTENVK_ICD DYLD_LIBRARY_PATH=$MOLTENVK_LIB:\$DYLD_LIBRARY_PATH cargo run --release --features wgpu,vulkan --quiet"
    else
        echo -e "\n📋 Test: wgpu with MoltenVK"
        echo "   ⚠️  SKIPPED - VulkanSDK not found"
    fi
    
    # Test 4: ash with MoltenVK
    if [ -n "$MOLTENVK_ICD" ]; then
        run_test "ash with MoltenVK" \
            "VK_ICD_FILENAMES=$MOLTENVK_ICD DYLD_LIBRARY_PATH=$MOLTENVK_LIB:\$DYLD_LIBRARY_PATH cargo run --release --features ash --quiet"
    else
        echo -e "\n📋 Test: ash with MoltenVK"
        echo "   ⚠️  SKIPPED - VulkanSDK not found"
    fi
    
    # Test 5: Linux container tests
    if [ "$SKIP_CONTAINER" = true ]; then
        echo -e "\n🐧 Linux Container Tests"
        echo "========================"
        echo "   ⚠️  SKIPPED - Container tests disabled via --skip-container"
    else
        echo -e "\n🐧 Linux Container Tests"
        echo "========================"
        
        # Check if Docker is available
        if command -v docker &> /dev/null; then
            # Check if container exists, build if not
            if ! docker images | grep -q "rust-gpu-demo"; then
                echo "Building Linux test container (first time only)..."
                cd container
                if ! docker build -f Dockerfile.linux-arm64-mesa -t rust-gpu-demo .; then
                    echo "   ❌ Failed to build Docker container"
                    exit 1
                fi
                cd ..
            else
                echo "Using existing Linux test container..."
            fi
            echo "Running tests in Linux container..."
            
            # Create a temporary test script for inside the container
            CONTAINER_TEST_SCRIPT=$(mktemp -t container-test.XXXXXX)
            cat > "$CONTAINER_TEST_SCRIPT" << 'EOF'
#!/bin/bash
set -e

echo "🐧 Running tests inside Linux container"
echo ""

# Function to run a test
run_test() {
    local name="$1"
    local cmd="$2"
    echo -e "\n📋 Test: $name"
    echo "   Command: $cmd"
    echo "   -------------------"
    if eval "$cmd"; then
        echo "   ✅ PASSED"
    else
        echo "   ❌ FAILED"
    fi
}

cd /workspace

# Test 1: CPU only
run_test "CPU execution" "cargo run --release --quiet"

# Test 2: wgpu with Mesa Vulkan
run_test "wgpu with Mesa Vulkan" "cargo run --release --features wgpu --quiet"

# Test 3: Check Vulkan availability
echo -e "\n📋 Vulkan Information:"
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null || echo "   No Vulkan devices found"
else
    echo "   vulkaninfo not available"
fi
EOF
            chmod +x "$CONTAINER_TEST_SCRIPT"
            
            # Check if stateful container is already running
            if docker ps --format '{{.Names}}' | grep -q '^rust-gpu-demo-test$'; then
                echo "Reusing existing running container 'rust-gpu-demo-test'..."
            elif docker ps -a --format '{{.Names}}' | grep -q '^rust-gpu-demo-test$'; then
                echo "Starting existing stopped container 'rust-gpu-demo-test'..."
                docker start rust-gpu-demo-test > /dev/null 2>&1
            else
                echo "Creating container 'rust-gpu-demo-test' with volume mount..."
                # Create container with volume mount
                docker run -d --name rust-gpu-demo-test \
                    -v "$(pwd)":/workspace:rw \
                    rust-gpu-demo tail -f /dev/null > /dev/null 2>&1
            fi
            # Copy test script and run it
            docker cp "$CONTAINER_TEST_SCRIPT" rust-gpu-demo-test:/container-test.sh
            docker exec rust-gpu-demo-test /container-test.sh
            
            # Clean up
            rm "$CONTAINER_TEST_SCRIPT"
        else
            echo "   ⚠️  SKIPPED - Docker not available"
        fi
    fi
    
    # Test 6: CUDA build test
    if [ "$SKIP_CONTAINER" = true ]; then
        echo -e "\n🚀 CUDA Build"
        echo "=================="
        echo "   ⚠️  SKIPPED - Container tests disabled via --skip-container"
    else
        echo -e "\n🚀 CUDA Build"
        echo "=================="
        
        # Check if Docker is available
        if command -v docker &> /dev/null; then
        echo "Testing CUDA build in rust-cuda container..."
        
        # Pull the rust-cuda image if needed
        if ! docker images | grep -q "ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12"; then
            echo "Pulling rust-cuda container image (first time only)..."
            docker pull ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:main
        fi
        
        # Check if CUDA stateful container is already running
        if docker ps --format '{{.Names}}' | grep -q '^rust-cuda-test$'; then
            echo "Reusing existing running container 'rust-cuda-test'..."
        elif docker ps -a --format '{{.Names}}' | grep -q '^rust-cuda-test$'; then
            echo "Starting existing stopped container 'rust-cuda-test'..."
            docker start rust-cuda-test > /dev/null 2>&1
        else
            echo "Creating container 'rust-cuda-test' with volume mount..."
            # Create container with volume mount
            docker run -d --name rust-cuda-test \
                -v "$(pwd)":/workspace:rw \
                ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:main \
                tail -f /dev/null > /dev/null 2>&1
        fi
        
        echo "   Building with CUDA feature..."
        if docker exec rust-cuda-test bash -c "cd /workspace && cargo build --release --features cuda" > /dev/null 2>&1; then
            echo "   ✅ CUDA build successful"
        else
            echo "   ❌ CUDA build failed"
            echo "   Run with verbose output: docker exec rust-cuda-test bash -c 'cd /workspace && cargo build --release --features cuda'"
        fi
        else
            echo "   ⚠️  SKIPPED - Docker not available"
        fi
    fi
    
elif [[ "$PLATFORM" == "Linux" ]]; then
    echo "Running Linux native tests..."
    
    # Test 1: CPU only
    run_test "CPU execution" "cargo run --release --quiet"
    
    # Test 2: wgpu with Vulkan
    run_test "wgpu with Vulkan" "cargo run --release --features wgpu --quiet"
    
    # Test 3: wgpu with LLVMpipe (if available)
    if [ -f "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json" ]; then
        run_test "wgpu with LLVMpipe" \
            "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json cargo run --release --features wgpu --quiet"
    fi
    
    # Test 4: ash with Vulkan
    run_test "ash with Vulkan" "cargo run --release --features ash --quiet"
    
    # Test 5: CUDA (requires NVIDIA GPU)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        run_test "CUDA" "cargo run --release --features cuda --quiet"
    else
        echo -e "\n📋 Test: CUDA"
        echo "   ⚠️  SKIPPED - No NVIDIA GPU detected"
    fi
    
    # Test 6: Check Vulkan availability
    echo -e "\n📋 Vulkan Information:"
    if command -v vulkaninfo &> /dev/null; then
        vulkaninfo --summary 2>/dev/null || echo "   No Vulkan devices found"
    else
        echo "   vulkaninfo not available"
    fi
fi

echo -e "\n✅ Test suite completed!"
