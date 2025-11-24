#!/bin/bash

# Setup verification script
# Tests all components before running full training

echo "============================================"
echo "EE 782 Project - Setup Verification"
echo "============================================"

# Check Java
echo ""
echo "[1/5] Checking Java installation..."
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    echo "✓ Java found: $JAVA_VERSION"
    
    # Check if Java 17+
    if java -version 2>&1 | grep -q "version \"1[7-9]"; then
        echo "✓ Java 17+ detected"
    else
        echo "✗ Warning: Java 17+ recommended"
    fi
else
    echo "✗ Java not found. Please install JDK 17+"
    exit 1
fi

# Check Maven
echo ""
echo "[2/5] Checking Maven installation..."
if command -v mvn &> /dev/null; then
    MVN_VERSION=$(mvn -version | head -n 1)
    echo "✓ Maven found: $MVN_VERSION"
else
    echo "✗ Maven not found. Installing..."
    sudo apt-get update && sudo apt-get install -y maven
fi

# Check Python
echo ""
echo "[3/5] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python not found. Please install Python 3.9+"
    exit 1
fi

# Check Conda
echo ""
echo "[4/5] Checking Conda installation..."
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "✓ Conda found: $CONDA_VERSION"
else
    echo "⚠ Conda not found. You can still use pip for Python packages."
fi

# Test CloudSim build
echo ""
echo "[5/5] Testing CloudSim build..."
cd simulation/java
if mvn clean compile -q; then
    echo "✓ CloudSim builds successfully"
else
    echo "✗ CloudSim build failed. Check Maven dependencies."
    exit 1
fi
cd ../..

echo ""
echo "============================================"
echo "✓ All checks passed!"
echo "============================================"
echo ""
echo "You're ready to start training. Run:"
echo "  ./quick_start.sh"
echo ""
echo "Or manually:"
echo "  1. Terminal 1: cd simulation/java && mvn exec:java"
echo "  2. Terminal 2: cd algorithms/rl/dqn && python train_dqn.py"
echo ""
