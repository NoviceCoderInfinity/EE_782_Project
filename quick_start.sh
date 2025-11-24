#!/bin/bash

# Quick start script for DQN CloudSim Training
# This script sets up the environment and starts training

set -e

echo "============================================"
echo "DQN CloudSim Training Quick Start"
echo "============================================"

# Set Java environment
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

echo ""
echo "Step 1: Checking Java version..."
java -version

echo ""
echo "Step 2: Building CloudSim Java simulation..."
cd simulation/java
mvn clean compile

echo ""
echo "Step 3: Starting CloudSim server in background..."
mvn exec:java -Dexec.mainClass="org.ee782.CloudSimSocketServer" > ../../results/logs/cloudsim_server.log 2>&1 &
SERVER_PID=$!
echo "CloudSim server started with PID: $SERVER_PID"
sleep 5  # Wait for server to start

echo ""
echo "Step 4: Activating Python environment..."
cd ../..
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh
conda activate ee782 || {
    echo "Creating conda environment ee782..."
    conda create -n ee782 python=3.9 -y
    conda activate ee782
    pip install -r requirements.txt
}

echo ""
echo "Step 5: Starting DQN training..."
cd algorithms/rl/dqn
python train_dqn.py --episodes 100 --save-freq 25 --log-freq 10

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Evaluate model: python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_*.pth"
echo "2. Visualize results: python ../../../utils/visualization.py --log-path ../../../results/models/dqn_cloudsim_*_log.json"
echo ""
echo "Stopping CloudSim server..."
kill $SERVER_PID 2>/dev/null || true

echo "Done!"
