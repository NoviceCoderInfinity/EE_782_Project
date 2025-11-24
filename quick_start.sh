#!/bin/bash

# Quick start script for RL CloudSim Training
# This script sets up the environment and starts training
# Usage: ./quick_start.sh [qlearning|dqn]

set -e

ALGORITHM=${1:-dqn}

echo "============================================"
echo "RL CloudSim Training Quick Start"
echo "Algorithm: $ALGORITHM"
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
echo "Step 5: Starting $ALGORITHM training..."
cd algorithms/rl

if [ "$ALGORITHM" = "qlearning" ]; then
    cd qlearning
    python train_qlearning.py --episodes 100 --lr 0.1 --gamma 0.9
    
    echo ""
    echo "============================================"
    echo "Q-Learning Training completed!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model: python test_qlearning.py --model ../../../results/qlearning/models/qlearning_final.pkl --episodes 100"
    echo "2. Check results in: results/qlearning/logs/"
else
    cd dqn
    python train_dqn.py --episodes 100 --save-freq 25 --log-freq 10
    
    echo ""
    echo "============================================"
    echo "DQN Training completed!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate model: python test_dqn.py --model-path ../../../results/models/dqn_cloudsim_*.pth"
    echo "2. Visualize results: python ../../../utils/visualization.py --log-path ../../../results/models/dqn_cloudsim_*_log.json"
fi

echo ""
echo "Stopping CloudSim server..."
kill $SERVER_PID 2>/dev/null || true

echo "Done!"
