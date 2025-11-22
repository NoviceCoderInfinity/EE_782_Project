#!/bin/bash

# Phase 3 Testing Script

echo "========================================================================"
echo "Phase 3 DQN Training Test"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Start CloudSim Java server in background"
echo "  2. Wait for server to initialize"
echo "  3. Run Python DQN training"
echo ""
echo "Press Ctrl+C to stop at any time"
echo "========================================================================"
echo ""

# Start Java server in background
echo "Starting Java CloudSim server..."
cd /home/anupam/Desktop/EE_782_Project/cloudsim-rl-project
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation" > /tmp/cloudsim_server.log 2>&1 &
JAVA_PID=$!

echo "  Java server PID: $JAVA_PID"
echo "  Waiting for server initialization (10 seconds)..."
sleep 10

# Check if server is running
if ps -p $JAVA_PID > /dev/null; then
    echo "  ✓ Java server started successfully"
    echo ""
else
    echo "  ✗ Java server failed to start!"
    echo "  Check /tmp/cloudsim_server.log for errors"
    exit 1
fi

# Run Python training
echo "Starting Python DQN training..."
cd /home/anupam/Desktop/EE_782_Project/python-rl
conda activate cloudsim_rl
python train_dqn.py

# Cleanup
echo ""
echo "Stopping Java server..."
kill $JAVA_PID 2>/dev/null
wait $JAVA_PID 2>/dev/null

echo ""
echo "========================================================================"
echo "Test complete!"
echo "========================================================================"
