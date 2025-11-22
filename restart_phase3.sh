#!/bin/bash

# Quick fix script for Phase 3 JSON serialization issue

echo "========================================================================"
echo "Phase 3 - JSON Serialization Fix Applied"
echo "========================================================================"
echo ""
echo "Issue Fixed: numpy int64/int32 types were not JSON serializable"
echo "Solution: Added _convert_numpy_types() method to cloudsim_gym_env.py"
echo ""
echo "========================================================================"
echo "RESTARTING TRAINING"
echo "========================================================================"
echo ""

# Kill any existing Java processes
echo "1. Stopping any existing Java CloudSim servers..."
pkill -f "CloudSimRLSimulation" 2>/dev/null
sleep 2
echo "   ✓ Cleaned up"
echo ""

# Start Java server
echo "2. Starting Java CloudSim server..."
cd /home/anupam/Desktop/EE_782_Project/cloudsim-rl-project
mvn exec:java -Dexec.mainClass="com.ee782.cloudsim.CloudSimRLSimulation" > /tmp/cloudsim_server.log 2>&1 &
JAVA_PID=$!
echo "   Java server PID: $JAVA_PID"
echo "   Log file: /tmp/cloudsim_server.log"
echo ""

echo "3. Waiting for server to start (15 seconds)..."
sleep 15
echo ""

# Check if server is running
if ps -p $JAVA_PID > /dev/null; then
    echo "   ✓ Java server is running"
    echo ""
    
    # Show last few lines of log
    echo "   Server status:"
    tail -n 5 /tmp/cloudsim_server.log | sed 's/^/   /'
    echo ""
else
    echo "   ✗ Java server failed to start!"
    echo "   Check the log file: tail -f /tmp/cloudsim_server.log"
    exit 1
fi

echo "========================================================================"
echo "READY TO TRAIN"
echo "========================================================================"
echo ""
echo "In another terminal, run:"
echo "  cd ~/Desktop/EE_782_Project/python-rl"
echo "  conda activate cloudsim_rl"
echo "  python train_dqn.py"
echo ""
echo "Or to stop the Java server and restart manually:"
echo "  kill $JAVA_PID"
echo ""
echo "To view live server logs:"
echo "  tail -f /tmp/cloudsim_server.log"
echo ""
echo "========================================================================"
