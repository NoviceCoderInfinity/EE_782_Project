package com.ee782.cloudsim;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.vms.Vm;

import java.util.*;

/**
 * A DatacenterBroker that implements Q-Learning for VM selection
 */
public class QLearningBroker extends DatacenterBrokerSimple {
    
    // Q-Learning parameters
    private static final double LEARNING_RATE = 0.1;      // α
    private static final double DISCOUNT_FACTOR = 0.9;    // γ
    private static final double EPSILON = 0.2;            // ε for ε-greedy
    
    // Q-Table: Map<StateHash, Map<Action(VmId), QValue>>
    private Map<Integer, Map<Integer, Double>> qTable;
    
    // Track previous state and action for learning
    private int previousState = -1;
    private int previousAction = -1;
    
    private Random random;
    
    public QLearningBroker(CloudSimPlus simulation) {
        super(simulation);
        this.qTable = new HashMap<>();
        this.random = new Random(42); // Fixed seed for reproducibility
    }
    
    /**
     * Override the default VM selection policy to use Q-Learning
     */
    @Override
    protected Vm defaultVmMapper(Cloudlet cloudlet) {
        if (getVmExecList().isEmpty()) {
            return Vm.NULL;
        }
        
        // Get current state (VM loads)
        int currentState = getCurrentStateHash();
        
        // Select action using ε-greedy policy
        int selectedVmIndex = selectAction(currentState);
        Vm selectedVm = getVmExecList().get(selectedVmIndex);
        
        // If we have a previous state, update Q-table
        if (previousState != -1) {
            double reward = calculateReward(cloudlet);
            updateQTable(previousState, previousAction, reward, currentState);
        }
        
        // Store current state and action for next update
        previousState = currentState;
        previousAction = selectedVmIndex;
        
        return selectedVm;
    }
    
    /**
     * Hash the current state based on VM loads
     * State = [VM1_CPU_Usage, VM2_CPU_Usage, ..., VMn_CPU_Usage]
     */
    private int getCurrentStateHash() {
        StringBuilder stateBuilder = new StringBuilder();
        
        for (Vm vm : getVmExecList()) {
            // Discretize CPU usage into buckets (0-25%, 25-50%, 50-75%, 75-100%)
            double cpuUsage = vm.getCpuPercentUtilization() * 100;
            int bucket = (int) (cpuUsage / 25.0);
            bucket = Math.min(bucket, 3); // Cap at 3
            stateBuilder.append(bucket);
        }
        
        return stateBuilder.toString().hashCode();
    }
    
    /**
     * Select action using ε-greedy policy
     */
    private int selectAction(int state) {
        // Explore: random action
        if (random.nextDouble() < EPSILON) {
            return random.nextInt(getVmExecList().size());
        }
        
        // Exploit: choose best action from Q-table
        Map<Integer, Double> actionValues = qTable.getOrDefault(state, new HashMap<>());
        
        int bestAction = 0;
        double bestValue = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < getVmExecList().size(); i++) {
            double qValue = actionValues.getOrDefault(i, 0.0);
            if (qValue > bestValue) {
                bestValue = qValue;
                bestAction = i;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Calculate reward based on VM load balancing
     * Reward = 1 / (1 + VM_CPU_Usage)
     * Higher reward for assigning to less loaded VMs
     */
    private double calculateReward(Cloudlet cloudlet) {
        if (previousAction == -1 || getVmExecList().isEmpty()) {
            return 0.0;
        }
        
        Vm selectedVm = getVmExecList().get(previousAction);
        double cpuUsage = selectedVm.getCpuPercentUtilization();
        
        // Reward inversely proportional to CPU usage
        // Less loaded VM = higher reward
        double reward = 1.0 / (1.0 + cpuUsage);
        
        // Penalty for overloaded VMs (> 80% usage)
        if (cpuUsage > 0.8) {
            reward -= 0.5;
        }
        
        return reward;
    }
    
    /**
     * Update Q-Table using Q-Learning update rule:
     * Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
     */
    private void updateQTable(int state, int action, double reward, int nextState) {
        // Get or initialize Q-values for current state
        Map<Integer, Double> stateActions = qTable.computeIfAbsent(state, k -> new HashMap<>());
        double currentQ = stateActions.getOrDefault(action, 0.0);
        
        // Find max Q-value for next state
        Map<Integer, Double> nextStateActions = qTable.getOrDefault(nextState, new HashMap<>());
        double maxNextQ = 0.0;
        for (int i = 0; i < getVmExecList().size(); i++) {
            double qValue = nextStateActions.getOrDefault(i, 0.0);
            maxNextQ = Math.max(maxNextQ, qValue);
        }
        
        // Q-Learning update
        double newQ = currentQ + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maxNextQ - currentQ);
        stateActions.put(action, newQ);
    }
    
    /**
     * Print Q-Table statistics for analysis
     */
    public void printQTableStats() {
        System.out.println("\n=== Q-Table Statistics ===");
        System.out.println("Total states learned: " + qTable.size());
        
        if (!qTable.isEmpty()) {
            double avgQValue = qTable.values().stream()
                .flatMap(actions -> actions.values().stream())
                .mapToDouble(Double::doubleValue)
                .average()
                .orElse(0.0);
            System.out.println("Average Q-value: " + String.format("%.4f", avgQValue));
        }
        System.out.println("=========================\n");
    }
    
    // Getter methods for Q-Learning parameters
    public double getLearningRate() {
        return LEARNING_RATE;
    }
    
    public double getDiscountFactor() {
        return DISCOUNT_FACTOR;
    }
    
    public double getEpsilon() {
        return EPSILON;
    }
    
    // Alias for printQTableStats
    public void printQLearningStatistics() {
        printQTableStats();
    }
}
