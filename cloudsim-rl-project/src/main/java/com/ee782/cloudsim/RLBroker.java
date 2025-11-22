package com.ee782.cloudsim;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.vms.Vm;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;

/**
 * RLBroker - Reinforcement Learning Broker with Socket Communication
 * Phase 3: Deep RL Integration
 * 
 * This broker acts as a socket server that:
 * 1. Receives actions from Python RL agents (DQN/PPO)
 * 2. Executes actions in CloudSim simulation
 * 3. Returns state observations and rewards
 * 
 * Communication Protocol:
 * - Python sends: {"command": "reset"} or {"command": "step", "action": 5}
 * - Java responds: {"status": "success", "vm_states": [...], "reward": 0.5, ...}
 */
public class RLBroker extends DatacenterBrokerSimple {
    
    private ServerSocket serverSocket;
    private Socket clientSocket;
    private BufferedReader reader;
    private PrintWriter writer;
    private int port;
    private boolean serverRunning;
    
    // Simulation state
    private int currentCloudletIndex;
    private List<Cloudlet> pendingCloudlets;
    private int cloudletsCompleted;
    private double totalWaitingTime;
    private double totalResponseTime;
    private int slaViolations;
    
    // Configuration
    private static final double SLA_THRESHOLD = 100.0; // seconds
    
    /**
     * Constructor
     * @param simulation CloudSim simulation instance
     * @param port Port for socket server
     */
    public RLBroker(CloudSimPlus simulation, int port) {
        super(simulation);
        this.port = port;
        this.serverRunning = false;
        this.currentCloudletIndex = 0;
        this.pendingCloudlets = new ArrayList<>();
        this.cloudletsCompleted = 0;
        this.totalWaitingTime = 0.0;
        this.totalResponseTime = 0.0;
        this.slaViolations = 0;
    }
    
    /**
     * Start socket server and wait for Python client connection
     */
    public void startServer() {
        try {
            serverSocket = new ServerSocket(port);
            serverRunning = true;
            System.out.printf("%n✓ Socket server started on port %d%n", port);
            System.out.println("⏳ Waiting for Python client connection...");
            
            clientSocket = serverSocket.accept();
            reader = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            writer = new PrintWriter(clientSocket.getOutputStream(), true);
            
            System.out.printf("✓ Python client connected from %s%n%n", 
                clientSocket.getInetAddress().getHostAddress());
            
        } catch (IOException e) {
            System.err.println("✗ Server error: " + e.getMessage());
            e.printStackTrace();
            serverRunning = false;
        }
    }
    
    /**
     * Stop socket server
     */
    public void stopServer() {
        serverRunning = false;
        try {
            if (reader != null) reader.close();
            if (writer != null) writer.close();
            if (clientSocket != null) clientSocket.close();
            if (serverSocket != null) serverSocket.close();
            System.out.println("%n✓ Socket server stopped");
        } catch (IOException e) {
            System.err.println("✗ Error closing server: " + e.getMessage());
        }
    }
    
    /**
     * Handle incoming message from Python client
     */
    @SuppressWarnings("unchecked")
    public void handleMessage() {
        if (!serverRunning || reader == null) {
            return;
        }
        
        try {
            String messageStr = reader.readLine();
            if (messageStr == null) {
                System.out.println("⚠ Client disconnected");
                stopServer();
                return;
            }
            
            // Parse JSON message
            JSONParser parser = new JSONParser();
            JSONObject message = (JSONObject) parser.parse(messageStr);
            
            String command = (String) message.get("command");
            
            JSONObject response = new JSONObject();
            
            if ("reset".equals(command)) {
                // Reset simulation
                response = handleReset();
                
            } else if ("step".equals(command)) {
                // Execute step with given action
                Long actionLong = (Long) message.get("action");
                int action = actionLong.intValue();
                response = handleStep(action);
                
            } else {
                response.put("status", "error");
                response.put("message", "Unknown command: " + command);
            }
            
            // Send response
            String responseStr = response.toJSONString() + "\n";
            writer.print(responseStr);
            writer.flush();
            
        } catch (Exception e) {
            System.err.println("✗ Message handling error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Handle reset command
     */
    @SuppressWarnings("unchecked")
    private JSONObject handleReset() {
        JSONObject response = new JSONObject();
        
        // Reset state
        currentCloudletIndex = 0;
        cloudletsCompleted = 0;
        totalWaitingTime = 0.0;
        totalResponseTime = 0.0;
        slaViolations = 0;
        
        // Get initial VM states
        JSONArray vmStates = getVmStates();
        
        response.put("status", "success");
        response.put("vm_states", vmStates);
        response.put("done", false);
        response.put("cloudlets_completed", 0);
        response.put("message", "Environment reset");
        
        return response;
    }
    
    /**
     * Handle step command with action (VM selection)
     */
    @SuppressWarnings("unchecked")
    private JSONObject handleStep(int action) {
        JSONObject response = new JSONObject();
        
        // Get current VM list (exec or waiting)
        List<Vm> vmList = !getVmExecList().isEmpty() ? getVmExecList() : getVmWaitingList();
        
        // Validate action
        if (action < 0 || action >= vmList.size()) {
            response.put("status", "error");
            response.put("message", "Invalid action: " + action);
            return response;
        }
        
        // Get next cloudlet to schedule
        if (currentCloudletIndex >= pendingCloudlets.size()) {
            // No more cloudlets - episode done
            response.put("status", "success");
            response.put("done", true);
            response.put("vm_states", getVmStates());
            response.put("cloudlets_completed", cloudletsCompleted);
            response.put("response_time", 0.0);
            response.put("waiting_time", 0.0);
            response.put("load_imbalance", calculateLoadImbalance());
            response.put("sla_violation", 0);
            return response;
        }
        
        Cloudlet cloudlet = pendingCloudlets.get(currentCloudletIndex);
        Vm selectedVm = vmList.get(action);
        
        // Bind cloudlet to selected VM
        bindCloudletToVm(cloudlet, selectedVm);
        
        // Calculate metrics for reward
        double responseTime = estimateResponseTime(cloudlet, selectedVm);
        double waitingTime = calculateWaitingTime(selectedVm);
        int slaViolation = (responseTime > SLA_THRESHOLD) ? 1 : 0;
        
        // Update statistics
        totalResponseTime += responseTime;
        totalWaitingTime += waitingTime;
        slaViolations += slaViolation;
        cloudletsCompleted++;
        currentCloudletIndex++;
        
        // Get new VM states
        JSONArray vmStates = getVmStates();
        
        // Check if episode is done
        boolean done = currentCloudletIndex >= pendingCloudlets.size();
        
        // Prepare response
        response.put("status", "success");
        response.put("vm_states", vmStates);
        response.put("done", done);
        response.put("cloudlets_completed", cloudletsCompleted);
        response.put("response_time", responseTime);
        response.put("waiting_time", waitingTime);
        response.put("load_imbalance", calculateLoadImbalance());
        response.put("sla_violation", slaViolation);
        
        return response;
    }
    
    /**
     * Get current state of all VMs
     */
    @SuppressWarnings("unchecked")
    private JSONArray getVmStates() {
        JSONArray vmStates = new JSONArray();
        
        // Use getVmExecList if simulation is running, otherwise use getVmWaitingList
        List<Vm> vmList = !getVmExecList().isEmpty() ? getVmExecList() : getVmWaitingList();
        
        for (Vm vm : vmList) {
            JSONObject vmState = new JSONObject();
            
            // CPU utilization (0-1)
            double cpuUtil = vm.getCpuPercentUtilization();
            
            // RAM utilization (0-1)
            double ramUtil = vm.getRam().getPercentUtilization();
            
            // Bandwidth utilization (0-1)
            double bwUtil = vm.getBw().getPercentUtilization();
            
            vmState.put("cpu", cpuUtil);
            vmState.put("ram", ramUtil);
            vmState.put("bw", bwUtil);
            
            vmStates.add(vmState);
        }
        
        return vmStates;
    }
    
    /**
     * Estimate response time for cloudlet on selected VM
     */
    private double estimateResponseTime(Cloudlet cloudlet, Vm vm) {
        // Estimate execution time based on cloudlet length and VM MIPS
        double executionTime = (double) cloudlet.getLength() / (vm.getMips() * vm.getPesNumber());
        
        // Add waiting time based on current VM load
        double waitingTime = calculateWaitingTime(vm);
        
        return executionTime + waitingTime;
    }
    
    /**
     * Calculate waiting time based on VM's current cloudlet queue
     */
    private double calculateWaitingTime(Vm vm) {
        // Estimate based on number of cloudlets already assigned
        int queueSize = vm.getCloudletScheduler().getCloudletExecList().size();
        return queueSize * 5.0; // Rough estimate: 5 seconds per queued cloudlet
    }
    
    /**
     * Calculate load imbalance across all VMs
     */
    private double calculateLoadImbalance() {
        List<Vm> vmList = !getVmExecList().isEmpty() ? getVmExecList() : getVmWaitingList();
        
        if (vmList.isEmpty()) {
            return 0.0;
        }
        
        int maxLoad = 0;
        int minLoad = Integer.MAX_VALUE;
        
        for (Vm vm : vmList) {
            int load = vm.getCloudletScheduler().getCloudletExecList().size();
            maxLoad = Math.max(maxLoad, load);
            minLoad = Math.min(minLoad, load);
        }
        
        if (maxLoad == 0) {
            return 0.0;
        }
        
        return (double) (maxLoad - minLoad) / maxLoad;
    }
    
    /**
     * Set pending cloudlets for this episode
     */
    public void setPendingCloudlets(List<Cloudlet> cloudlets) {
        this.pendingCloudlets = new ArrayList<>(cloudlets);
        this.currentCloudletIndex = 0;
    }
    
    /**
     * Check if server is running
     */
    public boolean isServerRunning() {
        return serverRunning;
    }
    
    /**
     * Get statistics
     */
    public void printStatistics() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("RL BROKER STATISTICS");
        System.out.println("=".repeat(80));
        
        System.out.printf("%nCloudlets Completed: %d%n", cloudletsCompleted);
        
        if (cloudletsCompleted > 0) {
            System.out.printf("Average Response Time: %.2f seconds%n", 
                totalResponseTime / cloudletsCompleted);
            System.out.printf("Average Waiting Time: %.2f seconds%n", 
                totalWaitingTime / cloudletsCompleted);
            System.out.printf("SLA Violations: %d (%.1f%%)%n", 
                slaViolations, 
                (slaViolations * 100.0 / cloudletsCompleted));
        }
        
        System.out.printf("Load Imbalance: %.2f%%%n", calculateLoadImbalance() * 100);
        
        System.out.println("=".repeat(80));
    }
}
