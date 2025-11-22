package com.ee782.cloudsim;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * CloudSim RL Simulation with Socket Server
 * Phase 3: Deep RL Integration (DQN/PPO)
 * 
 * This simulation:
 * 1. Creates CloudSim infrastructure (datacenter, hosts, VMs)
 * 2. Loads realistic workload from trace data
 * 3. Starts RLBroker socket server for Python communication
 * 4. Lets Python RL agents (DQN/PPO) control task scheduling
 * 
 * The simulation acts as the environment, and Python agents act as controllers.
 */
public class CloudSimRLSimulation {
    
    // Infrastructure configuration
    private static final int HOSTS = 8;
    private static final int HOST_PES = 8;
    private static final int HOST_MIPS = 10000;
    private static final int HOST_RAM = 16384; // MB
    private static final long HOST_STORAGE = 1000000; // MB
    private static final int HOST_BW = 10000; // Mbps
    
    private static final int VMS = 30;
    private static final int VM_PES = 2;
    private static final int VM_MIPS = 1000;
    private static final int VM_RAM = 2048; // MB
    private static final long VM_STORAGE = 10000; // MB
    private static final int VM_BW = 1000; // Mbps
    
    private static final int SOCKET_PORT = 5555;
    
    private CloudSimPlus simulation;
    private Datacenter datacenter;
    private RLBroker broker;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletList;
    
    public static void main(String[] args) {
        new CloudSimRLSimulation();
    }
    
    public CloudSimRLSimulation() {
        System.out.println("=".repeat(80));
        System.out.println("CloudSim Plus RL Simulation - Phase 3");
        System.out.println("Deep Reinforcement Learning Integration (DQN/PPO)");
        System.out.println("=".repeat(80));
        
        // Initialize simulation
        simulation = new CloudSimPlus();
        
        // Create infrastructure
        datacenter = createDatacenter();
        System.out.printf("%n✓ Created datacenter with %d hosts%n", HOSTS);
        
        // Create RL broker with socket server
        broker = new RLBroker(simulation, SOCKET_PORT);
        System.out.printf("✓ Created RL broker%n");
        
        // Create VMs
        vmList = createVms();
        broker.submitVmList(vmList);
        System.out.printf("✓ Created %d VMs%n", VMS);
        
        // Load cloudlets from Google trace
        try {
            cloudletList = loadCloudletsFromTrace();
            System.out.printf("✓ Loaded %d cloudlets from trace data%n", cloudletList.size());
        } catch (IOException e) {
            System.err.println("✗ Error loading trace data: " + e.getMessage());
            System.err.println("  Make sure synthetic_workload.csv is in src/main/resources/");
            System.exit(1);
            return;
        }
        
        // Set cloudlets for broker (but don't submit yet - Python will control scheduling)
        broker.setPendingCloudlets(cloudletList);
        System.out.printf("✓ Cloudlets ready for RL scheduling%n");
        
        // Start socket server
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Starting Socket Server for Python RL Agent");
        System.out.println("=".repeat(80));
        broker.startServer();
        
        // Run simulation with RL control
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Simulation Running - Controlled by Python RL Agent");
        System.out.println("=".repeat(80) + "\n");
        
        // The simulation will run and be controlled by Python agent via socket
        // Messages are handled in broker.handleMessage() called from Python
        
        // Note: In a real implementation, you'd need a separate thread or event loop
        // to handle socket messages while simulation runs. This is a simplified version.
        
        // For now, just keep server running and wait for Python to connect
        System.out.println("⏳ Waiting for Python RL agent to connect and control simulation...");
        System.out.println("   Run: python train_dqn.py or python train_ppo.py");
        
        // Keep running until interrupted
        try {
            // Simulation loop controlled by Python agent
            while (broker.isServerRunning()) {
                broker.handleMessage();
                Thread.sleep(100); // Small delay between message checks
            }
        } catch (InterruptedException e) {
            System.err.println("Simulation interrupted");
        } finally {
            broker.stopServer();
            printFinalStatistics();
        }
    }
    
    /**
     * Load cloudlets from Google Cluster trace data
     */
    private List<Cloudlet> loadCloudletsFromTrace() throws IOException {
        System.out.println("\n" + "-".repeat(80));
        System.out.println("Loading Workload from Google Cluster Trace");
        System.out.println("-".repeat(80));
        
        // Read from resources folder
        GoogleTraceReader reader = new GoogleTraceReader("synthetic_workload.csv");
        List<Cloudlet> cloudlets = reader.loadCloudlets();
        
        System.out.println("-".repeat(80));
        
        return cloudlets;
    }
    
    /**
     * Create datacenter with hosts
     */
    private Datacenter createDatacenter() {
        List<Host> hostList = new ArrayList<>();
        
        for (int i = 0; i < HOSTS; i++) {
            Host host = createHost();
            hostList.add(host);
        }
        
        return new DatacenterSimple(simulation, hostList);
    }
    
    /**
     * Create a single host
     */
    private Host createHost() {
        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < HOST_PES; i++) {
            peList.add(new PeSimple(HOST_MIPS));
        }
        
        return new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList);
    }
    
    /**
     * Create VMs
     */
    private List<Vm> createVms() {
        List<Vm> vms = new ArrayList<>();
        
        for (int i = 0; i < VMS; i++) {
            Vm vm = new VmSimple(i, VM_MIPS, VM_PES);
            vm.setRam(VM_RAM);
            vm.setBw(VM_BW);
            vm.setSize(VM_STORAGE);
            vms.add(vm);
        }
        
        return vms;
    }
    
    /**
     * Print final statistics
     */
    private void printFinalStatistics() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SIMULATION COMPLETE");
        System.out.println("=".repeat(80));
        
        // Print broker statistics
        broker.printStatistics();
        
        // Print final cloudlet results
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        
        if (!finishedCloudlets.isEmpty()) {
            System.out.printf("%nCloudlet Execution Summary:%n");
            System.out.printf("  Total: %d%n", cloudletList.size());
            System.out.printf("  Finished: %d%n", finishedCloudlets.size());
            System.out.printf("  Success Rate: %.1f%%%n", 
                (finishedCloudlets.size() * 100.0 / cloudletList.size()));
        }
        
        System.out.println("=".repeat(80));
    }
    
    /**
     * Helper method to check if server is running
     */
    public boolean isRunning() {
        return broker != null && broker.isServerRunning();
    }
}
