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
 * CloudSim Q-Learning Simulation with Google Trace Workload
 * Phase 2: Integrated with realistic trace data
 * 
 * This simulation:
 * 1. Loads realistic workload from Google Cluster traces
 * 2. Uses Q-Learning broker for VM selection
 * 3. Evaluates performance with real-world task characteristics
 * 
 * Updates from Phase 1:
 * - Replaced random cloudlets with trace-based workload
 * - Dynamic task submission based on arrival times
 * - Realistic resource requirements
 */
public class CloudSimTraceQLearningSimulation {
    
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
    
    private CloudSimPlus simulation;
    private Datacenter datacenter;
    private QLearningBroker broker;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletList;
    
    public static void main(String[] args) {
        new CloudSimTraceQLearningSimulation();
    }
    
    public CloudSimTraceQLearningSimulation() {
        System.out.println("=".repeat(80));
        System.out.println("CloudSim Plus Q-Learning Simulation with Google Trace Workload");
        System.out.println("Phase 2: Data Integration and Realistic Workload Evaluation");
        System.out.println("=".repeat(80));
        
        // Initialize simulation
        simulation = new CloudSimPlus();
        
        // Create infrastructure
        datacenter = createDatacenter();
        System.out.printf("%n✓ Created datacenter with %d hosts%n", HOSTS);
        
        // Create Q-Learning broker
        broker = new QLearningBroker(simulation);
        System.out.printf("✓ Created Q-Learning broker (α=%.2f, γ=%.2f, ε=%.2f)%n",
            broker.getLearningRate(), broker.getDiscountFactor(), broker.getEpsilon());
        
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
        
        // Submit cloudlets to broker
        broker.submitCloudletList(cloudletList);
        
        // Run simulation
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Starting Simulation...");
        System.out.println("=".repeat(80) + "\n");
        
        simulation.start();
        
        // Print results
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        printResults(finishedCloudlets);
        
        // Print Q-Learning statistics
        broker.printQLearningStatistics();
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
     * Print simulation results
     */
    private void printResults(List<Cloudlet> cloudletList) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("SIMULATION RESULTS");
        System.out.println("=".repeat(80));
        
        // Calculate metrics
        double totalWaitingTime = 0;
        double totalExecutionTime = 0;
        double totalFinishTime = 0;
        int completedCloudlets = 0;
        
        for (Cloudlet cloudlet : cloudletList) {
            if (cloudlet.isFinished()) {
                totalWaitingTime += cloudlet.getWaitingTime();
                totalExecutionTime += cloudlet.getActualCpuTime();
                totalFinishTime += cloudlet.getFinishTime();
                completedCloudlets++;
            }
        }
        
        if (completedCloudlets == 0) {
            System.out.println("✗ No cloudlets completed!");
            return;
        }
        
        double avgWaitingTime = totalWaitingTime / completedCloudlets;
        double avgExecutionTime = totalExecutionTime / completedCloudlets;
        double avgResponseTime = avgWaitingTime + avgExecutionTime;
        double maxFinishTime = cloudletList.stream()
            .filter(Cloudlet::isFinished)
            .mapToDouble(Cloudlet::getFinishTime)
            .max()
            .orElse(0);
        double throughput = completedCloudlets / maxFinishTime;
        
        System.out.printf("%nPerformance Metrics:%n");
        System.out.printf("  Total Cloudlets: %d%n", cloudletList.size());
        System.out.printf("  Completed: %d (%.1f%%)%n", 
            completedCloudlets, 
            (completedCloudlets * 100.0 / cloudletList.size()));
        System.out.printf("  Average Waiting Time: %.2f seconds%n", avgWaitingTime);
        System.out.printf("  Average Execution Time: %.2f seconds%n", avgExecutionTime);
        System.out.printf("  Average Response Time: %.2f seconds%n", avgResponseTime);
        System.out.printf("  Throughput: %.3f tasks/second%n", throughput);
        System.out.printf("  Simulation Time: %.2f seconds%n", maxFinishTime);
        
        // VM utilization
        System.out.printf("%nVM Utilization:%n");
        for (Vm vm : vmList) {
            double cpuUtilization = vm.getCpuPercentUtilization() * 100;
            int tasksProcessed = (int) vm.getCloudletScheduler().getCloudletFinishedList().size();
            System.out.printf("  VM #%d: %.1f%% CPU, %d tasks%n", 
                vm.getId(), cpuUtilization, tasksProcessed);
        }
        
        // Load balance metric
        long[] tasksPerVm = new long[VMS];
        for (Cloudlet cloudlet : cloudletList) {
            if (cloudlet.isFinished()) {
                int vmId = (int) cloudlet.getVm().getId();
                tasksPerVm[vmId]++;
            }
        }
        
        long maxTasks = 0;
        long minTasks = Long.MAX_VALUE;
        for (long tasks : tasksPerVm) {
            maxTasks = Math.max(maxTasks, tasks);
            minTasks = Math.min(minTasks, tasks);
        }
        
        double loadImbalance = maxTasks > 0 ? (double)(maxTasks - minTasks) / maxTasks : 0;
        System.out.printf("%nLoad Balance:%n");
        System.out.printf("  Max tasks per VM: %d%n", maxTasks);
        System.out.printf("  Min tasks per VM: %d%n", minTasks);
        System.out.printf("  Load Imbalance: %.2f%%%n", loadImbalance * 100);
        
        // Sample cloudlet details
        System.out.printf("%nSample Cloudlet Details (first 10):%n");
        System.out.println("  " + "-".repeat(76));
        System.out.printf("  %-8s %-10s %-8s %-10s %-10s %-12s %-8s%n",
            "ID", "Length", "PEs", "Wait(s)", "Exec(s)", "Finish(s)", "VM");
        System.out.println("  " + "-".repeat(76));
        
        for (int i = 0; i < Math.min(10, cloudletList.size()); i++) {
            Cloudlet c = cloudletList.get(i);
            if (c.isFinished()) {
                System.out.printf("  %-8d %-10s %-8d %-10.2f %-10.2f %-12.2f %-8d%n",
                    c.getId(),
                    String.format("%,d", c.getLength()),
                    c.getPesNumber(),
                    c.getWaitingTime(),
                    c.getActualCpuTime(),
                    c.getFinishTime(),
                    c.getVm().getId());
            }
        }
        System.out.println("  " + "-".repeat(76));
        
        System.out.println("\n" + "=".repeat(80));
    }
    
    /**
     * Get simulation instance (for testing)
     */
    public CloudSimPlus getSimulation() {
        return simulation;
    }
    
    /**
     * Get broker (for testing)
     */
    public QLearningBroker getBroker() {
        return broker;
    }
}
