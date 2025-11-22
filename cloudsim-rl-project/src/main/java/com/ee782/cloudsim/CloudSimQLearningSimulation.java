package com.ee782.cloudsim;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.builders.tables.CloudletsTableBuilder;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Phase 1: CloudSim Plus simulation with Q-Learning based load balancing
 * 
 * This simulation creates:
 * - 1 Datacenter
 * - 8 Hosts with defined resources
 * - 30 VMs
 * - 100 Cloudlets (tasks) with varying lengths
 */
public class CloudSimQLearningSimulation {
    
    // Simulation parameters
    private static final int NUM_HOSTS = 8;
    private static final int NUM_VMS = 30;
    private static final int NUM_CLOUDLETS = 100;
    
    // Host specifications
    private static final long HOST_RAM = 16384; // MB
    private static final long HOST_STORAGE = 1000000; // MB
    private static final long HOST_BW = 10000; // Mbps
    private static final int HOST_PES = 8; // Processing Elements
    private static final long HOST_MIPS = 10000; // Million Instructions Per Second per PE
    
    // VM specifications
    private static final long VM_RAM = 2048; // MB
    private static final long VM_STORAGE = 10000; // MB
    private static final long VM_BW = 1000; // Mbps
    private static final int VM_PES = 2;
    private static final long VM_MIPS = 1000;
    
    // Cloudlet specifications
    private static final int CLOUDLET_PES = 1;
    private static final long CLOUDLET_LENGTH_MIN = 10000; // MI
    private static final long CLOUDLET_LENGTH_MAX = 50000; // MI
    
    private CloudSimPlus simulation;
    private List<Host> hostList;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletList;
    private QLearningBroker broker;
    private Random random;
    
    public static void main(String[] args) {
        new CloudSimQLearningSimulation();
    }
    
    public CloudSimQLearningSimulation() {
        System.out.println("=================================================");
        System.out.println("  CloudSim Plus - Q-Learning Load Balancing");
        System.out.println("=================================================\n");
        
        this.random = new Random(42); // Fixed seed for reproducibility
        
        // Initialize CloudSim Plus
        simulation = new CloudSimPlus();
        
        // Create Datacenter with Hosts
        Datacenter datacenter = createDatacenter();
        
        // Create Q-Learning Broker
        broker = new QLearningBroker(simulation);
        broker.setVmDestructionDelay(10.0);
        
        // Create VMs and Cloudlets
        vmList = createVms();
        cloudletList = createCloudlets();
        
        // Submit VMs and Cloudlets to broker
        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);
        
        // Run simulation
        System.out.println("Starting simulation...\n");
        simulation.start();
        
        // Print results
        printResults();
        
        System.out.println("\n=================================================");
        System.out.println("  Simulation completed successfully!");
        System.out.println("=================================================");
    }
    
    /**
     * Create Datacenter with Hosts
     */
    private Datacenter createDatacenter() {
        hostList = new ArrayList<>();
        
        for (int i = 0; i < NUM_HOSTS; i++) {
            Host host = createHost(i);
            hostList.add(host);
        }
        
        Datacenter datacenter = new DatacenterSimple(simulation, hostList);
        datacenter.setSchedulingInterval(2);
        
        System.out.println("Created Datacenter with " + NUM_HOSTS + " hosts");
        return datacenter;
    }
    
    /**
     * Create a single Host with specified resources
     */
    private Host createHost(int id) {
        List<Pe> peList = new ArrayList<>();
        
        // Create Processing Elements (Cores)
        for (int i = 0; i < HOST_PES; i++) {
            peList.add(new PeSimple(HOST_MIPS));
        }
        
        return new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList);
    }
    
    /**
     * Create VMs
     */
    private List<Vm> createVms() {
        List<Vm> list = new ArrayList<>();
        
        for (int i = 0; i < NUM_VMS; i++) {
            Vm vm = new VmSimple(i, VM_MIPS, VM_PES);
            vm.setRam(VM_RAM)
              .setBw(VM_BW)
              .setSize(VM_STORAGE);
            list.add(vm);
        }
        
        System.out.println("Created " + NUM_VMS + " VMs");
        return list;
    }
    
    /**
     * Create Cloudlets with random lengths
     */
    private List<Cloudlet> createCloudlets() {
        List<Cloudlet> list = new ArrayList<>();
        
        // Dynamic utilization model (CPU usage varies over time)
        UtilizationModelDynamic utilizationModel = new UtilizationModelDynamic(0.3);
        
        for (int i = 0; i < NUM_CLOUDLETS; i++) {
            // Random cloudlet length for variety
            long length = CLOUDLET_LENGTH_MIN + 
                         random.nextInt((int)(CLOUDLET_LENGTH_MAX - CLOUDLET_LENGTH_MIN));
            
            Cloudlet cloudlet = new CloudletSimple(i, length, CLOUDLET_PES);
            cloudlet.setUtilizationModelCpu(new UtilizationModelFull())
                    .setUtilizationModelRam(utilizationModel)
                    .setUtilizationModelBw(utilizationModel);
            
            list.add(cloudlet);
        }
        
        System.out.println("Created " + NUM_CLOUDLETS + " cloudlets with varying lengths\n");
        return list;
    }
    
    /**
     * Print simulation results and metrics
     */
    private void printResults() {
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        
        // Print cloudlet execution details
        new CloudletsTableBuilder(finishedCloudlets).build();
        
        // Calculate and print metrics
        calculateMetrics(finishedCloudlets);
        
        // Print Q-Learning statistics
        broker.printQTableStats();
    }
    
    /**
     * Calculate performance metrics
     */
    private void calculateMetrics(List<Cloudlet> cloudlets) {
        System.out.println("\n=== Performance Metrics ===");
        
        // 1. Average Response Time
        double totalResponseTime = 0.0;
        double totalWaitTime = 0.0;
        double totalExecTime = 0.0;
        
        for (Cloudlet cloudlet : cloudlets) {
            double responseTime = cloudlet.getFinishTime() - cloudlet.getSubmissionDelay();
            double waitTime = cloudlet.getWaitingTime();
            double execTime = cloudlet.getActualCpuTime();
            
            totalResponseTime += responseTime;
            totalWaitTime += waitTime;
            totalExecTime += execTime;
        }
        
        double avgResponseTime = totalResponseTime / cloudlets.size();
        double avgWaitTime = totalWaitTime / cloudlets.size();
        double avgExecTime = totalExecTime / cloudlets.size();
        
        System.out.println("Average Response Time: " + String.format("%.2f", avgResponseTime) + " seconds");
        System.out.println("Average Waiting Time: " + String.format("%.2f", avgWaitTime) + " seconds");
        System.out.println("Average Execution Time: " + String.format("%.2f", avgExecTime) + " seconds");
        
        // 2. Throughput
        double simulationTime = simulation.clock();
        double throughput = cloudlets.size() / simulationTime;
        System.out.println("Throughput: " + String.format("%.2f", throughput) + " tasks/second");
        
        // 3. VM Load Balancing (Standard Deviation)
        double[] vmLoads = new double[NUM_VMS];
        for (Cloudlet cloudlet : cloudlets) {
            int vmId = (int) cloudlet.getVm().getId();
            vmLoads[vmId]++;
        }
        
        double meanLoad = cloudlets.size() / (double) NUM_VMS;
        double sumSquaredDiff = 0.0;
        for (double load : vmLoads) {
            sumSquaredDiff += Math.pow(load - meanLoad, 2);
        }
        double stdDev = Math.sqrt(sumSquaredDiff / NUM_VMS);
        double imbalanceDegree = (stdDev / meanLoad) * 100;
        
        System.out.println("Load Imbalance Degree: " + String.format("%.2f", imbalanceDegree) + "%");
        System.out.println("Standard Deviation of VM loads: " + String.format("%.2f", stdDev));
        
        // 4. Resource Utilization
        double totalCpuUtilization = 0.0;
        for (Vm vm : vmList) {
            totalCpuUtilization += vm.getCpuPercentUtilization();
        }
        double avgCpuUtilization = (totalCpuUtilization / vmList.size()) * 100;
        System.out.println("Average CPU Utilization: " + String.format("%.2f", avgCpuUtilization) + "%");
        
        System.out.println("===========================\n");
    }
}
