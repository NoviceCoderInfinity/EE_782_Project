package org.ee782;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;

import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;

/**
 * CloudSim Plus Socket Server for RL Agent Communication
 * 
 * Protocol:
 * - Python sends: {"action": <vm_id>}
 * - Java responds: {"state": [load1, load2, ...], "reward": <float>, "done": <bool>, "info": {...}}
 */
public class CloudSimSocketServer {
    private static final int PORT = 9999;
    private static final int NUM_HOSTS = 5;
    private static final int NUM_VMS = 20;
    private static final int CLOUDLETS_PER_EPISODE = 100;
    
    private CloudSimPlus simulation;
    private DatacenterBroker broker;
    private List<Vm> vmList;
    private List<Cloudlet> cloudletQueue;
    private int currentCloudletIndex;
    private Map<Long, Double> vmLoads;
    private Gson gson;
    
    private double totalResponseTime;
    private int completedCloudlets;
    private int slaViolations;
    
    public CloudSimSocketServer() {
        this.gson = new Gson();
        this.vmLoads = new HashMap<>();
        this.cloudletQueue = new ArrayList<>();
    }
    
    public static void main(String[] args) {
        CloudSimSocketServer server = new CloudSimSocketServer();
        server.startServer();
    }
    
    public void startServer() {
        System.out.println("CloudSim Socket Server starting on port " + PORT);
        
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("Server ready. Waiting for Python client...");
            
            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("Client connected!");
                handleClient(clientSocket);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private void handleClient(Socket clientSocket) {
        try (
            BufferedReader in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            PrintWriter out = new PrintWriter(clientSocket.getOutputStream(), true)
        ) {
            String inputLine;
            while ((inputLine = in.readLine()) != null) {
                JsonObject request = gson.fromJson(inputLine, JsonObject.class);
                String command = request.get("command").getAsString();
                
                JsonObject response = new JsonObject();
                
                switch (command) {
                    case "reset":
                        response = reset();
                        break;
                    case "step":
                        int action = request.get("action").getAsInt();
                        response = step(action);
                        break;
                    case "close":
                        out.println(gson.toJson(response));
                        return;
                    default:
                        response.addProperty("error", "Unknown command");
                }
                
                out.println(gson.toJson(response));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    private JsonObject reset() {
        // Initialize CloudSim simulation
        simulation = new CloudSimPlus();
        broker = new DatacenterBrokerSimple(simulation);
        
        // Create datacenter
        List<Host> hostList = createHosts();
        Datacenter datacenter = new DatacenterSimple(simulation, hostList);
        
        // Create VMs
        vmList = createVms();
        broker.submitVmList(vmList);
        
        // Generate cloudlets
        cloudletQueue = generateCloudlets(CLOUDLETS_PER_EPISODE);
        currentCloudletIndex = 0;
        
        // Reset metrics
        totalResponseTime = 0;
        completedCloudlets = 0;
        slaViolations = 0;
        
        // Initialize VM loads
        for (Vm vm : vmList) {
            vmLoads.put(vm.getId(), 0.0);
        }
        
        JsonObject response = new JsonObject();
        response.add("state", gson.toJsonTree(getState()));
        response.addProperty("info", "Environment reset");
        
        return response;
    }
    
    private JsonObject step(int action) {
        JsonObject response = new JsonObject();
        
        // Check if valid action
        if (action < 0 || action >= vmList.size()) {
            response.addProperty("error", "Invalid action");
            return response;
        }
        
        // Check if episode is done
        if (currentCloudletIndex >= cloudletQueue.size()) {
            response.add("state", gson.toJsonTree(getState()));
            response.addProperty("reward", 0.0);
            response.addProperty("done", true);
            response.add("info", getEpisodeInfo());
            return response;
        }
        
        // Get current cloudlet and assign to VM
        Cloudlet cloudlet = cloudletQueue.get(currentCloudletIndex);
        Vm selectedVm = vmList.get(action);
        cloudlet.setVm(selectedVm);
        
        // Submit cloudlet
        broker.submitCloudlet(cloudlet);
        
        // Update VM load (estimated)
        double currentLoad = vmLoads.getOrDefault(selectedVm.getId(), 0.0);
        double cloudletLoad = (double) cloudlet.getLength() / selectedVm.getMips();
        vmLoads.put(selectedVm.getId(), currentLoad + cloudletLoad);
        
        // Run simulation for one step (until cloudlet starts)
        simulation.startSync();
        
        // Calculate reward
        double reward = calculateReward(cloudlet, selectedVm);
        
        // Update metrics
        if (cloudlet.isFinished()) {
            double responseTime = cloudlet.getFinishTime() - cloudlet.getSubmitTime();
            totalResponseTime += responseTime;
            completedCloudlets++;
            
            // Check SLA violation (assume SLA is 2x optimal time)
            double optimalTime = cloudlet.getLength() / selectedVm.getMips();
            if (responseTime > 2 * optimalTime) {
                slaViolations++;
            }
            
            // Reduce VM load
            vmLoads.put(selectedVm.getId(), Math.max(0, vmLoads.get(selectedVm.getId()) - cloudletLoad));
        }
        
        currentCloudletIndex++;
        
        // Prepare response
        response.add("state", gson.toJsonTree(getState()));
        response.addProperty("reward", reward);
        response.addProperty("done", currentCloudletIndex >= cloudletQueue.size());
        response.add("info", getStepInfo());
        
        return response;
    }
    
    private double[] getState() {
        // State: [vm_load_1, vm_load_2, ..., vm_load_n, next_cloudlet_length_normalized]
        double[] state = new double[vmList.size() + 1];
        
        for (int i = 0; i < vmList.size(); i++) {
            Vm vm = vmList.get(i);
            // Normalize load by VM MIPS
            state[i] = vmLoads.getOrDefault(vm.getId(), 0.0) / vm.getMips();
        }
        
        // Next cloudlet length (normalized)
        if (currentCloudletIndex < cloudletQueue.size()) {
            Cloudlet nextCloudlet = cloudletQueue.get(currentCloudletIndex);
            state[vmList.size()] = nextCloudlet.getLength() / 100000.0; // Normalize
        } else {
            state[vmList.size()] = 0.0;
        }
        
        return state;
    }
    
    private double calculateReward(Cloudlet cloudlet, Vm vm) {
        // Multi-objective reward: minimize response time and balance load
        
        // Response time component (negative because we want to minimize)
        double estimatedResponseTime = cloudlet.getLength() / vm.getMips();
        double timeReward = -estimatedResponseTime / 100.0;
        
        // Load balancing component (negative variance)
        double[] loads = new double[vmList.size()];
        for (int i = 0; i < vmList.size(); i++) {
            loads[i] = vmLoads.getOrDefault(vmList.get(i).getId(), 0.0);
        }
        double loadVariance = calculateVariance(loads);
        double balanceReward = -loadVariance / 1000.0;
        
        // Combined reward
        return 0.7 * timeReward + 0.3 * balanceReward;
    }
    
    private double calculateVariance(double[] values) {
        double mean = Arrays.stream(values).average().orElse(0.0);
        return Arrays.stream(values).map(v -> Math.pow(v - mean, 2)).average().orElse(0.0);
    }
    
    private JsonObject getStepInfo() {
        JsonObject info = new JsonObject();
        info.addProperty("current_cloudlet", currentCloudletIndex);
        info.addProperty("completed", completedCloudlets);
        return info;
    }
    
    private JsonObject getEpisodeInfo() {
        JsonObject info = new JsonObject();
        info.addProperty("total_cloudlets", cloudletQueue.size());
        info.addProperty("completed_cloudlets", completedCloudlets);
        info.addProperty("avg_response_time", completedCloudlets > 0 ? totalResponseTime / completedCloudlets : 0);
        info.addProperty("sla_violations", slaViolations);
        info.addProperty("sla_violation_rate", (double) slaViolations / cloudletQueue.size());
        return info;
    }
    
    private List<Host> createHosts() {
        List<Host> hostList = new ArrayList<>();
        
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < 4; j++) {
                peList.add(new PeSimple(2000)); // 2000 MIPS per PE
            }
            
            Host host = new HostSimple(16384, 100000, 100000, peList);
            hostList.add(host);
        }
        
        return hostList;
    }
    
    private List<Vm> createVms() {
        List<Vm> list = new ArrayList<>();
        
        for (int i = 0; i < NUM_VMS; i++) {
            Vm vm = new VmSimple(1000 + i * 50, 1); // Varying MIPS
            vm.setRam(2048).setBw(10000).setSize(10000);
            list.add(vm);
        }
        
        return list;
    }
    
    private List<Cloudlet> generateCloudlets(int count) {
        List<Cloudlet> list = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < count; i++) {
            long length = 10000 + random.nextInt(90000); // 10k to 100k MI
            Cloudlet cloudlet = new CloudletSimple(length, 1);
            cloudlet.setUtilizationModelCpu(new UtilizationModelDynamic(0.5))
                     .setUtilizationModelRam(new UtilizationModelDynamic(0.3))
                     .setUtilizationModelBw(new UtilizationModelDynamic(0.2));
            list.add(cloudlet);
        }
        
        return list;
    }
}
