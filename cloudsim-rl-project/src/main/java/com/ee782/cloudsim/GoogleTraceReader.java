package com.ee782.cloudsim;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.utilizationmodels.UtilizationModelFull;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * GoogleTraceReader - Phase 2 Data Integration
 * 
 * Reads preprocessed Google Cluster trace data and converts it to CloudSim Cloudlets.
 * 
 * CSV Format:
 * cloudlet_id,length,pes,ram,file_size,output_size,priority,submission_delay
 * 
 * Features:
 * - Reads from resources folder or file path
 * - Creates CloudSim Cloudlets with realistic workload characteristics
 * - Supports dynamic submission delays for realistic arrival patterns
 * - Configurable utilization models
 */
public class GoogleTraceReader {
    
    private String csvFilePath;
    private List<Cloudlet> cloudlets;
    private boolean useResourceFile;
    
    /**
     * Constructor to read from resources folder
     * @param resourceFileName Name of CSV file in src/main/resources/
     */
    public GoogleTraceReader(String resourceFileName) {
        this.csvFilePath = resourceFileName;
        this.useResourceFile = true;
        this.cloudlets = new ArrayList<>();
    }
    
    /**
     * Constructor to read from absolute file path
     * @param filePath Absolute path to CSV file
     * @param useResourceFile Set to false for absolute path
     */
    public GoogleTraceReader(String filePath, boolean useResourceFile) {
        this.csvFilePath = filePath;
        this.useResourceFile = useResourceFile;
        this.cloudlets = new ArrayList<>();
    }
    
    /**
     * Load cloudlets from CSV file
     * @return List of Cloudlet objects
     * @throws IOException If file cannot be read
     */
    public List<Cloudlet> loadCloudlets() throws IOException {
        BufferedReader reader;
        
        if (useResourceFile) {
            // Read from resources folder
            InputStream is = getClass().getClassLoader().getResourceAsStream(csvFilePath);
            if (is == null) {
                throw new IOException("Resource file not found: " + csvFilePath);
            }
            reader = new BufferedReader(new InputStreamReader(is));
        } else {
            // Read from file system
            reader = new BufferedReader(new FileReader(csvFilePath));
        }
        
        String line;
        boolean headerSkipped = false;
        int lineNumber = 0;
        
        try {
            while ((line = reader.readLine()) != null) {
                lineNumber++;
                
                // Skip header
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }
                
                // Skip empty lines
                if (line.trim().isEmpty()) {
                    continue;
                }
                
                try {
                    Cloudlet cloudlet = parseCloudletFromLine(line);
                    if (cloudlet != null) {
                        cloudlets.add(cloudlet);
                    }
                } catch (Exception e) {
                    System.err.printf("Warning: Error parsing line %d: %s%n", lineNumber, e.getMessage());
                    // Continue processing other lines
                }
            }
        } finally {
            reader.close();
        }
        
        System.out.printf("✓ Loaded %d cloudlets from %s%n", cloudlets.size(), csvFilePath);
        printStatistics();
        
        return cloudlets;
    }
    
    /**
     * Parse a single CSV line into a Cloudlet object
     * 
     * CSV Format: cloudlet_id,length,pes,ram,file_size,output_size,priority,submission_delay
     */
    private Cloudlet parseCloudletFromLine(String line) {
        String[] fields = line.split(",");
        
        if (fields.length < 8) {
            throw new IllegalArgumentException("Invalid CSV format: expected 8 fields, got " + fields.length);
        }
        
        try {
            // Parse fields
            long cloudletId = Long.parseLong(fields[0].trim());
            long length = Long.parseLong(fields[1].trim());
            int pes = Integer.parseInt(fields[2].trim());
            long ram = Long.parseLong(fields[3].trim());
            long fileSize = Long.parseLong(fields[4].trim());
            long outputSize = Long.parseLong(fields[5].trim());
            int priority = Integer.parseInt(fields[6].trim());
            double submissionDelay = Double.parseDouble(fields[7].trim());
            
            // Create Cloudlet with Google trace characteristics
            CloudletSimple cloudlet = new CloudletSimple(cloudletId, length, pes);
            
            // Set resource requirements
            cloudlet.setFileSize(fileSize);
            cloudlet.setOutputSize(outputSize);
            
            // Set utilization models
            // CPU: Full utilization (task runs at maximum capacity)
            cloudlet.setUtilizationModelCpu(new UtilizationModelFull());
            
            // RAM: Dynamic based on cloudlet RAM requirement
            // CloudSim expects 0-1 scale, normalize by VM RAM capacity
            double ramUtilization = Math.min(1.0, ram / 2048.0); // Assuming 2GB VM RAM
            cloudlet.setUtilizationModelRam(new UtilizationModelDynamic(ramUtilization));
            
            // Bandwidth: Dynamic (moderate usage)
            cloudlet.setUtilizationModelBw(new UtilizationModelDynamic(0.5));
            
            // Set priority (optional - CloudSim Plus supports this)
            cloudlet.setPriority(priority);
            
            // Store submission delay as cloudlet attribute (for dynamic submission)
            cloudlet.setSubmissionDelay(submissionDelay);
            
            return cloudlet;
            
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number format in line: " + line, e);
        }
    }
    
    /**
     * Get loaded cloudlets
     */
    public List<Cloudlet> getCloudlets() {
        return cloudlets;
    }
    
    /**
     * Get number of loaded cloudlets
     */
    public int getCloudletCount() {
        return cloudlets.size();
    }
    
    /**
     * Print statistics about loaded cloudlets
     */
    private void printStatistics() {
        if (cloudlets.isEmpty()) {
            System.out.println("No cloudlets loaded.");
            return;
        }
        
        // Calculate statistics
        long minLength = Long.MAX_VALUE;
        long maxLength = Long.MIN_VALUE;
        long totalLength = 0;
        
        int minPes = Integer.MAX_VALUE;
        int maxPes = Integer.MIN_VALUE;
        
        double minDelay = Double.MAX_VALUE;
        double maxDelay = Double.MIN_VALUE;
        
        for (Cloudlet c : cloudlets) {
            long length = c.getLength();
            int pes = (int) c.getPesNumber();
            double delay = c.getSubmissionDelay();
            
            minLength = Math.min(minLength, length);
            maxLength = Math.max(maxLength, length);
            totalLength += length;
            
            minPes = Math.min(minPes, pes);
            maxPes = Math.max(maxPes, pes);
            
            minDelay = Math.min(minDelay, delay);
            maxDelay = Math.max(maxDelay, delay);
        }
        
        double avgLength = (double) totalLength / cloudlets.size();
        
        System.out.println("\nCloudlet Statistics:");
        System.out.printf("  Count: %d%n", cloudlets.size());
        System.out.printf("  Length (MI): min=%,d, max=%,d, avg=%,.0f%n", minLength, maxLength, avgLength);
        System.out.printf("  PEs: min=%d, max=%d%n", minPes, maxPes);
        System.out.printf("  Submission delay (s): min=%.1f, max=%.1f%n", minDelay, maxDelay);
        System.out.printf("  Simulation duration: %.1f seconds%n", maxDelay);
    }
    
    /**
     * Example usage
     */
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(60));
            System.out.println("GOOGLE TRACE READER TEST");
            System.out.println("=".repeat(60));
            
            // Test reading from resources
            GoogleTraceReader reader = new GoogleTraceReader("synthetic_workload.csv");
            List<Cloudlet> cloudlets = reader.loadCloudlets();
            
            System.out.printf("%n✓ Successfully loaded %d cloudlets%n", cloudlets.size());
            
            // Show first 5 cloudlets
            System.out.println("\nFirst 5 Cloudlets:");
            for (int i = 0; i < Math.min(5, cloudlets.size()); i++) {
                Cloudlet c = cloudlets.get(i);
                System.out.printf("  [%d] Length=%,d MI, PEs=%d, Delay=%.1fs, Priority=%d%n",
                    c.getId(), c.getLength(), c.getPesNumber(), 
                    c.getSubmissionDelay(), c.getPriority());
            }
            
            System.out.println("\n" + "=".repeat(60));
            
        } catch (IOException e) {
            System.err.println("✗ Error reading trace file: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
