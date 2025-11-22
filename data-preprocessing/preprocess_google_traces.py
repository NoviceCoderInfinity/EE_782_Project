"""
Google Cluster Trace Preprocessor
Phase 2: Data Preprocessing for CloudSim

This script:
1. Reads Google Cluster trace data (compressed CSV)
2. Filters data (1-hour window or sample 1000 tasks)
3. Normalizes values (0-1 scale to CloudSim MIPS)
4. Exports to CloudSim-compatible CSV format

Dataset Schema (task_events table):
- time: timestamp (microseconds)
- missing_type: type of missing information
- job_id: unique job identifier
- task_index: task index within the job
- machine_id: machine where task ran
- event_type: 0=SUBMIT, 1=SCHEDULE, 2=EVICT, 3=FAIL, 4=FINISH, etc.
- user: anonymized username
- scheduling_class: priority (0=free, 1=best-effort, 2=mid, 3=high)
- priority: numeric priority
- resource_request: CPU, RAM, disk (normalized 0-1)
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import sys

# CloudSim Configuration
CLOUDSIM_CONFIG = {
    'HOST_MIPS': 10000,      # MIPS per host PE
    'HOST_RAM': 16384,       # MB
    'HOST_BW': 10000,        # Mbps
    'VM_MIPS': 1000,         # MIPS per VM PE
    'TIME_SCALE': 1000,      # Microseconds to seconds conversion
}

def read_trace_file(file_path, sample_size=None):
    """
    Read a Google Cluster trace file
    
    Args:
        file_path: Path to .csv.gz file
        sample_size: Number of rows to sample (None for all)
    
    Returns:
        DataFrame with trace data
    """
    print(f"\n⏳ Reading trace file: {file_path}")
    
    # Column names from Google Cluster Data schema
    columns = [
        'time', 'missing_type', 'job_id', 'task_index', 'machine_id',
        'event_type', 'user', 'scheduling_class', 'priority',
        'resource_request_cpus', 'resource_request_memory',
        'constraint'
    ]
    
    try:
        if str(file_path).endswith('.gz'):
            with gzip.open(file_path, 'rt') as f:
                if sample_size:
                    # Read in chunks and sample
                    chunks = []
                    for chunk in pd.read_csv(f, names=columns, nrows=sample_size*10):
                        chunks.append(chunk)
                    df = pd.concat(chunks).sample(n=min(sample_size, len(pd.concat(chunks))))
                else:
                    df = pd.read_csv(f, names=columns)
        else:
            if sample_size:
                df = pd.read_csv(file_path, names=columns, nrows=sample_size*10).sample(n=sample_size)
            else:
                df = pd.read_csv(file_path, names=columns)
        
        print(f"✓ Loaded {len(df):,} rows")
        return df
    
    except Exception as e:
        print(f"✗ Error reading file: {e}")
        return None

def filter_submit_events(df):
    """
    Filter for task SUBMIT events only (event_type == 0)
    These represent new tasks arriving in the system
    """
    print(f"\n⏳ Filtering for SUBMIT events...")
    submit_df = df[df['event_type'] == 0].copy()
    print(f"✓ Found {len(submit_df):,} SUBMIT events ({len(submit_df)/len(df)*100:.1f}%)")
    return submit_df

def filter_time_window(df, window_hours=1):
    """
    Filter data to a specific time window
    
    Args:
        df: DataFrame with 'time' column (microseconds)
        window_hours: Number of hours from start
    """
    print(f"\n⏳ Filtering to {window_hours}-hour window...")
    
    min_time = df['time'].min()
    window_microseconds = window_hours * 3600 * 1000000
    max_time = min_time + window_microseconds
    
    filtered_df = df[(df['time'] >= min_time) & (df['time'] <= max_time)].copy()
    
    print(f"✓ Time range: {min_time} to {max_time}")
    print(f"✓ Kept {len(filtered_df):,} events ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return filtered_df

def normalize_to_cloudsim(df):
    """
    Normalize Google trace data to CloudSim format
    
    Conversions:
    - CPU request (0-1) -> Cloudlet length in MI
    - Memory request (0-1) -> RAM requirement in MB
    - Time (microseconds) -> Submission delay in seconds
    """
    print(f"\n⏳ Normalizing to CloudSim format...")
    
    cloudsim_df = pd.DataFrame()
    
    # Cloudlet ID: unique identifier
    cloudsim_df['cloudlet_id'] = range(len(df))
    
    # Job ID and Task Index (for reference)
    cloudsim_df['job_id'] = df['job_id'].values
    cloudsim_df['task_index'] = df['task_index'].values
    
    # Submission time: convert from microseconds to seconds, relative to start
    min_time = df['time'].min()
    cloudsim_df['submission_delay'] = (df['time'].values - min_time) / 1000000.0
    
    # CPU request -> Cloudlet length in MI (Million Instructions)
    # Formula: cpu_request * VM_MIPS * estimated_duration
    # We'll use a base duration and scale by CPU request
    base_duration = 10.0  # seconds
    cpu_request = df['resource_request_cpus'].fillna(0.1).clip(0.01, 1.0)
    cloudsim_df['length'] = (cpu_request * CLOUDSIM_CONFIG['VM_MIPS'] * base_duration).astype(int)
    
    # Ensure minimum length
    cloudsim_df['length'] = cloudsim_df['length'].clip(lower=1000)
    
    # Number of PEs (cores) required
    cloudsim_df['pes'] = np.ceil(cpu_request * 4).astype(int).clip(1, 4)
    
    # RAM requirement: memory_request (0-1) -> MB
    memory_request = df['resource_request_memory'].fillna(0.1).clip(0.01, 1.0)
    cloudsim_df['ram'] = (memory_request * CLOUDSIM_CONFIG['HOST_RAM'] / 4).astype(int)
    cloudsim_df['ram'] = cloudsim_df['ram'].clip(lower=128)
    
    # Priority: scheduling_class (0-3)
    cloudsim_df['priority'] = df['scheduling_class'].fillna(1).astype(int)
    
    # File size (for I/O simulation) - estimated based on RAM
    cloudsim_df['file_size'] = (cloudsim_df['ram'] * 0.5).astype(int)
    
    # Output size - typically smaller
    cloudsim_df['output_size'] = (cloudsim_df['file_size'] * 0.3).astype(int)
    
    print(f"✓ Normalized {len(cloudsim_df):,} cloudlets")
    print(f"\nCloudlet Statistics:")
    print(f"  Length (MI): min={cloudsim_df['length'].min():,}, "
          f"max={cloudsim_df['length'].max():,}, "
          f"mean={cloudsim_df['length'].mean():.0f}")
    print(f"  PEs: min={cloudsim_df['pes'].min()}, "
          f"max={cloudsim_df['pes'].max()}, "
          f"mean={cloudsim_df['pes'].mean():.1f}")
    print(f"  RAM (MB): min={cloudsim_df['ram'].min()}, "
          f"max={cloudsim_df['ram'].max()}, "
          f"mean={cloudsim_df['ram'].mean():.0f}")
    
    return cloudsim_df

def export_to_csv(df, output_file='cloudsim_workload.csv'):
    """Export processed data to CSV for CloudSim"""
    print(f"\n⏳ Exporting to {output_file}...")
    
    # Select columns for CloudSim
    export_df = df[[
        'cloudlet_id', 'length', 'pes', 'ram', 
        'file_size', 'output_size', 'priority', 'submission_delay'
    ]]
    
    # Sort by submission delay
    export_df = export_df.sort_values('submission_delay')
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(export_df):,} cloudlets to {output_file}")
    print(f"  File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
    
    return output_file

def create_synthetic_workload(num_tasks=1000, output_file='synthetic_workload.csv'):
    """
    Create synthetic workload (alternative if Google traces unavailable)
    """
    print(f"\n⏳ Creating synthetic workload with {num_tasks} tasks...")
    
    np.random.seed(42)
    
    df = pd.DataFrame()
    df['cloudlet_id'] = range(num_tasks)
    df['job_id'] = np.random.randint(0, num_tasks // 10, num_tasks)
    df['task_index'] = np.random.randint(0, 100, num_tasks)
    
    # Submission times: Poisson arrival process
    inter_arrival = np.random.exponential(scale=5.0, size=num_tasks)
    df['submission_delay'] = np.cumsum(inter_arrival)
    
    # Task characteristics with realistic distribution
    df['length'] = np.random.lognormal(mean=9.0, sigma=1.5, size=num_tasks).astype(int)
    df['length'] = df['length'].clip(1000, 100000)
    
    df['pes'] = np.random.choice([1, 2, 4], size=num_tasks, p=[0.6, 0.3, 0.1])
    df['ram'] = np.random.randint(128, 4096, num_tasks)
    df['file_size'] = (df['ram'] * 0.5).astype(int)
    df['output_size'] = (df['file_size'] * 0.3).astype(int)
    df['priority'] = np.random.choice([0, 1, 2, 3], size=num_tasks, p=[0.1, 0.5, 0.3, 0.1])
    
    export_df = df[[
        'cloudlet_id', 'length', 'pes', 'ram',
        'file_size', 'output_size', 'priority', 'submission_delay'
    ]]
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Created synthetic workload: {output_file}")
    
    return output_file

def main():
    print("\n" + "="*60)
    print("GOOGLE CLUSTER TRACE PREPROCESSOR")
    print("Phase 2: Data Preprocessing")
    print("="*60 + "\n")
    
    # Check for trace files
    trace_dir = Path('./google_traces')
    
    if not trace_dir.exists():
        print("⚠ No google_traces directory found.")
        print("\nOptions:")
        print("  1. Run: python download_google_traces.py")
        print("  2. Create synthetic workload instead")
        
        choice = input("\nCreate synthetic workload? [y/N]: ").lower()
        if choice == 'y':
            num_tasks = int(input("Number of tasks [100-10000, default=1000]: ") or "1000")
            num_tasks = max(100, min(10000, num_tasks))
            
            output_file = create_synthetic_workload(num_tasks)
            print(f"\n✓ Synthetic workload created: {output_file}")
            print("\nNext steps:")
            print("  1. Copy this file to: cloudsim-rl-project/src/main/resources/")
            print("  2. Update Java code to use GoogleTraceReader")
            return
        else:
            print("\n✗ Preprocessing cancelled.")
            sys.exit(1)
    
    # Find trace files
    trace_files = list(trace_dir.glob('task_events-*.csv.gz'))
    
    if not trace_files:
        print(f"✗ No trace files found in {trace_dir}")
        print("   Run: python download_google_traces.py")
        sys.exit(1)
    
    print(f"Found {len(trace_files)} trace file(s)")
    
    # Choose processing mode
    print("\n" + "="*60)
    print("PROCESSING OPTIONS")
    print("="*60)
    print("\n1. Time window: Process 1-hour window")
    print("2. Sample: Randomly sample 1000 tasks")
    print("3. Full: Process entire file (may take time)")
    
    choice = input("\nSelect option [1/2/3, default=2]: ") or "2"
    
    # Read first file
    trace_file = trace_files[0]
    
    if choice == "1":
        # Time window approach
        df = read_trace_file(trace_file)
        if df is None:
            sys.exit(1)
        
        df = filter_submit_events(df)
        df = filter_time_window(df, window_hours=1)
        
    elif choice == "2":
        # Sample approach (faster)
        df = read_trace_file(trace_file, sample_size=10000)
        if df is None:
            sys.exit(1)
        
        df = filter_submit_events(df)
        df = df.head(1000)  # Take first 1000
        
    else:
        # Full processing
        df = read_trace_file(trace_file)
        if df is None:
            sys.exit(1)
        
        df = filter_submit_events(df)
    
    # Normalize and export
    cloudsim_df = normalize_to_cloudsim(df)
    output_file = export_to_csv(cloudsim_df, 'cloudsim_workload.csv')
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\n✓ Output file: {output_file}")
    print(f"✓ Number of cloudlets: {len(cloudsim_df):,}")
    print(f"✓ Simulation duration: {cloudsim_df['submission_delay'].max():.1f} seconds")
    
    print("\nNext steps:")
    print("  1. Copy workload file to: cloudsim-rl-project/src/main/resources/")
    print("  2. Implement GoogleTraceReader.java")
    print("  3. Update simulation to use trace data")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
