"""
Google Cluster Trace Data Downloader
Phase 2: Download Google Cluster-Usage Traces v3

This script downloads task events from the Google Cluster trace dataset.
The dataset is hosted on Google Cloud Storage.

Dataset: https://github.com/google/cluster-data
Official location: gs://clusterdata-2019-a/task_events/
"""

import os
import subprocess
import sys
from pathlib import Path

def check_gsutil():
    """Check if gsutil is installed"""
    try:
        result = subprocess.run(['gsutil', 'version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        print("✓ gsutil is installed")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ gsutil is not installed")
        return False

def install_gsutil_instructions():
    """Print instructions for installing gsutil"""
    print("\n" + "="*60)
    print("GSUTIL INSTALLATION REQUIRED")
    print("="*60)
    print("\nTo download Google Cluster traces, you need Google Cloud SDK.")
    print("\nInstallation options:")
    print("\n1. Using snap (Ubuntu/Linux):")
    print("   sudo snap install google-cloud-cli --classic")
    print("\n2. Using apt (Ubuntu/Debian):")
    print("   sudo apt-get install google-cloud-cli")
    print("\n3. Manual installation:")
    print("   curl https://sdk.cloud.google.com | bash")
    print("   exec -l $SHELL")
    print("\nAfter installation, run this script again.")
    print("="*60 + "\n")

def download_sample_traces(output_dir='./google_traces', num_files=1):
    """
    Download a sample of Google Cluster trace data
    
    Args:
        output_dir: Directory to save the downloaded files
        num_files: Number of trace files to download (1-500)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING GOOGLE CLUSTER TRACES")
    print(f"{'='*60}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"Number of files to download: {num_files}")
    print(f"{'='*60}\n")
    
    # Google Cluster Data 2019 location
    # Task events are sharded into 500 files
    base_url = "gs://clusterdata-2019-a/task_events/"
    
    downloaded = 0
    failed = 0
    
    for i in range(num_files):
        # Files are named: task_events-000000000000.csv.gz to task_events-000000000499.csv.gz
        file_name = f"task_events-{i:012d}.csv.gz"
        source = base_url + file_name
        destination = output_path / file_name
        
        if destination.exists():
            print(f"✓ File {i+1}/{num_files} already exists: {file_name}")
            downloaded += 1
            continue
        
        print(f"⬇ Downloading {i+1}/{num_files}: {file_name}...", end=' ')
        
        try:
            result = subprocess.run(
                ['gsutil', '-m', 'cp', source, str(destination)],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print("✓ Success")
                downloaded += 1
            else:
                print(f"✗ Failed: {result.stderr}")
                failed += 1
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
            failed += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successfully downloaded: {downloaded}/{num_files}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{num_files}")
    print(f"Total size: {get_directory_size(output_path):.2f} MB")
    print(f"{'='*60}\n")
    
    return downloaded > 0

def get_directory_size(path):
    """Calculate total size of directory in MB"""
    total = 0
    for entry in Path(path).rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024 * 1024)  # Convert to MB

def download_schema():
    """Download the schema documentation"""
    schema_url = "https://raw.githubusercontent.com/google/cluster-data/master/ClusterData2019.md"
    output_file = "./google_traces/schema.md"
    
    print("\n⬇ Downloading schema documentation...", end=' ')
    try:
        import urllib.request
        urllib.request.urlretrieve(schema_url, output_file)
        print("✓ Success")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("GOOGLE CLUSTER TRACE DATA DOWNLOADER")
    print("Phase 2: Data Preprocessing")
    print("="*60 + "\n")
    
    # Check if gsutil is available
    if not check_gsutil():
        install_gsutil_instructions()
        print("\n💡 Alternative: You can manually download from:")
        print("   https://console.cloud.google.com/storage/browser/clusterdata-2019-a")
        print("\n   Or use the preprocessor with sample/synthetic data instead.")
        sys.exit(1)
    
    # Download schema
    download_schema()
    
    # Ask user how many files to download
    print("\n" + "="*60)
    print("DOWNLOAD OPTIONS")
    print("="*60)
    print("\nGoogle Cluster trace data is split into 500 files.")
    print("Each file contains ~2-3 million task events (~500MB compressed).")
    print("\nRecommendations:")
    print("  - For testing: 1 file (~500MB)")
    print("  - For small experiments: 2-5 files (~2.5GB)")
    print("  - For full dataset: 500 files (~250GB)")
    
    try:
        num_files = int(input("\nHow many files to download? [1-500, default=1]: ") or "1")
        num_files = max(1, min(500, num_files))
    except ValueError:
        num_files = 1
    
    # Confirm download
    estimated_size = num_files * 500  # Approximate MB
    print(f"\n⚠ This will download approximately {estimated_size}MB of data.")
    confirm = input("Continue? [y/N]: ").lower()
    
    if confirm != 'y':
        print("\n✗ Download cancelled.")
        sys.exit(0)
    
    # Download traces
    success = download_sample_traces(num_files=num_files)
    
    if success:
        print("\n✓ Download complete!")
        print("\nNext steps:")
        print("  1. Run: python preprocess_google_traces.py")
        print("  2. This will filter and normalize the data for CloudSim")
    else:
        print("\n✗ Download failed. Check your internet connection and gsutil setup.")

if __name__ == "__main__":
    main()
