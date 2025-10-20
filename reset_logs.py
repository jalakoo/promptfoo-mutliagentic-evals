#!/usr/bin/env python3
"""
Log reset utility for promptfoo multiagentic evals
"""

import os
import glob
import sys

def reset_logs():
    """Reset all log files in the logs directory"""
    log_dir = "./logs"
    
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist")
        return
    
    # Find all log files
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print("No log files found to reset")
        return
    
    print("Resetting log files:")
    for log_file in log_files:
        try:
            # Clear the file by opening in write mode
            with open(log_file, 'w') as f:
                f.write("")
            print(f"  ✅ Reset: {log_file}")
        except Exception as e:
            print(f"  ❌ Failed to reset {log_file}: {e}")
    
    print(f"\nReset {len(log_files)} log files")

if __name__ == "__main__":
    reset_logs()
