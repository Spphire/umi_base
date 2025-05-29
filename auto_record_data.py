import os
import re
import subprocess
import sys
import signal
import time

EP_START = 1
EP_END = 50

def update_makefile(trial_num):
    """Update the SAVE_FILE_NAME in the Makefile to the specified trial number."""
    makefile_path = os.path.join(os.path.dirname(__file__), 'Makefile')
    
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    # Replace the SAVE_FILE_NAME line with the new trial number
    new_content = re.sub(
        r'SAVE_FILE_NAME := trial\d+\.pkl',
        f'SAVE_FILE_NAME := trial{trial_num}.pkl',
        content
    )
    
    with open(makefile_path, 'w') as f:
        f.write(new_content)
    
    print(f"Updated Makefile: SAVE_FILE_NAME := trial{trial_num}.pkl")

def main():
    for trial_num in range(EP_START, EP_END + 1):
        # Update the Makefile with the current trial number
        update_makefile(trial_num)
        
        # Display current trial
        print(f"\n===== Starting Trial {trial_num}/{EP_END} =====")
        
        # Run the make command
        process = None
        try:
            # Start the make command as a subprocess
            process = subprocess.Popen(["make", "teleop.start_record"])
            
            # Wait for the process to complete or user to interrupt
            process.wait()
            
        except KeyboardInterrupt:
            print("\nRecording interrupted by user (Ctrl+C)")
            # If the process is still running, terminate it gracefully
            if process and process.poll() is None:
                process.send_signal(signal.SIGINT)
                time.sleep(1)
                # If it's still running after SIGINT, force kill it
                if process.poll() is None:
                    process.terminate()
            
            print(f"Trial {trial_num} recording stopped")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running make command: {e}")
            choice = input("Continue to next trial? (y/n): ")
            if choice.lower() != 'y':
                sys.exit(1)
        
        # If this is the last trial, break without asking for confirmation
        if trial_num == EP_END:
            print("All trials completed!")
            break
        
        # Ask for confirmation before continuing to the next trial
        input("\nPress Enter to continue to the next trial...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript terminated by user")
        sys.exit(1)
