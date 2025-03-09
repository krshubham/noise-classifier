#!/usr/bin/env python3
import os
import sys

def rename_wav_files(directory):
    """
    Recursively traverse through the specified directory and rename all WAV files
    by replacing spaces with underscores.
    
    Args:
        directory (str): Path to the directory containing WAV files
    """
    count = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is a WAV file and has spaces in its name
            if filename.lower().endswith('.wav') and ' ' in filename:
                old_path = os.path.join(root, filename)
                # Replace spaces with underscores
                new_filename = filename.replace(' ', '_')
                new_path = os.path.join(root, new_filename)
                
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                count += 1
    
    print(f"\nRenamed {count} WAV files in total.")

if __name__ == "__main__":
    # Use the provided directory or default to the data directory
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        sys.exit(1)
    
    print(f"Scanning for WAV files with spaces in {directory}...")
    rename_wav_files(directory)
