#!/usr/bin/env python3
import os
import argparse
from signal_processing import SignalProcessor
import shutil

def main():
    parser = argparse.ArgumentParser(description='Enhance valve lash audio files with specialized signal processing')
    parser.add_argument('--input-dir', type=str, default='data/valve_lash', help='Directory containing valve lash audio files')
    parser.add_argument('--output-dir', type=str, default='data/enhanced_valve_lash', help='Directory to save enhanced files')
    parser.add_argument('--replace', action='store_true', help='Replace original files with enhanced ones')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate for processing')
    args = parser.parse_args()
    
    # Create processor
    processor = SignalProcessor(sr=args.sr)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file
    print(f"Enhancing valve lash audio files from {args.input_dir}...")
    processor.process_valve_lash_dataset(args.input_dir, args.output_dir)
    
    # Replace original files if requested
    if args.replace:
        print("Replacing original files with enhanced versions...")
        for filename in os.listdir(args.output_dir):
            if filename.startswith("enhanced_") and filename.endswith('.wav'):
                # Get original filename
                original_name = filename.replace("enhanced_", "")
                original_path = os.path.join(args.input_dir, original_name)
                enhanced_path = os.path.join(args.output_dir, filename)
                
                # Make backup of original file
                backup_dir = os.path.join(args.input_dir, 'backup')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, original_name)
                shutil.copy2(original_path, backup_path)
                
                # Replace original with enhanced
                shutil.copy2(enhanced_path, original_path)
                print(f"Replaced {original_name} (backup saved to {backup_dir})")
    
    print("Enhancement complete!")
    
    # Provide next steps
    print("\nNext steps:")
    print("1. Run 'python augment_and_train.py --target-count 20' to train models with enhanced data")
    print("2. Run 'python app.py' to test the improved models")

if __name__ == "__main__":
    main()
