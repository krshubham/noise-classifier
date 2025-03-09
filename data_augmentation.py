#!/usr/bin/env python3
import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import shutil
from signal_processing import SignalProcessor

class AudioAugmenter:
    def __init__(self, sr=22050):
        self.sr = sr
        self.signal_processor = SignalProcessor(sr=sr)
    
    def time_stretch(self, y, rate=1.2):
        """Time stretch the audio signal"""
        return librosa.effects.time_stretch(y, rate=rate)
    
    def pitch_shift(self, y, steps=2):
        """Pitch shift the audio signal"""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=steps)
    
    def add_noise(self, y, noise_factor=0.005):
        """Add random noise to the audio signal"""
        noise = np.random.randn(len(y))
        return y + noise_factor * noise
    
    def shift(self, y, shift_max=1000):
        """Shift the audio signal"""
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(y, shift)
    
    def change_volume(self, y, volume_factor=0.8):
        """Change the volume of the audio signal"""
        return y * volume_factor
    
    def apply_filter(self, y, filter_type='highpass', cutoff_freq=1000):
        """Apply a filter to the audio signal"""
        nyquist = 0.5 * self.sr
        cutoff = cutoff_freq / nyquist
        
        if filter_type == 'highpass':
            b, a = signal.butter(4, cutoff, btype='highpass')
        elif filter_type == 'lowpass':
            b, a = signal.butter(4, cutoff, btype='lowpass')
        else:
            return y
        
        return signal.filtfilt(b, a, y)
    
    def augment_audio(self, y, augmentation_type):
        """Apply a specific augmentation to the audio signal"""
        if augmentation_type == 'time_stretch':
            y_aug = self.time_stretch(y)
        elif augmentation_type == 'pitch_shift':
            y_aug = self.pitch_shift(y)
        elif augmentation_type == 'noise':
            y_aug = self.add_noise(y)
        elif augmentation_type == 'shift':
            y_aug = self.shift(y)
        elif augmentation_type == 'volume':
            y_aug = self.change_volume(y)
        elif augmentation_type == 'highpass':
            y_aug = self.apply_filter(y, 'highpass')
        elif augmentation_type == 'lowpass':
            y_aug = self.apply_filter(y, 'lowpass')
        elif augmentation_type == 'valve_lash_enhance':
            # Apply specialized valve lash processing
            y_aug = self.signal_processor.bandpass_filter(y, low_freq=800, high_freq=5000)
            y_aug = self.signal_processor.enhance_transients(y_aug, threshold=0.05, boost_factor=2.5)
            y_aug = self.signal_processor.harmonic_percussive_separation(y_aug, margin=4.0)
        else:
            y_aug = y
        
        return y_aug
    
    def augment_file(self, input_file, output_dir, augmentation_types=None, prefix='aug'):
        """Augment a single audio file with multiple augmentation types"""
        if augmentation_types is None:
            augmentation_types = ['time_stretch', 'pitch_shift', 'noise', 'shift', 'volume', 'highpass', 'lowpass']
        
        # Add valve_lash_enhance for valve lash files
        if 'valve_lash' in input_file and 'valve_lash_enhance' not in augmentation_types:
            augmentation_types.append('valve_lash_enhance')
        
        # Load audio file
        y, sr = librosa.load(input_file, sr=self.sr)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.basename(input_file)
        
        # Apply each augmentation and save the result
        for aug_type in augmentation_types:
            y_aug = self.augment_audio(y, aug_type)
            
            # Create output filename
            output_file = os.path.join(output_dir, f"{prefix}_{aug_type}_{base_name}")
            
            # Save augmented audio
            sf.write(output_file, y_aug, sr)
    
    def augment_directory(self, input_dir, output_dir, target_count=None, augmentation_types=None):
        """Augment all audio files in a directory"""
        if augmentation_types is None:
            augmentation_types = ['time_stretch', 'pitch_shift', 'noise', 'shift', 'volume', 'highpass', 'lowpass']
            
            # Add valve_lash_enhance for valve lash directory
            if 'valve_lash' in input_dir and 'valve_lash_enhance' not in augmentation_types:
                augmentation_types.append('valve_lash_enhance')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all audio files in the input directory
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_files.append(os.path.join(root, file))
        
        # Determine how many augmentations to create per file
        num_files = len(audio_files)
        
        if target_count is not None and num_files > 0:
            # Calculate how many augmentations we need per file to reach target_count
            augs_per_file = max(1, int(np.ceil((target_count - num_files) / num_files)))
            
            # Limit to the number of augmentation types available
            augs_per_file = min(augs_per_file, len(augmentation_types))
            
            print(f"Creating {augs_per_file} augmentations per file to reach target of {target_count}")
            
            # Use only the needed augmentation types
            augmentation_types = augmentation_types[:augs_per_file]
        
        # Process each audio file
        for audio_file in audio_files:
            # Get the relative path from input_dir
            rel_path = os.path.relpath(audio_file, input_dir)
            parent_dir = os.path.dirname(rel_path)
            
            # Create corresponding output directory
            file_output_dir = os.path.join(output_dir, parent_dir)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Augment the file
            self.augment_file(audio_file, file_output_dir, augmentation_types)
            
            # Also copy the original file to the output directory
            output_file = os.path.join(file_output_dir, os.path.basename(audio_file))
            shutil.copy2(audio_file, output_file)

def augment_dataset(input_dir, output_dir, target_count=None, sr=22050):
    """
    Augment all audio files in the input directory and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing audio files to augment
        output_dir (str): Directory to save augmented files
        target_count (int, optional): Target number of samples per class
        sr (int): Sample rate
    """
    augmenter = AudioAugmenter(sr=sr)
    
    # Get all subdirectories (classes)
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for class_name in classes:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        
        # Determine augmentation types based on class
        augmentation_types = ['time_stretch', 'pitch_shift', 'noise', 'shift', 'volume', 'highpass', 'lowpass']
        
        # Add specialized augmentations for specific classes
        if class_name == 'valve_lash':
            augmentation_types.append('valve_lash_enhance')
        
        # Apply more aggressive augmentation for fan_belt_issue class
        if class_name == 'fan_belt_issue':
            # Add more variations of the same augmentation types with different parameters
            augmentation_types = ['time_stretch', 'pitch_shift', 'noise', 'shift', 'volume', 
                                 'highpass', 'lowpass', 'time_stretch', 'pitch_shift', 'noise']
        
        print(f"Augmenting {class_name} with {augmentation_types}")
        augmenter.augment_directory(class_input_dir, class_output_dir, target_count, augmentation_types)
    
    print(f"Augmentation complete. Augmented files saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment audio files for engine sound classification')
    parser.add_argument('--input-dir', type=str, default='data', help='Directory containing audio files to augment')
    parser.add_argument('--output-dir', type=str, default='augmented_data', help='Directory to save augmented files')
    parser.add_argument('--target-count', type=int, default=None, help='Target number of samples per class')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    
    args = parser.parse_args()
    
    augment_dataset(args.input_dir, args.output_dir, args.target_count, args.sr)
