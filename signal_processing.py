import numpy as np
import librosa
import soundfile as sf
import os
from scipy import signal

class SignalProcessor:
    """
    Class for specialized signal processing operations to enhance engine sound classification,
    with particular focus on improving valve lash detection.
    """
    
    def __init__(self, sr=22050):
        """
        Initialize the signal processor.
        
        Args:
            sr (int): Sample rate for processing
        """
        self.sr = sr
    
    def bandpass_filter(self, audio, low_freq=500, high_freq=4000):
        """
        Apply a bandpass filter to focus on frequencies typical for valve lash issues.
        
        Args:
            audio (np.array): Audio signal
            low_freq (int): Lower cutoff frequency in Hz
            high_freq (int): Upper cutoff frequency in Hz
            
        Returns:
            np.array: Filtered audio signal
        """
        nyquist = 0.5 * self.sr
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Create a bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply the filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def enhance_transients(self, audio, threshold=0.1, boost_factor=2.0):
        """
        Enhance transient sounds which are characteristic of valve lash issues.
        
        Args:
            audio (np.array): Audio signal
            threshold (float): Threshold for detecting transients
            boost_factor (float): Factor to boost transients
            
        Returns:
            np.array: Audio with enhanced transients
        """
        # Compute the envelope
        envelope = np.abs(signal.hilbert(audio))
        
        # Compute the derivative of the envelope to detect rapid changes
        envelope_diff = np.diff(envelope, prepend=0)
        
        # Create a mask for transients
        transient_mask = np.zeros_like(audio)
        transient_mask[envelope_diff > threshold] = 1
        
        # Smooth the mask
        transient_mask = signal.convolve(transient_mask, 
                                         signal.windows.hann(int(0.01 * self.sr)), 
                                         mode='same')
        
        # Boost the transients
        enhanced_audio = audio.copy()
        enhanced_audio += audio * transient_mask * (boost_factor - 1)
        
        # Normalize
        if np.max(np.abs(enhanced_audio)) > 0:
            enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
        
        return enhanced_audio
    
    def harmonic_percussive_separation(self, audio, margin=3.0):
        """
        Separate harmonic and percussive components, emphasizing the percussive 
        elements that are often present in valve lash issues.
        
        Args:
            audio (np.array): Audio signal
            margin (float): Margin for separation
            
        Returns:
            np.array: Enhanced audio focusing on percussive elements
        """
        # Separate harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio, margin=margin)
        
        # Emphasize percussive components for valve lash detection
        enhanced = harmonic * 0.3 + percussive * 1.7
        
        # Normalize
        if np.max(np.abs(enhanced)) > 0:
            enhanced = enhanced / np.max(np.abs(enhanced))
        
        return enhanced
    
    def spectral_contrast_enhancement(self, audio, n_bands=6, boost=2.0):
        """
        Enhance spectral contrast to make valve lash sounds more distinguishable.
        
        Args:
            audio (np.array): Audio signal
            n_bands (int): Number of frequency bands
            boost (float): Boost factor for contrast
            
        Returns:
            np.array: Audio with enhanced spectral contrast
        """
        # Compute spectrogram
        S = librosa.stft(audio)
        
        # Compute spectral contrast
        contrast = librosa.feature.spectral_contrast(S=np.abs(S), sr=self.sr, n_bands=n_bands)
        
        # Enhance contrast
        contrast_enhanced = contrast * boost
        
        # Reconstruct signal (simplified approach)
        # This is a basic approximation - in a real implementation, you would 
        # modify the original spectrogram based on the enhanced contrast
        S_contrast = np.abs(S) * (1 + np.mean(contrast_enhanced, axis=0, keepdims=True))
        y_enhanced = librosa.istft(S_contrast * np.exp(1j * np.angle(S)))
        
        # Normalize
        if np.max(np.abs(y_enhanced)) > 0:
            y_enhanced = y_enhanced / np.max(np.abs(y_enhanced))
        
        return y_enhanced
    
    def process_valve_lash_audio(self, audio_path, output_path=None):
        """
        Apply a combination of processing techniques specifically designed
        for valve lash issue detection.
        
        Args:
            audio_path (str): Path to the audio file
            output_path (str, optional): Path to save the processed audio
            
        Returns:
            np.array: Processed audio signal
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Apply bandpass filter to focus on valve lash frequency range
        audio = self.bandpass_filter(audio, low_freq=800, high_freq=5000)
        
        # Enhance transients
        audio = self.enhance_transients(audio, threshold=0.05, boost_factor=2.5)
        
        # Apply harmonic-percussive separation
        audio = self.harmonic_percussive_separation(audio, margin=4.0)
        
        # Enhance spectral contrast
        audio = self.spectral_contrast_enhancement(audio, n_bands=6, boost=2.0)
        
        # Save processed audio if output path is provided
        if output_path:
            sf.write(output_path, audio, self.sr)
        
        return audio
    
    def process_valve_lash_dataset(self, input_dir, output_dir):
        """
        Process all valve lash audio files in a directory.
        
        Args:
            input_dir (str): Directory containing valve lash audio files
            output_dir (str): Directory to save processed files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each file
        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"enhanced_{filename}")
                
                print(f"Processing {filename}...")
                self.process_valve_lash_audio(input_path, output_path)
                
        print(f"All files processed and saved to {output_dir}")
    
    def extract_valve_lash_features(self, audio):
        """
        Extract features specifically designed to capture valve lash characteristics.
        
        Args:
            audio (np.array): Audio signal
            
        Returns:
            np.array: Extracted features
        """
        # Standard features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        
        # Add more features for better classification (same as standard extraction)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Valve lash specific features
        
        # 1. Spectral flatness - valve lash often has more noise-like components
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        
        # 2. Onset strength - valve lash has distinctive onsets
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        onset_mean = float(np.mean(onset_env))
        onset_std = float(np.std(onset_env))
        onset_max = float(np.max(onset_env))
        
        # 3. Rhythm features - valve lash has a specific rhythm
        # Handle empty beats array safely
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            # Ensure tempo is a scalar float
            tempo = float(tempo)
            
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                mean_intervals = float(np.mean(beat_intervals))
                std_intervals = float(np.std(beat_intervals))
            else:
                mean_intervals = 0.0
                std_intervals = 0.0
        except Exception as e:
            # Fallback values if beat tracking fails
            tempo = 0.0
            mean_intervals = 0.0
            std_intervals = 0.0
        
        # Combine all features to match the standard feature extraction length (41 features)
        # Standard features: 13 (mfcc mean) + 13 (mfcc std) + 1 (centroid) + 1 (rolloff) + 12 (chroma) + 1 (zcr) = 41
        features = np.concatenate([
            mfccs.mean(axis=1),                      # 13 features
            mfccs.std(axis=1),                       # 13 features
            spectral_centroid.mean(axis=1),          # 1 feature
            spectral_rolloff.mean(axis=1),           # 1 feature
            chroma.mean(axis=1),                     # 12 features
            zero_crossing_rate.mean(axis=1).reshape(-1)  # 1 feature
        ])
        
        # Ensure we have exactly 41 features to match the standard extraction
        assert len(features) == 41, f"Feature length mismatch: {len(features)} != 41"
        
        return features


def enhance_valve_lash_dataset(input_dir, output_dir, sr=22050):
    """
    Convenience function to enhance valve lash audio files.
    
    Args:
        input_dir (str): Directory containing valve lash audio files
        output_dir (str): Directory to save processed files
        sr (int): Sample rate
    """
    processor = SignalProcessor(sr=sr)
    processor.process_valve_lash_dataset(input_dir, output_dir)
