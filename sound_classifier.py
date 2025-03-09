import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import xgboost as xgb
import joblib
from signal_processing import SignalProcessor


class SoundClassifier:
    def __init__(self, data_dir, model_type='rf', sr=22050, duration=20, include_benchmark=True, 
                 use_class_weights=True, augmented_data_dir=None, use_enhanced_features=True):
        self.data_dir = data_dir
        self.sr = sr
        self.duration = duration
        self.model = None
        self.le = LabelEncoder()
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.include_benchmark = include_benchmark
        self.use_class_weights = use_class_weights
        self.augmented_data_dir = augmented_data_dir
        self.use_enhanced_features = use_enhanced_features
        self.signal_processor = SignalProcessor(sr=sr)

    def extract_features(self, file_path):
        # Load audio file
        y, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)

        # Pad or truncate to fixed length
        if len(y) < self.sr * self.duration:
            y = np.pad(y, (0, self.sr * self.duration - len(y)))
        else:
            y = y[:self.sr * self.duration]

        # Check if this is a valve lash file and use enhanced features if enabled
        if self.use_enhanced_features and ('valve_lash' in file_path or 'enhanced_valve_lash' in file_path):
            # Apply valve lash specific processing
            y = self.signal_processor.bandpass_filter(y, low_freq=800, high_freq=5000)
            y = self.signal_processor.enhance_transients(y, threshold=0.05, boost_factor=2.5)
            
            # Use specialized valve lash feature extraction
            return self.signal_processor.extract_valve_lash_features(y)
        
        # Standard feature extraction for other audio types
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        
        # Add more features for better classification
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Compute statistics - ensure all arrays are 1D
        features = np.concatenate([
            mfccs.mean(axis=1),
            mfccs.std(axis=1),
            spectral_centroid.mean(axis=1),
            spectral_rolloff.mean(axis=1),
            chroma.mean(axis=1),
            zero_crossing_rate.mean(axis=1).reshape(-1)  # Ensure 1D array
        ])

        return features

    def prepare_data(self):
        X = []
        y = []
        
        # Determine which data directory to use
        data_dirs = [self.data_dir]
        if self.augmented_data_dir and os.path.exists(self.augmented_data_dir):
            data_dirs = [self.augmented_data_dir]  # Use only augmented data if available
            print(f"Using augmented data from {self.augmented_data_dir}")
        
        # Check if enhanced valve lash directory exists and add it
        enhanced_valve_lash_dir = os.path.join(os.path.dirname(self.data_dir), 'enhanced_valve_lash')
        if self.use_enhanced_features and os.path.exists(enhanced_valve_lash_dir):
            print(f"Including enhanced valve lash data from {enhanced_valve_lash_dir}")
            # Add enhanced valve lash directory to data dirs if using original data
            if self.data_dir in data_dirs:
                data_dirs.append(enhanced_valve_lash_dir)
        
        for data_dir in data_dirs:
            # Iterate through each issue folder
            for issue in os.listdir(data_dir):
                issue_path = os.path.join(data_dir, issue)
                if os.path.isdir(issue_path):
                    # Skip benchmark folder if not included
                    if issue == 'benchmark' and not self.include_benchmark:
                        continue
                    
                    # Process each audio file in the folder
                    for audio_file in os.listdir(issue_path):
                        if audio_file.endswith('.wav'):
                            file_path = os.path.join(issue_path, audio_file)
                            features = self.extract_features(file_path)
                            X.append(features)
                            
                            # Label benchmark data as 'normal' and other folders as their respective issues
                            if issue == 'benchmark':
                                y.append('normal')
                            # Handle enhanced valve lash directory
                            elif data_dir == enhanced_valve_lash_dir:
                                y.append('valve_lash')
                            else:
                                y.append(issue)
        
        print(f"Total samples: {len(X)}")
        
        # Count samples per class
        class_counts = {}
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        print(f"Class distribution: {class_counts}")
        
        # Check feature dimensions
        feature_lengths = [len(x) for x in X]
        if len(set(feature_lengths)) > 1:
            print(f"Warning: Inconsistent feature lengths detected: {set(feature_lengths)}")
            # Find the most common feature length
            from collections import Counter
            most_common_length = Counter(feature_lengths).most_common(1)[0][0]
            print(f"Standardizing to length {most_common_length}")
            
            # Standardize feature lengths
            X_standardized = []
            y_standardized = []
            for i, x in enumerate(X):
                if len(x) == most_common_length:
                    X_standardized.append(x)
                    y_standardized.append(y[i])
                else:
                    print(f"Skipping sample with length {len(x)}")
            
            X = X_standardized
            y = y_standardized
            
            print(f"After standardization: {len(X)} samples")
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        
        return X, y_encoded, y

    def train(self):
        # Prepare data
        X, y_encoded, y_original = self.prepare_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate class weights if enabled
        class_weights = None
        if self.use_class_weights:
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train
            )
            class_weights = dict(zip(np.unique(y_train), class_weights))
            print(f"Using class weights: {class_weights}")

        # Train model based on model_type
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight=class_weights if self.use_class_weights else None,
                n_jobs=-1  # Use all available cores
            )
        elif self.model_type == 'lr':
            self.model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight=class_weights if self.use_class_weights else None,
                multi_class='multinomial',
                solver='lbfgs'
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf', 
                random_state=42,
                class_weight=class_weights if self.use_class_weights else None,
                probability=True  # Enable probability estimates
            )
        elif self.model_type == 'nn':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50), 
                max_iter=1000, 
                random_state=42,
                early_stopping=True,  # Enable early stopping
                validation_fraction=0.1  # Use 10% of training data for validation
            )
        elif self.model_type == 'xgb':
            # Prepare sample weights if class weights are enabled
            sample_weights = None
            if self.use_class_weights:
                sample_weights = np.ones(len(y_train))
                for i, y in enumerate(y_train):
                    sample_weights[i] = class_weights.get(y, 1.0)
            
            # Create XGBoost model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1  # Use all available cores
            )
            
            # Fit with sample weights if available
            if sample_weights is not None:
                self.model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
            else:
                self.model.fit(X_train_scaled, y_train)
                
            # Skip the general fit below since we've already fit the model
            fitted = True
        else:
            raise ValueError("Invalid model type. Choose 'rf', 'lr', 'svm', 'nn', or 'xgb'.")

        # Fit the model if not already fitted
        if not locals().get('fitted', False):
            self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        print(f"\nModel Performance ({self.model_type}):")
        print(classification_report(y_test, y_pred,
                                    labels=np.unique(y_test),
                                    target_names=self.le.classes_[np.unique(y_test)]))

        return self.model

    def predict(self, audio_file):
        # Extract features from new audio
        features = self.extract_features(audio_file)

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        predicted_label = self.le.inverse_transform([prediction])[0]
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = proba[prediction]
        
        return predicted_label, confidence

    def save_model(self, model_path='sound_classifier_model.joblib'):
        """Save the trained model, label encoder, and scaler"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")

        model_data = {
            'model': self.model,
            'label_encoder': self.le,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'include_benchmark': self.include_benchmark,
            'use_class_weights': self.use_class_weights,
            'use_enhanced_features': self.use_enhanced_features
        }
        joblib.dump(model_data, model_path)

    @classmethod
    def load_model(cls, model_path='sound_classifier_model.joblib'):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        
        # Create instance with appropriate parameters
        classifier = cls(
            data_dir=None, 
            model_type=model_data.get('model_type', 'rf'),
            include_benchmark=model_data.get('include_benchmark', True),
            use_class_weights=model_data.get('use_class_weights', True),
            use_enhanced_features=model_data.get('use_enhanced_features', True)
        )
        
        classifier.model = model_data['model']
        classifier.le = model_data['label_encoder']
        classifier.scaler = model_data['scaler']
        return classifier
