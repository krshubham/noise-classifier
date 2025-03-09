#!/usr/bin/env python3
import os
import argparse
import shutil
from data_augmentation import augment_dataset
from sound_classifier import SoundClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import numpy as np
from signal_processing import enhance_valve_lash_dataset

def main():
    parser = argparse.ArgumentParser(description='Augment data and train sound classifier models')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the dataset')
    parser.add_argument('--augmented-dir', type=str, default='augmented_data', help='Directory to save augmented data')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save trained models')
    parser.add_argument('--target-count', type=int, default=20, help='Target number of samples per class')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate')
    parser.add_argument('--duration', type=int, default=20, help='Duration in seconds')
    parser.add_argument('--skip-augmentation', action='store_true', help='Skip data augmentation')
    parser.add_argument('--enhance-valve-lash', action='store_true', help='Apply specialized valve lash enhancement')
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Clean and recreate augmented data directory
    if os.path.exists(args.augmented_dir) and not args.skip_augmentation:
        print(f"Removing existing augmented data directory: {args.augmented_dir}")
        shutil.rmtree(args.augmented_dir)
    
    os.makedirs(args.augmented_dir, exist_ok=True)
    
    # Apply specialized valve lash enhancement if requested
    if args.enhance_valve_lash:
        valve_lash_dir = os.path.join(args.data_dir, 'valve_lash')
        enhanced_valve_lash_dir = os.path.join(args.data_dir, 'enhanced_valve_lash')
        
        if os.path.exists(valve_lash_dir):
            print(f"Enhancing valve lash audio files...")
            enhance_valve_lash_dataset(valve_lash_dir, enhanced_valve_lash_dir, sr=args.sr)
            print(f"Valve lash enhancement complete. Enhanced files saved to {enhanced_valve_lash_dir}")
    
    # Augment data
    if not args.skip_augmentation:
        print("Augmenting data...")
        augment_dataset(args.data_dir, args.augmented_dir, args.target_count, args.sr)
    
    # Define model types and their hyperparameter grids
    model_types = ['rf', 'svm', 'nn', 'lr', 'xgb']
    
    param_grids = {
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'svm': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'nn': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        },
        'lr': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [1000, 2000]
        },
        'xgb': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    }
    
    # Train models
    for model_type in model_types:
        print(f"\nTraining {model_type.upper()} model...")
        
        # Create classifier
        classifier = SoundClassifier(
            data_dir=args.data_dir,
            model_type=model_type,
            sr=args.sr,
            duration=args.duration,
            augmented_data_dir=args.augmented_dir,
            use_enhanced_features=True  # Enable the specialized valve lash features
        )
        
        # Prepare data
        X, y_encoded, _ = classifier.prepare_data()
        
        # Scale features
        X_scaled = classifier.scaler.fit_transform(X)
        
        # Calculate class weights
        from sklearn.utils import class_weight
        class_weights = class_weight.compute_class_weight(
            'balanced', classes=np.unique(y_encoded), y=y_encoded
        )
        class_weights_dict = dict(zip(np.unique(y_encoded), class_weights))
        
        # Create base model for grid search
        if model_type == 'rf':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            base_model = SVC(random_state=42, probability=True)
        elif model_type == 'nn':
            base_model = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
        elif model_type == 'lr':
            base_model = LogisticRegression(random_state=42, multi_class='multinomial')
        elif model_type == 'xgb':
            base_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        
        # Perform grid search
        print(f"Performing grid search for {model_type.upper()}...")
        grid_search = GridSearchCV(
            base_model,
            param_grids[model_type],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        # Handle sample weights for XGBoost
        if model_type == 'xgb':
            sample_weights = np.ones(len(y_encoded))
            for i, y in enumerate(y_encoded):
                sample_weights[i] = class_weights_dict.get(y, 1.0)
            grid_search.fit(X_scaled, y_encoded, sample_weight=sample_weights)
        else:
            # For other models, use class_weight parameter if supported
            if model_type in ['rf', 'svm', 'lr']:
                grid_search.fit(X_scaled, y_encoded)
            else:
                grid_search.fit(X_scaled, y_encoded)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Set the best model
        classifier.model = grid_search.best_estimator_
        
        # Save the model
        model_path = os.path.join(args.models_dir, f"{model_type}_model.joblib")
        classifier.save_model(model_path)
        print(f"Model saved to {model_path}")
    
    print("\nTraining complete! All models have been saved.")
    print(f"You can now run the app with: python app.py")

if __name__ == "__main__":
    main()
