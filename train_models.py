#!/usr/bin/env python3
import os
from sound_classifier import SoundClassifier

def train_and_save_models(data_dir='data', models_dir='models'):
    """
    Train and save multiple models for engine sound classification.
    
    Args:
        data_dir (str): Directory containing the sound data
        models_dir (str): Directory to save the trained models
    """
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Model types to train
    model_types = ['rf', 'lr', 'svm', 'nn']
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model with benchmark data as 'normal'...")
        print(f"{'='*50}")
        
        # Initialize classifier with benchmark data included
        classifier = SoundClassifier(
            data_dir=data_dir, 
            model_type=model_type,
            include_benchmark=True
        )
        
        # Train the model
        classifier.train()
        
        # Save the model
        model_path = os.path.join(models_dir, f"{model_type}_sound_classifier_model.joblib")
        classifier.save_model(model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_models()
