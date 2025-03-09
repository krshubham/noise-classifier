# import gradio as gr
import os
import joblib
from sound_classifier import SoundClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tempfile
from collections import Counter

# Get list of available models and their friendly names
MODELS_DIR = 'models'
MODEL_NAMES = {
    'lr_model.joblib': 'Logistic Regression',
    'nn_model.joblib': 'Neural Network',
    'rf_model.joblib': 'Random Forest',
    'svm_model.joblib': 'Support Vector Machine',
    'xgb_model.joblib': 'XGBoost'
}

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_model.joblib')]
model_choices = {MODEL_NAMES[file]: file for file in model_files if file in MODEL_NAMES}

print(model_choices)

def load_model(model_file):
    """Load a saved model and its associated scaler and label encoder"""
    model_path = os.path.join(MODELS_DIR, model_file)
    saved_data = joblib.load(model_path)
    return saved_data['model'], saved_data['scaler'], saved_data['label_encoder']

def format_issue(issue_text):
    """Format the issue text to be more readable"""
    # Replace underscores with spaces and title case the text
    if issue_text == 'normal':
        return 'Normal Engine Sound (No Issues)'
    formatted = issue_text.replace('_', ' ').title()
    return formatted

def get_all_model_predictions(audio_file):
    """Get predictions from all available models"""
    results = {}
    highest_confidence = 0
    best_model = None
    best_prediction = None
    all_predictions = []
    all_confidences = {}
    
    # Initialize classifier for feature extraction only
    classifier = SoundClassifier(data_dir='data')
    features = classifier.extract_features(audio_file)
    features = features.reshape(1, -1)
    
    # Get predictions from each model
    for model_name, model_file in model_choices.items():
        try:
            model, scaler, le = load_model(model_file)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            predicted_label = le.inverse_transform([prediction])[0]
            formatted_label = format_issue(predicted_label)
            
            # Get confidence
            confidence = 0
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
                confidence = proba[prediction]
            
            results[model_name] = {
                'label': formatted_label,
                'confidence': confidence,
                'raw_label': predicted_label  # Store raw label for voting
            }
            
            # Track highest confidence
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_model = model_name
                best_prediction = formatted_label
            
            # Store for voting
            all_predictions.append(predicted_label)
            if predicted_label not in all_confidences:
                all_confidences[predicted_label] = []
            all_confidences[predicted_label].append(confidence)
                
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            results[model_name] = {
                'label': 'Error',
                'confidence': 0,
                'raw_label': 'error'
            }
    
    # Perform voting
    vote_results = Counter(all_predictions)
    if vote_results:
        # Get the most common prediction
        voted_prediction, vote_count = vote_results.most_common(1)[0]
        
        # Calculate average confidence for the voted prediction
        avg_confidence = np.mean(all_confidences.get(voted_prediction, [0]))
        
        # Format the voted prediction
        voted_formatted = format_issue(voted_prediction)
        
        # Add voting results
        results['Ensemble (Voting)'] = {
            'label': voted_formatted,
            'confidence': avg_confidence,
            'raw_label': voted_prediction,
            'vote_count': vote_count,
            'total_votes': len(all_predictions)
        }
        
        # Check if voting has higher confidence than individual models
        if avg_confidence > highest_confidence:
            highest_confidence = avg_confidence
            best_model = 'Ensemble (Voting)'
            best_prediction = voted_formatted
    
    return results, best_model, best_prediction, highest_confidence

def create_confidence_chart(results, best_model):
    """Create a bar chart of confidence scores"""
    models = []
    confidences = []
    colors = []
    
    for model, data in results.items():
        models.append(model)
        confidences.append(data['confidence'] * 100)  # Convert to percentage
        # Highlight the best model
        if model == best_model:
            colors.append('green')
        elif model == 'Ensemble (Voting)':
            colors.append('purple')  # Highlight voting in a different color
        else:
            colors.append('blue')
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, confidences, color=colors)
    plt.xlabel('Model')
    plt.ylabel('Confidence (%)')
    plt.title('Model Confidence Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name

def create_voting_chart(results):
    """Create a pie chart showing the voting distribution"""
    if 'Ensemble (Voting)' not in results:
        return None
        
    # Count votes for each class
    vote_counts = {}
    for model, data in results.items():
        if model != 'Ensemble (Voting)':  # Skip the ensemble result itself
            raw_label = data.get('raw_label', 'unknown')
            if raw_label not in vote_counts:
                vote_counts[raw_label] = 0
            vote_counts[raw_label] += 1
    
    # Create pie chart
    labels = [format_issue(label) for label in vote_counts.keys()]
    counts = list(vote_counts.values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Voting Distribution')
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name)
        plt.close()
        return tmp.name

def predict_sound(audio_file):
    """
    Function to make predictions on uploaded audio files using all models
    and show a comparison chart
    """
    if not audio_file:
        return "Please upload an audio file", None, None
    
    # Get predictions from all models
    results, best_model, best_prediction, highest_confidence = get_all_model_predictions(audio_file)
    
    # Create confidence comparison chart
    confidence_chart = create_confidence_chart(results, best_model)
    
    # Create voting distribution chart
    voting_chart = create_voting_chart(results)
    
    # Format the text output
    output_text = f"Best Prediction: {best_prediction} (Confidence: {highest_confidence:.2%})\n\n"
    
    # Add voting details if available
    if 'Ensemble (Voting)' in results:
        voting_data = results['Ensemble (Voting)']
        output_text += f"Ensemble Voting Result: {voting_data['label']} "
        output_text += f"(Confidence: {voting_data['confidence']:.2%}, "
        output_text += f"Votes: {voting_data['vote_count']}/{voting_data['total_votes']})\n\n"
    
    output_text += "All Model Predictions:\n"
    
    for model, data in results.items():
        if model != 'Ensemble (Voting)':  # Skip ensemble in this section
            confidence_str = f"{data['confidence']:.2%}" if data['confidence'] > 0 else "N/A"
            output_text += f"- {model}: {data['label']} (Confidence: {confidence_str})\n"
    
    return output_text, confidence_chart, voting_chart

# Create Gradio interface
# iface = gr.Interface(
#     fn=predict_sound,
#     inputs=gr.Audio(type="filepath", label="Upload Sound File"),
#     outputs=[
#         gr.Textbox(label="Prediction Results"),
#         gr.Image(label="Confidence Comparison"),
#         gr.Image(label="Voting Distribution")
#     ],
#     title="Engine Sound Issue Classifier",
#     description="Upload an audio file of engine sound to identify potential issues or normal operation. The system will compare predictions across all available models and use ensemble voting to provide a consensus prediction.",
#     examples=[
#         [os.path.join("test_data", "air_filter_sample_5.wav")],
#         [os.path.join("test_data", "cd_sample_16.wav")],
#         [os.path.join("test_data", "vl_sample_4.wav")],
#         # Add example for fan belt issue if available
#         [os.path.join("test_data", "fan_belt_sample.wav") if os.path.exists(os.path.join("test_data", "fan_belt_sample.wav")) else None]
#     ]
# )

# if __name__ == "__main__":
#     iface.launch()
