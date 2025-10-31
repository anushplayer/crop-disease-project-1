import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_crop_recommendation_model():
    # Load dataset (replace with actual crop dataset)
    # For demo, creating synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'nitrogen': np.random.uniform(0, 140, n_samples),
        'phosphorus': np.random.uniform(5, 145, n_samples),
        'potassium': np.random.uniform(5, 205, n_samples),
        'temperature': np.random.uniform(8, 45, n_samples),
        'humidity': np.random.uniform(14, 100, n_samples),
        'ph': np.random.uniform(3.5, 10, n_samples),
        'rainfall': np.random.uniform(20, 300, n_samples),
        'crop': np.random.choice(['Rice', 'Wheat', 'Cotton', 'Sugarcane', 'Maize'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Features and target
    X = df.drop('crop', axis=1)
    y = df['crop']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    # Save model
    joblib.dump(model, 'crop_recommendation_model.pkl')
    print("Crop recommendation model saved!")

if __name__ == "__main__":
    import numpy as np
    train_crop_recommendation_model()