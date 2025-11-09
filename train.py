#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - Workout Type Recommendation Model Training Script

This script trains the final XGBoost model for workout type recommendation
and saves it for deployment.

Usage:
    python train.py
"""

import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

DATA_PATH = 'models/data_prepared.pkl'
MODEL_OUTPUT_PATH = 'models/final_model.pkl'

# Model hyperparameters (tuned from notebook experiments)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'mlogloss'
}

# ============================================
# FUNCTIONS
# ============================================

def load_data(data_path):
    """Load prepared training data."""
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ Data loaded successfully!")
    print(f"   Training samples: {len(data['X_train'])}")
    print(f"   Validation samples: {len(data['X_val'])}")
    print(f"   Test samples: {len(data['X_test'])}")
    print(f"   Features: {len(data['feature_names'])}")
    
    return data

def prepare_labels(y_train, y_val, y_test):
    """Encode target labels for XGBoost."""
    print("\nEncoding target labels...")
    label_encoder = LabelEncoder()
    
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"✅ Labels encoded: {label_encoder.classes_}")
    
    return y_train_encoded, y_val_encoded, y_test_encoded, label_encoder

def train_model(X_train, y_train_encoded, params):
    """Train XGBoost model."""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    print("\nModel parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    model = XGBClassifier(**params)
    
    print("\n🔄 Training in progress...")
    model.fit(X_train, y_train_encoded)
    
    print("✅ Training complete!")
    
    return model

def evaluate_model(model, X_train, X_val, X_test, 
                   y_train, y_val, y_test, 
                   label_encoder):
    """Evaluate model performance."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Convert back to original labels
    y_train_pred_labels = label_encoder.inverse_transform(y_train_pred)
    y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_train_pred_labels)
    val_acc = accuracy_score(y_val, y_val_pred_labels)
    test_acc = accuracy_score(y_test, y_test_pred_labels)
    
    print("\n📊 Accuracy Scores:")
    print(f"  Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\n📋 Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred_labels))
    
    print("\n📊 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred_labels)
    classes = sorted(y_test.unique())
    
    # Print confusion matrix nicely
    print("\n       ", end="")
    for cls in classes:
        print(f"{cls:>10}", end="")
    print()
    
    for i, cls in enumerate(classes):
        print(f"{cls:>10}", end="")
        for j in range(len(classes)):
            print(f"{cm[i][j]:>10}", end="")
        print()
    
    return train_acc, val_acc, test_acc

def get_feature_importance(model, feature_names, top_n=15):
    """Get and display feature importance."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    for idx, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")
    
    return importance_df

def save_model(model, label_encoder, scaler, feature_names, 
               train_acc, val_acc, test_acc, output_path):
    """Save the trained model and associated components."""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    model_package = {
        'model': model,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_names': feature_names,
        'model_type': 'XGBoost',
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'classes': label_encoder.classes_.tolist()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved to: {output_path}")
    print("\n📦 Saved components:")
    print(f"  - Model: XGBoost Classifier")
    print(f"  - Label Encoder: {len(label_encoder.classes_)} classes")
    print(f"  - Scaler: StandardScaler")
    print(f"  - Features: {len(feature_names)} features")
    print(f"  - Train Accuracy: {train_acc:.4f}")
    print(f"  - Validation Accuracy: {val_acc:.4f}")
    print(f"  - Test Accuracy: {test_acc:.4f}")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main training pipeline."""
    print("="*60)
    print("WORKOUT TYPE RECOMMENDATION - MODEL TRAINING")
    print("="*60)
    
    # 1. Load data
    data = load_data(DATA_PATH)
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    scaler = data['scaler']
    feature_names = data['feature_names']
    
    # 2. Encode labels
    y_train_encoded, y_val_encoded, y_test_encoded, label_encoder = prepare_labels(
        y_train, y_val, y_test
    )
    
    # 3. Train model
    model = train_model(X_train, y_train_encoded, MODEL_PARAMS)
    
    # 4. Evaluate model
    train_acc, val_acc, test_acc = evaluate_model(
        model, X_train, X_val, X_test,
        y_train, y_val, y_test,
        label_encoder
    )
    
    # 5. Feature importance
    importance_df = get_feature_importance(model, feature_names)
    
    # 6. Save model
    save_model(
        model, label_encoder, scaler, feature_names,
        train_acc, val_acc, test_acc,
        MODEL_OUTPUT_PATH
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel ready for deployment at: {MODEL_OUTPUT_PATH}")
    print("\nNext steps:")
    print("  1. Test the model with predict.py")
    print("  2. Build Flask API")
    print("  3. Create Dockerfile")
    print("  4. Deploy to cloud")

if __name__ == "__main__":
    main()