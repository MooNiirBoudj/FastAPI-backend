from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import pickle
import numpy as np
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import os
import traceback
import json
import re
import uvicorn

# Load logistic regression model and other files with better error handling
try:
    with open('best_model_LR.pkl', 'rb') as file:
        lr_model = pickle.load(file)
    print("Logistic Regression model loaded successfully")
except Exception as e:
    print(f"Error loading Logistic Regression model: {str(e)}")
    lr_model = None

try:
    with open("ordinal_encoder.pkl", "rb") as f:
        ordinal_encoder = pickle.load(f)
    print("Ordinal encoder loaded successfully")
except Exception as e:
    print(f"Error loading ordinal encoder: {str(e)}")
    ordinal_encoder = None

try:
    with open("onehot_encoder.pkl", "rb") as f:
        onehot_encoder = pickle.load(f)
    print("One-hot encoder loaded successfully")
except Exception as e:
    print(f"Error loading one-hot encoder: {str(e)}")
    onehot_encoder = None

# Load training feature names/columns in exact order
try:
    with open("training_feature_names.pkl", "rb") as f:
        training_feature_names = pickle.load(f)
    print(f"Training feature names loaded: {len(training_feature_names)} features")
    training_feature_indices = None
except Exception as e:
    print(f"Warning: Could not load training feature names: {str(e)}")
    training_feature_names = None
    
    # If feature names not available, try to load feature indices
    try:
        with open("training_feature_indices.pkl", "rb") as f:
            training_feature_indices = pickle.load(f)
        print(f"Training feature indices loaded: {len(training_feature_indices)} features")
    except Exception as e:
        print(f"Warning: Could not load training feature indices: {str(e)}")
        training_feature_indices = [0, 6, 13, 18, 30, 37, 44, 46, 50, 52, 56, 57, 58, 64] 
        print(f"Using hardcoded feature indices: {len(training_feature_indices)} features")

app = FastAPI(title="Logistic Regression API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global state for Gemini chat
chat_sessions = {}  # Using a dictionary to store multiple chat sessions
absolute_max_questions = 30  # Safety limit to prevent infinite questioning

# -------------------- Pydantic models for request/response --------------------
class AnswersData(BaseModel):
    answers: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: int
    field: str
    probabilities: Dict[str, float]

# -------------------- Normalisation function --------------------
def normalize_text(text):
    if not isinstance(text, str):
        return text
    return text.replace("'", "'").replace(""", "\"").replace(""", "\"")

# -------------------- Enhanced Feature alignment function --------------------
def align_features_with_training(df, training_feature_names=None, training_feature_indices=None):
    """
    Align test data features with training data features ensuring exact order and feature matching.
    """
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Input DataFrame columns: {list(df.columns)}")
    
    if training_feature_names is not None:
        # Method 1: Use feature names approach (RECOMMENDED)
        print("Aligning features using feature names (exact order preservation)")
        
        current_features = set(df.columns)
        expected_features = set(training_feature_names)
        
        # Find missing and extra features
        missing_features = expected_features - current_features
        extra_features = current_features - expected_features
        
        print(f"Current features: {len(current_features)}")
        print(f"Expected features: {len(expected_features)}")
        print(f"Missing features: {len(missing_features)} - {list(missing_features)[:5]}...")
        print(f"Extra features: {len(extra_features)} - {list(extra_features)[:5]}...")
        
        # Add missing features with zeros
        if missing_features:
            print(f"Adding {len(missing_features)} missing features with zeros")
            for feature in missing_features:
                df[feature] = 0
        
        # Remove extra features
        if extra_features:
            print(f"Removing {len(extra_features)} extra features")
            df = df.drop(columns=list(extra_features))
        
        # CRITICAL: Reorder columns to match EXACT training order
        try:
            df = df[training_feature_names]
            print(f"Features reordered to match training set exactly")
        except KeyError as e:
            print(f"Error reordering features: {e}")
            # Fallback: select only available features in training order
            available_features = [f for f in training_feature_names if f in df.columns]
            df = df[available_features]
            print(f"Used available features in training order: {len(available_features)}")
        
    elif training_feature_indices is not None:
        # Method 2: Use feature indices approach
        print("Aligning features using feature indices")
        print(f"Total available features: {df.shape[1]}")
        print(f"Required feature indices: {training_feature_indices}")
        
        # Check if all required indices are available
        max_required_index = max(training_feature_indices)
        if max_required_index >= df.shape[1]:
            available_features = df.shape[1]
            print(f"ERROR: Available features: {available_features}")
            print(f"Required max index: {max_required_index}")
            print(f"Required indices: {training_feature_indices}")
            raise ValueError(f"Required feature index {max_required_index} exceeds available features ({available_features})")
        
        # Select only the required features by index IN THE EXACT ORDER
        df = df.iloc[:, training_feature_indices]
        print(f"Selected features by indices in exact order: {df.shape}")
        
    else:
        print("WARNING: No training feature alignment info available!")
        print("Using all available features - this may cause prediction errors")
    
    print(f"Final aligned features shape: {df.shape}")
    print(f"Final feature order: {list(df.columns)[:10]}...")  # Show first 10 features
    return df

# -------------------- Enhanced encoding pipeline --------------------
def encode_features_to_match_training(df, ordinal_encoder, onehot_encoder, ordinal_columns):
    """
    Encode features in the exact same way as training data to ensure feature alignment.
    """
    print("\n--- Starting feature encoding pipeline ---")
    
    # Step 1: Apply ordinal encoding
    print("Step 1: Applying ordinal encoding")
    df_ordinal = df[ordinal_columns].copy()
    
    # Debug: Show values being encoded
    print("Ordinal values to encode:")
    for col in ordinal_columns:
        if col in df_ordinal.columns:
            print(f"  {col}: {df_ordinal[col].iloc[0]}")
    
    try:
        df_ordinal_encoded = pd.DataFrame(
            ordinal_encoder.transform(df_ordinal),
            columns=ordinal_columns,
            index=df.index
        )
        print(f"Ordinal encoding complete: {df_ordinal_encoded.shape}")
    except Exception as e:
        print(f"Ordinal encoding error: {e}")
        print("Available categories in ordinal encoder:")
        for i, col in enumerate(ordinal_columns):
            if i < len(ordinal_encoder.categories_):
                print(f"  {col}: {ordinal_encoder.categories_[i]}")
        raise

    # Step 2: Apply one-hot encoding to categorical columns
    print("Step 2: Applying one-hot encoding")
    categorical_columns = [col for col in df.columns if col not in ordinal_columns]
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    
    if categorical_columns:
        df_categorical = df[categorical_columns].copy()
        
        # Debug: Show values being encoded
        print("Categorical values to encode:")
        for col in categorical_columns:
            if col in df_categorical.columns:
                print(f"  {col}: {df_categorical[col].iloc[0]}")
        
        try:
            # Transform and get feature names
            onehot_encoded = onehot_encoder.transform(df_categorical)
            onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
            
            df_onehot_encoded = pd.DataFrame(
                onehot_encoded,
                columns=onehot_feature_names,
                index=df.index
            )
            print(f"One-hot encoding complete: {df_onehot_encoded.shape}")
            print(f"One-hot feature names (first 10): {list(onehot_feature_names)[:10]}")
            
        except Exception as e:
            print(f"One-hot encoding error: {e}")
            print("Available categories in one-hot encoder:")
            try:
                for i, col in enumerate(categorical_columns):
                    if hasattr(onehot_encoder, 'categories_') and i < len(onehot_encoder.categories_):
                        print(f"  {col}: {onehot_encoder.categories_[i][:5]}...")  # Show first 5 categories
            except:
                print("Could not display one-hot encoder categories")
            raise
    else:
        print("No categorical columns found")
        df_onehot_encoded = pd.DataFrame(index=df.index)

    # Step 3: Combine encoded features in consistent order
    print("Step 3: Combining encoded features")
    try:
        # Always combine in the same order: ordinal first, then one-hot
        final_df = pd.concat([
            df_ordinal_encoded.reset_index(drop=True),
            df_onehot_encoded.reset_index(drop=True)
        ], axis=1)
        print(f"Features combined successfully: {final_df.shape}")
        print(f"Combined feature order (first 10): {list(final_df.columns)[:10]}")
        
        return final_df
        
    except Exception as e:
        print(f"Error combining encoded features: {e}")
        raise

# -------------------- Home --------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI App: Logistic Regression API"}

# -------------------- Enhanced Logistic Regression Route --------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: AnswersData):
    try:
        print("\n" + "="*50)
        print("NEW PREDICTION REQUEST")
        print("="*50)
        
        # Check if model and encoders are loaded
        if lr_model is None or ordinal_encoder is None or onehot_encoder is None:
            raise HTTPException(status_code=500, detail="Model or encoders not loaded. Check server logs.")
        
        answers = data.answers
        print(f"Received answers for {len(answers)} questions")
        
        # Create DataFrame
        try:
            df = pd.DataFrame([answers])
            print(f"Created DataFrame: {df.shape}")
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating DataFrame: {str(e)}")

        # Clean up text
        try:
            df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
            print("Text normalization complete")
        except Exception as e:
            print(f"Error normalizing text: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error normalizing text: {str(e)}")

        # Define ordinal columns (MUST match training exactly)
        ordinal_columns = [
            "Do you enjoy and feel comfortable with subjects like mathematics, physics, and biology?",
            "Are you excited by combining theoretical learning with hands-on practical work?",
            "How do you handle long study hours and challenging academic content?",
            "How comfortable are you navigating sensitive or emotional situations?",
            "How do you feel about public speaking or presenting?"
        ]
        
        # Validate ordinal columns
        missing_ordinal = [col for col in ordinal_columns if col not in df.columns]
        if missing_ordinal:
            print(f"Missing ordinal columns: {missing_ordinal}")
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_ordinal}")

        # Apply encoding pipeline
        print("\n--- ENCODING PIPELINE ---")
        try:
            encoded_df = encode_features_to_match_training(
                df, ordinal_encoder, onehot_encoder, ordinal_columns
            )
        except Exception as e:
            print(f"Encoding pipeline failed: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Feature encoding failed: {str(e)}")

        # Apply feature alignment
        print("\n--- FEATURE ALIGNMENT ---")
        try:
            final_df = align_features_with_training(
                encoded_df, training_feature_names, training_feature_indices
            )
            print(f"Feature alignment complete: {final_df.shape}")
        except Exception as e:
            print(f"Feature alignment failed: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Feature alignment failed: {str(e)}")
        
        # Field mapping
        label_map = {
            0: "Law",
            1: "Agriculture", 
            2: "Computer Science",
            3: "Medicine",
            4: "Business"
        }

        # Make prediction
        print("\n--- MAKING PREDICTION ---")
        try:
            print(f"Final input shape: {final_df.shape}")
            print(f"Model expects: {getattr(lr_model, 'n_features_in_', 'unknown')} features")
            
            # Ensure data types are correct
            final_df = final_df.astype(float)
            
            prediction = lr_model.predict(final_df)
            predicted_class = int(prediction[0])
            predicted_field = label_map.get(predicted_class, "Unknown")
            
            print(f"Prediction successful!")
            print(f"Predicted class: {predicted_class}")
            print(f"Predicted field: {predicted_field}")

            # Get probabilities
            try:
                probabilities = lr_model.predict_proba(final_df)[0]
                probs_dict = {label_map[i]: float(prob) for i, prob in enumerate(probabilities)}
                print(f"Probabilities: {probs_dict}")
            except Exception as e:
                print(f"Warning: Could not get probabilities: {str(e)}")
                probs_dict = {}
                
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            print(f"Input shape: {final_df.shape}")
            print(f"Input dtypes: {final_df.dtypes.value_counts()}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

        print("\n" + "="*50)
        print("PREDICTION COMPLETED SUCCESSFULLY")
        print("="*50)
        
        return {
            "prediction": predicted_class,
            "field": predicted_field,
            "probabilities": probs_dict
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# -------------------- Run App --------------------
if __name__ == '__main__':
    print("Starting FastAPI server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run("app-fastAPI:app", host="127.0.0.1", port=8000, reload=True)