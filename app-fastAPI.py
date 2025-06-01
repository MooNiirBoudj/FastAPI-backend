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

# Load training feature names if available
try:
    with open("training_feature_names.pkl", "rb") as f:
        training_feature_names = pickle.load(f)
    print(f"Training feature names loaded: {len(training_feature_names)} features")
except Exception as e:
    print(f"Warning: Could not load training feature names: {str(e)}")
    training_feature_names = None

app = FastAPI(title="Logistic Regression API & Gemini Career Assessment")

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

# -------------------- Feature alignment function --------------------
def align_features_with_training(df, training_feature_names):
    """
    Align test data features with training data features.
    Add missing columns with zeros and remove extra columns.
    """
    if training_feature_names is None:
        print("Warning: No training feature names available, using current features")
        return df
    
    # Get current feature names
    current_features = set(df.columns)
    expected_features = set(training_feature_names)
    
    # Find missing and extra features
    missing_features = expected_features - current_features
    extra_features = current_features - expected_features
    
    if missing_features:
        print(f"Adding {len(missing_features)} missing features with zeros")
        for feature in missing_features:
            df[feature] = 0
    
    if extra_features:
        print(f"Removing {len(extra_features)} extra features")
        df = df.drop(columns=list(extra_features))
    
    # Reorder columns to match training order
    df = df[training_feature_names]
    
    print(f"Final aligned features shape: {df.shape}")
    return df

# -------------------- Home --------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI App: Logistic Regression API & Gemini Career Assessment"}

# -------------------- Logistic Regression Route --------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: AnswersData):
    try:
        print("\n--- New prediction request received ---")
        
        # Check if model and encoders are loaded
        if lr_model is None or ordinal_encoder is None or onehot_encoder is None:
            raise HTTPException(status_code=500, detail="Model or encoders not loaded. Check server logs.")
        
        answers = data.answers
        print(f"Answers keys: {list(answers.keys()) if answers else 'None'}")
        
        # Create DataFrame
        try:
            df = pd.DataFrame([answers])
            print(f"Created DataFrame with shape: {df.shape}")
            print(f"DataFrame columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error creating DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error creating DataFrame: {str(e)}")

        # Clean up any curly quotes and strip whitespace
        try:
            df = df.applymap(lambda x: normalize_text(x) if isinstance(x, str) else x)
            print("Normalized text in DataFrame")
        except Exception as e:
            print(f"Error normalizing text: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error normalizing text: {str(e)}")

        # Define ordinal columns (these should match exactly what was used during training)
        ordinal_columns = [
            "Do you enjoy and feel comfortable with subjects like mathematics, physics, and biology?",
            "Are you excited by combining theoretical learning with hands-on practical work?",
            "How do you handle long study hours and challenging academic content?",
            "How comfortable are you navigating sensitive or emotional situations?",
            "How do you feel about public speaking or presenting?"
        ]
        
        # Check if all ordinal columns are present
        missing_ordinal_columns = [col for col in ordinal_columns if col not in df.columns]
        if missing_ordinal_columns:
            print(f"Missing required ordinal columns: {missing_ordinal_columns}")
            raise HTTPException(status_code=400, detail=f"Missing required ordinal columns: {missing_ordinal_columns}")

        # Apply ordinal encoding
        try:
            print("Applying ordinal encoding")
            df_ordinal = df[ordinal_columns].copy()
            df_ordinal_encoded = pd.DataFrame(
                ordinal_encoder.transform(df_ordinal),
                columns=ordinal_columns,
                index=df.index
            )
            print("Ordinal encoding complete")
        except Exception as e:
            print(f"Error in ordinal encoding: {str(e)}")
            print(f"Ordinal encoder categories: {ordinal_encoder.categories_}")
            print(f"Data values: {df[ordinal_columns].values}")
            raise HTTPException(status_code=400, detail=f"Error in ordinal encoding: {str(e)}")

        # Apply one-hot encoding to categorical columns
        try:
            categorical_columns = [col for col in df.columns if col not in ordinal_columns]
            print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
            
            if categorical_columns:
                print("Applying one-hot encoding")
                df_categorical = df[categorical_columns].copy()
                
                # Transform and get feature names
                onehot_encoded = onehot_encoder.transform(df_categorical)
                onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_columns)
                
                df_onehot_encoded = pd.DataFrame(
                    onehot_encoded,
                    columns=onehot_feature_names,
                    index=df.index
                )
                print(f"One-hot encoded shape: {df_onehot_encoded.shape}")
            else:
                print("No categorical columns to encode")
                df_onehot_encoded = pd.DataFrame(index=df.index)
                
        except Exception as e:
            print(f"Error in one-hot encoding: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error in one-hot encoding: {str(e)}")

        # Combine all encoded features
        try:
            print("Combining encoded features")
            final_df = pd.concat([
                df_ordinal_encoded.reset_index(drop=True),
                df_onehot_encoded.reset_index(drop=True)
            ], axis=1)
            print(f"Combined features shape: {final_df.shape}")
            print(f"Combined feature names: {list(final_df.columns)}")
        except Exception as e:
            print(f"Error combining encoded features: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error combining encoded features: {str(e)}")

        # Align features with training data
        try:
            print("Aligning features with training data")
            final_df = align_features_with_training(final_df, training_feature_names)
            print(f"Final aligned DataFrame shape: {final_df.shape}")
        except Exception as e:
            print(f"Error aligning features: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error aligning features: {str(e)}")
        
        # Mapping of class indices to field names
        label_map = {
            0: "Law",
            1: "Agriculture", 
            2: "Computer Science",
            3: "Medicine",
            4: "Business"
        }

        # Make prediction
        try:
            print("Making prediction with Logistic Regression model")
            print(f"Input shape for prediction: {final_df.shape}")
            
            prediction = lr_model.predict(final_df)
            predicted_class = int(prediction[0])
            predicted_field = label_map.get(predicted_class, "Unknown")
            print(f"Predicted class: {predicted_class}, field: {predicted_field}")

            # Get probabilities
            try:
                probabilities = lr_model.predict_proba(final_df)[0]
                probs_dict = {label_map[i]: float(prob) for i, prob in enumerate(probabilities)}
                print(f"Prediction probabilities: {probs_dict}")
            except Exception as e:
                print(f"Warning: Could not get prediction probabilities: {str(e)}")
                probs_dict = {}
                
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            print(f"Model expects features: {getattr(lr_model, 'n_features_in_', 'unknown')}")
            print(f"Provided features: {final_df.shape[1]}")
            print(traceback.format_exc())
            raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")

        print("Prediction completed successfully")
        return {
            "prediction": predicted_class,
            "field": predicted_field,
            "probabilities": probs_dict
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        print(f"Unexpected error in predict route: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# -------------------- Run App --------------------
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)