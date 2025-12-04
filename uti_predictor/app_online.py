import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import pandas as pd
import numpy as np
from io import BytesIO

# Define the optimal model architecture
class ClassifierNN(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function
        self.fcn1 = nn.Linear(14, 32)
        self.fcn2 = nn.Linear(32, 64)
        self.fcn3 = nn.Linear(64, 128)
        self.fcn4 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation_function(self.fcn1(x))
        x = self.activation_function(self.fcn2(x))
        x = self.activation_function(self.fcn3(x))
        x = self.fcn4(x)
        return x

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "uti_mlp_model.pth")
    scaler_path = os.path.join(base_path, "scaler.pkl")

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = ClassifierNN(activation_function=F.relu)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

model, scaler = load_model_and_scaler()

# Encoding functions
def encode_dataframe(df):
    """Encode a full DataFrame of patient data"""
    MAPPING_REFERENCE = {
        "ABSENCE_REFERENCE": {
            "NONE SEEN": 0, "RARE": 1, "FEW": 2, "OCCASIONAL": 3,
            "MODERATE": 4, "LOADED": 5, "PLENTY": 6
        },
        "Color": {
            "LIGHT YELLOW": 0, "STRAW": 1, "AMBER": 2, "BROWN": 3,
            "DARK YELLOW": 4, "YELLOW": 5, "REDDISH YELLOW": 6,
            "REDDISH": 7, "LIGHT RED": 8, "RED": 9
        },
        "Transparency": {
            "CLEAR": 0, "SLIGHTLY HAZY": 1, "HAZY": 2, "CLOUDY": 3, "TURBID": 4
        },
        "Protein_and_Glucose": {
            "NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5
        }
    }
    
    wbc_rbc_map = {
        "0-2": 0, "2-4": 1, "4-6": 2, "6-8": 3, "8-10": 4,
        "10-15": 5, "15-20": 6, ">20": 7, "RARE": 8, "FEW": 9,
        "MODERATE": 10, "PLENTY": 11
    }
    
    encoded_df = df.copy()
    
    # Encode categorical features
    encoded_df['Color'] = encoded_df['Color'].map(MAPPING_REFERENCE["Color"])
    encoded_df['Transparency'] = encoded_df['Transparency'].map(MAPPING_REFERENCE["Transparency"])
    encoded_df['Glucose'] = encoded_df['Glucose'].map(MAPPING_REFERENCE["Protein_and_Glucose"])
    encoded_df['Protein'] = encoded_df['Protein'].map(MAPPING_REFERENCE["Protein_and_Glucose"])
    encoded_df['WBC'] = encoded_df['WBC'].map(wbc_rbc_map)
    encoded_df['RBC'] = encoded_df['RBC'].map(wbc_rbc_map)
    encoded_df['Epithelial Cells'] = encoded_df['Epithelial Cells'].map(MAPPING_REFERENCE["ABSENCE_REFERENCE"])
    encoded_df['Mucous Threads'] = encoded_df['Mucous Threads'].map(MAPPING_REFERENCE["ABSENCE_REFERENCE"])
    encoded_df['Amorphous Urates'] = encoded_df['Amorphous Urates'].map(MAPPING_REFERENCE["ABSENCE_REFERENCE"])
    encoded_df['Bacteria'] = encoded_df['Bacteria'].map(MAPPING_REFERENCE["ABSENCE_REFERENCE"])
    encoded_df['FEMALE'] = (encoded_df['Gender'] == 'FEMALE').astype(int)
    encoded_df = encoded_df.drop('Gender', axis=1)
    
    return encoded_df

def predict_batch(df):
    """Batch-wise predictions"""
    continuous_features = [
        "Age", "Color", "Transparency", "Glucose", "Protein", "pH",
        "Specific Gravity", "WBC", "RBC", "Epithelial Cells",
        "Mucous Threads", "Amorphous Urates", "Bacteria"
    ]
    
    # Scale continuous features
    df[continuous_features] = scaler.transform(df[continuous_features])
    
    # Convert to tensor
    input_tensor = torch.tensor(df.values, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1).numpy()
        confidences = torch.max(probabilities, dim=1).values.numpy()
    
    return predictions, confidences

def encode_single_input(inputs):
    """Encode a single patient input"""
    MAPPING_REFERENCE = {
        "ABSENCE_REFERENCE": {
            "NONE SEEN": 0, "RARE": 1, "FEW": 2, "OCCASIONAL": 3,
            "MODERATE": 4, "LOADED": 5, "PLENTY": 6
        },
        "Color": {
            "LIGHT YELLOW": 0, "STRAW": 1, "AMBER": 2, "BROWN": 3,
            "DARK YELLOW": 4, "YELLOW": 5, "REDDISH YELLOW": 6,
            "REDDISH": 7, "LIGHT RED": 8, "RED": 9
        },
        "Transparency": {
            "CLEAR": 0, "SLIGHTLY HAZY": 1, "HAZY": 2, "CLOUDY": 3, "TURBID": 4
        },
        "Protein_and_Glucose": {
            "NEGATIVE": 0, "TRACE": 1, "1+": 2, "2+": 3, "3+": 4, "4+": 5
        }
    }
    
    wbc_rbc_map = {
        "0-2": 0, "2-4": 1, "4-6": 2, "6-8": 3, "8-10": 4,
        "10-15": 5, "15-20": 6, ">20": 7, "RARE": 8, "FEW": 9,
        "MODERATE": 10, "PLENTY": 11
    }
    
    encoded = {
        "Age": inputs['age'],
        "Color": MAPPING_REFERENCE["Color"][inputs['color']],
        "Transparency": MAPPING_REFERENCE["Transparency"][inputs['transparency']],
        "pH": inputs['ph'],
        "Glucose": MAPPING_REFERENCE["Protein_and_Glucose"][inputs['glucose']],
        "Protein": MAPPING_REFERENCE["Protein_and_Glucose"][inputs['protein']],
        "Specific Gravity": inputs['specific_gravity'],
        "WBC": wbc_rbc_map[inputs['wbc']],
        "RBC": wbc_rbc_map[inputs['rbc']],
        "Epithelial Cells": MAPPING_REFERENCE["ABSENCE_REFERENCE"][inputs['epithelial_cells']],
        "Mucous Threads": MAPPING_REFERENCE["ABSENCE_REFERENCE"][inputs['mucous_threads']],
        "Amorphous Urates": MAPPING_REFERENCE["ABSENCE_REFERENCE"][inputs['amorphous_urates']],
        "Bacteria": MAPPING_REFERENCE["ABSENCE_REFERENCE"][inputs['bacteria']],
        "FEMALE": 1 if inputs['gender'] == "FEMALE" else 0
    }
    
    return pd.DataFrame([encoded])

def create_csv_template():
    """CSV template for users to download"""
    template = pd.DataFrame({
        'Patient_ID': ['P001', 'P002'],
        'Age': [30, 45],
        'Gender': ['FEMALE', 'MALE'],
        'Color': ['YELLOW', 'STRAW'],
        'Transparency': ['CLEAR', 'SLIGHTLY HAZY'],
        'pH': [6.0, 5.5],
        'Glucose': ['NEGATIVE', 'NEGATIVE'],
        'Protein': ['NEGATIVE', 'TRACE'],
        'Specific Gravity': [1.020, 1.015],
        'WBC': ['0-2', '4-6'],
        'RBC': ['0-2', '2-4'],
        'Epithelial Cells': ['RARE', 'FEW'],
        'Mucous Threads': ['NONE SEEN', 'RARE'],
        'Amorphous Urates': ['NONE SEEN', 'NONE SEEN'],
        'Bacteria': ['NONE SEEN', 'FEW']
    })
    return template

# Streamlit UI
st.title("🔬 UTI Predictor")
st.write("Predict UTI diagnosis from urinalysis test results")

# Create tabs for single patient vs batch processing
tab1, tab2 = st.tabs(["Single Patient", "Batch Processing (CSV)"])

# TAB 1: Single Patient Input
with tab1:
    st.subheader("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        color = st.selectbox("Color", options=[
            "LIGHT YELLOW", "STRAW", "AMBER", "BROWN", "DARK YELLOW", 
            "YELLOW", "REDDISH YELLOW", "REDDISH", "LIGHT RED", "RED"
        ])
        transparency = st.selectbox("Transparency", options=[
            "CLEAR", "SLIGHTLY HAZY", "HAZY", "CLOUDY", "TURBID"
        ])
        ph = st.number_input("pH", min_value=4.0, max_value=11.0, value=6.0, step=0.1)
        glucose = st.selectbox("Glucose", options=[
            "NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"
        ])
        protein = st.selectbox("Protein", options=[
            "NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"
        ])
        specific_gravity = st.number_input("Specific Gravity", min_value=1.000, max_value=1.050, value=1.020, step=0.001)
    
    with col2:
        wbc = st.selectbox("WBC", options=[
            "0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20", 
            ">20", "RARE", "FEW", "MODERATE", "PLENTY"
        ])
        rbc = st.selectbox("RBC", options=[
            "0-2", "2-4", "4-6", "6-8", "8-10", "10-15", "15-20",
            ">20", "RARE", "FEW", "MODERATE", "PLENTY"
        ])
        epithelial_cells = st.selectbox("Epithelial Cells", options=[
            "NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"
        ])
        mucous_threads = st.selectbox("Mucous Threads", options=[
            "NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"
        ])
        amorphous_urates = st.selectbox("Amorphous Urates", options=[
            "NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"
        ])
        bacteria = st.selectbox("Bacteria", options=[
            "NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "LOADED", "PLENTY"
        ])
        gender = st.selectbox("Gender", options=["FEMALE", "MALE"])
    
    if st.button("Predict", type="primary"):
        user_inputs = {
            'age': age, 'color': color, 'transparency': transparency,
            'ph': ph, 'glucose': glucose, 'protein': protein,
            'specific_gravity': specific_gravity, 'wbc': wbc, 'rbc': rbc,
            'epithelial_cells': epithelial_cells, 'mucous_threads': mucous_threads,
            'amorphous_urates': amorphous_urates, 'bacteria': bacteria,
            'gender': gender
        }
        
        encoded_df = encode_single_input(user_inputs)
        
        continuous_features = [
            "Age", "Color", "Transparency", "Glucose", "Protein", "pH",
            "Specific Gravity", "WBC", "RBC", "Epithelial Cells",
            "Mucous Threads", "Amorphous Urates", "Bacteria"
        ]
        
        encoded_df[continuous_features] = scaler.transform(encoded_df[continuous_features])
        input_tensor = torch.tensor(encoded_df.values, dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        st.write("---")
        if prediction == 1:
            st.error(f"⚠️ **POSITIVE for UTI** (Confidence: {confidence*100:.1f}%)")
            st.write("Model indicates a positive classification. Please correlate with laboratory findings, patient history, and clinical judgment.")
        else:
            st.success(f"✓ **NEGATIVE for UTI** (Confidence: {confidence*100:.1f}%)")
            st.write("Model indicates a negative classification. Continue to evaluate alongside clinical presentation and diagnostic results.")

# TAB 2: Batch Processing
with tab2:
    st.subheader("Batch Process Multiple Patients")
    
    # Download template section
    st.write("### Step 1: Download CSV Template")
    st.write("Download the template below, fill in your patient data, and upload it back.")
    
    template = create_csv_template()
    csv_buffer = BytesIO()
    template.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_buffer,
        file_name="uti_prediction_template.csv",
        mime="text/csv"
    )
    
    st.write("**Required columns:** Patient_ID, Age, Gender, Color, Transparency, pH, Glucose, Protein, Specific Gravity, WBC, RBC, Epithelial Cells, Mucous Threads, Amorphous Urates, Bacteria")
    
    # Upload section
    st.write("### Step 2: Upload Filled CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV
            input_df = pd.read_csv(uploaded_file)
            
            st.write(f"**Loaded {len(input_df)} patients**")
            st.dataframe(input_df.head(), use_container_width=True)
            
            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Store Patient_ID for later
                    patient_ids = input_df['Patient_ID'] if 'Patient_ID' in input_df.columns else range(len(input_df))
                    
                    # Drop Patient_ID before encoding
                    if 'Patient_ID' in input_df.columns:
                        input_df_no_id = input_df.drop('Patient_ID', axis=1)
                    else:
                        input_df_no_id = input_df
                    
                    # Encode and predict
                    encoded_df = encode_dataframe(input_df_no_id)
                    predictions, confidences = predict_batch(encoded_df)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'Patient_ID': patient_ids,
                        'Prediction': ['POSITIVE' if p == 1 else 'NEGATIVE' for p in predictions],
                        'Confidence': [f"{c*100:.1f}%" for c in confidences]
                    })
                    
                    # Combine with original data
                    final_results = pd.concat([
                        input_df.reset_index(drop=True),
                        results_df[['Prediction', 'Confidence']].reset_index(drop=True)
                    ], axis=1)
                    
                    st.success(f"✅ Predictions complete for {len(final_results)} patients!")
                    
                    # Display results
                    st.dataframe(final_results, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive_count = sum(predictions == 1)
                        st.metric("Positive Cases", positive_count)
                    with col2:
                        negative_count = sum(predictions == 0)
                        st.metric("Negative Cases", negative_count)
                    with col3:
                        avg_confidence = np.mean(confidences) * 100
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    # Download results
                    results_buffer = BytesIO()
                    final_results.to_csv(results_buffer, index=False)
                    results_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=results_buffer,
                        file_name="uti_predictions_results.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your CSV has all required columns with correct values.")

st.write("---")

st.caption("⚕️ This tool serves as an initial screening layer and should not be considered a substitute for confirmatory testing.")


