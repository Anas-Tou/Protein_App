import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import joblib
import os

# --- 1. Define the MLP Model Class ---
# This MUST be identical to the class you used for training.
IN_DIM = 1280 + 5 # 1280 (ESM) + 5 (tabular features)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLPRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- 2. Define Caching Functions for Loading ---
# @st.cache_resource is critical: it loads the models ONCE.
@st.cache_resource
def get_esm_model():
    print("Loading ESM-2 Model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(DEVICE)
    model.eval()
    print("ESM-2 Model Loaded.")
    return tokenizer, model

@st.cache_resource
def load_expert_models():
    print("Loading expert models...")
    models = {}
    scalers = {}
    artifact_path = "moe_artifacts"

    models["spec_57"] = MLPRegressor(IN_DIM).to(DEVICE)
    models["spec_57"].load_state_dict(torch.load(os.path.join(artifact_path, "model_spec_57.pth"), map_location=DEVICE))
    models["spec_57"].eval()
    scalers["spec_57"] = joblib.load(os.path.join(artifact_path, "scaler_spec_57.joblib"))
    
    models["spec_9"] = MLPRegressor(IN_DIM).to(DEVICE)
    models["spec_9"].load_state_dict(torch.load(os.path.join(artifact_path, "model_spec_9.pth"), map_location=DEVICE))
    models["spec_9"].eval()
    scalers["spec_9"] = joblib.load(os.path.join(artifact_path, "scaler_spec_9.joblib"))
    
    models["spec_34"] = MLPRegressor(IN_DIM).to(DEVICE)
    models["spec_34"].load_state_dict(torch.load(os.path.join(artifact_path, "model_spec_34.pth"), map_location=DEVICE))
    models["spec_34"].eval()
    scalers["spec_34"] = joblib.load(os.path.join(artifact_path, "scaler_spec_34.joblib"))

    models["generalist"] = MLPRegressor(IN_DIM).to(DEVICE)
    models["generalist"].load_state_dict(torch.load(os.path.join(artifact_path, "model_generalist.pth"), map_location=DEVICE))
    models["generalist"].eval()
    scalers["generalist"] = joblib.load(os.path.join(artifact_path, "scaler_generalist.joblib"))
    
    print("All models loaded.")
    return models, scalers

@st.cache_data  # Use cache_data for the lookup table
def load_lookup_table():
    print("Loading protein lookup table...")
    df_lookup = pd.read_csv(os.path.join("data", "Protein_Sequence_Data.csv"))
    # Keep only the columns we need and drop duplicates to make it fast
    df_lookup = df_lookup[['protSeq1', 'group1']].drop_duplicates(subset=['protSeq1'])
    return df_lookup

def get_embeddings(seq_batch, model, tokenizer):
    inputs = tokenizer(
        seq_batch, return_tensors='pt', padding=True, truncation=True, max_length=1024
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state
    mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.shape)
    summed_embeddings = torch.sum(last_hidden_states * mask, dim=1)
    summed_mask = torch.sum(inputs['attention_mask'], dim=1).unsqueeze(-1)
    summed_mask = torch.clamp(summed_mask, min=1e-9)
    pooled_embeddings = summed_embeddings / summed_mask
    return pooled_embeddings.cpu().numpy()

# --- 4. Load All Assets ---
esm_tokenizer, esm_model = get_esm_model()
expert_models, expert_scalers = load_expert_models()
lookup_df = load_lookup_table()

# --- 5. User Interface ---
st.title("🧬 Protein Stability (∆Tm) Predictor")
st.write("A 'Mixture-of-Experts' (MoE) model to predict protein stability.")

st.header("Inputs")
col1, col2 = st.columns(2)
with col1:
    tm1 = st.number_input("Original Tm (tm1)", value=62.9)
    pH1 = st.number_input("Original pH (pH1)", value=6.5)
with col2:
    # We still need pH2 to calculate deltaPh
    pH2 = st.number_input("Mutant pH (pH2)", value=6.5)

protSeq1 = st.text_area("Original Protein Sequence (protSeq1)", "MNAFEMLRIDERLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSEL...")
protSeq2 = st.text_area("Mutant Protein Sequence (protSeq2)", "MNDFEMLRIDERLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSEL...")

# --- 6. The "Search-and-Route" Logic ---
if st.button("Predict Stability (∆Tm)"):
    
    # 1. SEARCH for the group_id using protSeq1
    st.info("1. Searching for protein in database...")
    match = lookup_df[lookup_df['protSeq1'] == protSeq1]
    
    if not match.empty:
        group_id = match['group1'].iloc[0]
    else:
        group_id = -1 # A "default" ID for any protein not in our database
        st.warning("Protein not in database. Using 'Generalist' model.")

    # 2. ROUTE to the correct expert model
    if group_id == 57:
        model = expert_models["spec_57"]
        scaler = expert_scalers["spec_57"]
        st.info("2. Found Group 57. Using Specialist 57 Model.")
    elif group_id == 9:
        model = expert_models["spec_9"]
        scaler = expert_scalers["spec_9"]
        st.info("2. Found Group 9. Using Specialist 9 Model.")
    elif group_id == 34:
        model = expert_models["spec_34"]
        scaler = expert_scalers["spec_34"]
        st.info("2. Found Group 34. Using Specialist 34 Model.")
    else:
        # This now catches groups 5, 51, 50, 28 AND any new protein
        model = expert_models["generalist"]
        scaler = expert_scalers["generalist"]
        st.info("2. Using 'Generalist' Model for leftover/new groups.")
        
    try:
        with st.spinner("3. Generating protein embeddings... (This may take a moment)"):
            # 3. Get embeddings
            seqs = [protSeq1, protSeq2]
            embeddings = get_embeddings(seqs, esm_model, esm_tokenizer)
            emb_diff = (embeddings[1] - embeddings[0]).reshape(1, -1) # [1, 1280]

        # 4. Prepare tabular features
        deltaPh = pH2 - pH1
        seq_len_diff = len(protSeq2) - len(protSeq1)
        
        tabular_data = np.array([[
            tm1, pH1, pH2, deltaPh, seq_len_diff
        ]])
        
        # 5. Scale tabular features
        tabular_data_scaled = scaler.transform(tabular_data)
        
        # 6. Combine and predict
        final_features = np.concatenate([emb_diff, tabular_data_scaled], axis=1)
        features_tensor = torch.tensor(final_features, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(features_tensor)
            
        st.success(f"**Predicted ∆Tm: {prediction.item():.4f}**")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")