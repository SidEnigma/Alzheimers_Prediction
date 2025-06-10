import streamlit as st
import io
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import requests


# FILE_PATH = 'https://raw.githubusercontent.com/SidEnigma/Alzheimers_Prediction/NeuroSyncAI/mci_dataset_neuropose_small/Healthy'
# ROOT_PATH = 'https://raw.githubusercontent.com/SidEnigma/Alzheimers_Prediction/NeuroSyncAI/mci_dataset_neuropose_small'   # directory with subfolders 'Healthy' and 'MildCognitiveDisorder'
LLM_PREDICTIONS_PATH = 'https://raw.githubusercontent.com/SidEnigma/Alzheimers_Prediction/tree/main/NeuroSyncAI/mci_dataset_neuropose_small/llm_subject_predictions_latest_eeg_neuropose.csv'
eeg_channels = ['AF3', 'AF4', 'C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4']
system_prompt = "You are a clinical assistant analyzing EEG patterns for MCI detection."


# GitHub utility functions
def list_github_files(directory):
    api_url = f"https://api.github.com/repos/SidEnigma/Alzheimers_Prediction/tree/main/NeuroSyncAI/{directory}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        files = response.json()
        return [f["name"] for f in files if f["type"] == "file"]
    except requests.RequestException as e:
        st.error(f"Error fetching file list from GitHub: {e}")
        return []


def load_csv_from_github(file_path):
    raw_url = f"https://raw.githubusercontent.com/SidEnigma/Alzheimers_Prediction/tree/main/NeuroSyncAI/{file_path}"
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except requests.RequestException as e:
        st.error(f"Error loading CSV file from GitHub: {e}")
        return None

# --- Helper Functions ---
def get_file_content(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.text


# Utility function to extract trial ID
def extract_trial_id(filename):
    match = re.search(r'(trial\d+)', filename)
    return match.group(1) if match else None

st.set_page_config(layout="wide")

# UI Layout
st.title("Analyzing EEG signals for MCI detection")
st.header("📈 Signal Visualization")

# Sidebar
st.sidebar.markdown("## Select Subject and EEG Channel")

# Subject Dropdown
condition = "Healthy"
file_list = list_github_files(f"mci_dataset_neuropose_small/{condition}")
subject_ids = sorted({extract_trial_id(f) for f in file_list if "eeg" in f and extract_trial_id(f)})

if not subject_ids:
    st.error("No EEG files found in the GitHub directory.")
    st.stop()

default_index_s = subject_ids.index("trial1")
subject = st.sidebar.selectbox("Select Subject", subject_ids, index=default_index_s)

# Condition Radio Buttons
condition = st.sidebar.radio("Select Subject's Condition", ["Healthy", "Mild Cognitive Disorder"])
if condition is not "Healthy":
    condition = "MCI"

# Channel Dropdown
default_index_c = eeg_channels.index("Fp1")
channel = st.sidebar.selectbox("Select EEG Channel", eeg_channels, index=default_index_c)

#  Visualization type selector
viz_type = st.sidebar.selectbox("Select type of visualization", ["Signal Line Graph", "Histogram"])

# Comparing Visualizations with another subject
st.sidebar.markdown("## Comparative Visualization")

# Asking user for comparison
user_choice = st.sidebar.radio("Compare with another subject?", ["Yes", "No"], index=1)

# Subject Dropdown for Comparison
default_index_s2 = subject_ids.index("trial2")
subject2 = st.sidebar.selectbox("Select 2nd Subject for comparison", subject_ids, index=default_index_s2)

# Condition Radio Buttons
condition2 = st.sidebar.radio("Select 2nd Subject's Condition", ["Healthy", "Mild Cognitive Disorder"], index=1)
if condition2 is not "Healthy":
    condition2 = "MCI"

# Channel Dropdown
default_index_c2 = eeg_channels.index("Fp1")
channel2 = st.sidebar.selectbox("Select 2nd Subject's EEG Channel", eeg_channels, index=default_index_c2)

try:
    file_list_main = list_github_files(f"mci_dataset_neuropose_small/{condition}")
    file_name_main = next((f for f in file_list_main if "eeg" in f and subject in f), None)
    if file_name_main is None:
        st.error("No EEG file found for the selected subject.")
        st.stop()

    df_trial = load_csv_from_github(f"mci_dataset_neuropose_small/{condition}/{file_name_main}")
    
    if df_trial is not None:
        if viz_type == "Signal Line Graph":
            fig, ax = plt.subplots(figsize=(30, 6))
            ax.plot(df_trial[channel][0:768], label=f'{condition} patient', color='cornflowerblue')

            if user_choice == "Yes":
                file_list_cmp = list_github_files(f"mci_dataset_neuropose_small/{condition2}")
                file_name_cmp = next((f for f in file_list_cmp if "eeg" in f and subject2 in f), None)
                df_trial2 = load_csv_from_github(f"mci_dataset_neuropose_small/{condition2}/{file_name_cmp}")
                if df_trial2 is not None:
                    ax.plot(df_trial2[channel2][0:768], label=f'{condition2} patient', color='orange')

            ax.set_xlabel('Time (samples)', fontsize=22)
            ax.set_ylabel('Amplitude (µV)', fontsize=22)
            ax.set_title(f'EEG Signal - {channel} Channel', fontsize=22)
            ax.tick_params(axis='both', labelsize=20)
            ax.legend(fontsize=20)
            st.pyplot(fig)

        elif viz_type == "Histogram":
            st.markdown(f'<h3>Histogram of EEG Channel {channel}</h3>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(30, 6))
            ax.hist(df_trial[channel], bins=50, alpha=0.5, color='cornflowerblue', label=condition, density=True)

            if user_choice == "Yes":
                file_list_cmp = list_github_files(f"mci_dataset_neuropose_small/{condition2}")
                file_name_cmp = next((f for f in file_list_cmp if "eeg" in f and subject2 in f), None)
                df_trial2 = load_csv_from_github(f"mci_dataset_neuropose_small/{condition2}/{file_name_cmp}")
                if df_trial2 is not None:
                    ax.hist(df_trial2[channel2], bins=50, alpha=0.5, color='orange', label=condition2, density=True)

            ax.set_xlabel("Amplitude (µV)", fontsize=16)
            ax.set_ylabel("Density", fontsize=16)
            ax.set_title(f"Histogram of EEG Channel {channel}", fontsize=18)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(fontsize=14)
            st.pyplot(fig)
except Exception as e:
    st.error(f"Unexpected error during visualization: {e}")

# LLM Prediction Section
st.header("🤖 Predict and Summarize Subject's Condition")
st.text_input("System Prompt", value=system_prompt, disabled=True)
ask_llm = st.button("Ask LLM")

if ask_llm:
    try:
        df_llm = pd.read_csv(LLM_PREDICTIONS_PATH)
        subject_key = f"{subject}_{condition}"
        result_row = df_llm[df_llm['Subject'] == subject_key]
        if not result_row.empty:
            llm_response = result_row.iloc[0]['LLM Output']
            prediction = result_row.iloc[0]['LLM Decision']
            confidence = str(result_row.iloc[0]['Confidence']).capitalize()

            st.markdown('<h3>LLM Response</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.markdown(f"**Prediction:** {prediction}")
            col2.markdown(f"**Confidence Level:** {confidence}")

            if '\n' in llm_response and ':' in llm_response.split('\n', 1)[1]:
                llm_response_trimmed = llm_response.split('\n', 1)[1].split(':', 1)[1].strip()
            else:
                llm_response_trimmed = llm_response

            st.text_area("Explanation", value=llm_response_trimmed, height=150)
        else:
            st.error(f"No LLM response found for subject: {subject_key}")
    except Exception as e:
        st.error(f"Error loading LLM predictions: {e}")