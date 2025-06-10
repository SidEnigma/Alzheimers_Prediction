import streamlit as st
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


FILE_PATH = 'NeuroSyncAI\mci_dataset_neuropose_small\Healthy'
ROOT_PATH = 'NeuroSyncAI\mci_dataset_neuropose_small'   # directory with subfolders 'Healthy' and 'MildCognitiveDisorder'
LLM_PREDICTIONS_PATH = 'NeuroSyncAI\mci_dataset_neuropose_small\llm_subject_predictions_latest_eeg_neuropose.csv'
eeg_channels = ['AF3', 'AF4', 'C3', 'C4', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4']
system_prompt = "You are a clinical assistant analyzing EEG patterns for MCI detection."

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
subject_ids = []
for file in os.listdir(FILE_PATH):
    if "eeg" in file:
        trial_id = extract_trial_id(file)
        subject_ids.append(trial_id)

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

# Load and Plot EEG Signal
subject_file_path = os.path.join(ROOT_PATH, condition)
if os.path.exists(subject_file_path):
    for file in os.listdir(subject_file_path):
        if "eeg" and subject in file:
            file_path = os.path.join(subject_file_path, file)
            df_trial = pd.read_csv(file_path)

            if viz_type == "Signal Line Graph":
                # st.markdown(f'<h3>EEG Signal from {channel} Channel (First 3s)</h3>', unsafe_allow_html=True)
            
                fig, ax = plt.subplots(figsize=(30, 6))
                ax.plot(df_trial[channel][0:768], label=f'{condition} patient', color='cornflowerblue')

                # If comparing with another subject
                if str(user_choice) == "Yes" and subject2 and condition2 and channel2:
                    # Load and Plot 2nd Subject's EEG Signal
                    subject2_file_path = os.path.join(ROOT_PATH, condition2)
                    for file2 in os.listdir(subject2_file_path):
                        if "eeg" and subject2 in file2:
                            file_path2 = os.path.join(subject2_file_path, file2)
                            df_trial2 = pd.read_csv(file_path2)
                            ax.plot(df_trial2[channel2][0:768], label=f'{condition2} patient', color='orange')
                            break

                ax.set_xlabel('Time (samples)', fontsize=22)
                ax.set_ylabel('Amplitude (µV)', fontsize=22)
                ax.set_title(f'EEG Signal - {channel} Channel', fontsize=22)
                ax.tick_params(axis='both', labelsize=20)
                ax.legend(fontsize=20)
                st.pyplot(fig)
                break

            elif viz_type == "Histogram":
                st.markdown(f'<h3>Histogram of EEG Channel {channel}</h3>', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(30, 6))
                ax.hist(df_trial[channel], bins=50, alpha=0.5, color='cornflowerblue', label=condition, density=True)

                # If comparing with another subject
                if str(user_choice) == "Yes" and subject2 and condition2 and channel2:
                    # Load and Plot 2nd Subject's EEG Signal
                    subject2_file_path = os.path.join(ROOT_PATH, condition2)
                    for file2 in os.listdir(subject2_file_path):
                        if "eeg" and subject2 in file2:
                            file_path2 = os.path.join(subject2_file_path, file2)
                            df_trial2 = pd.read_csv(file_path2)
                            ax.hist(df_trial2[channel2], bins=50, alpha=0.5, color='orange', label=condition2, density=True)
                            break

                ax.set_xlabel("Amplitude (µV)", fontsize=16)
                ax.set_ylabel("Density", fontsize=16)
                ax.set_title(f"Histogram of EEG Channel {channel}", fontsize=18)
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(fontsize=14)
                st.pyplot(fig)
                break
else:
    st.warning(f"File not found: {subject_file_path}")

st.header("🤖 Predict and Summarize Subject's Condition")

# System Prompt + Ask LLM Button
st.text_input("System Prompt", value=system_prompt, disabled=True)
ask_llm = st.button("Ask LLM")

# LLM Response Box
if ask_llm:
    if os.path.exists(LLM_PREDICTIONS_PATH):
        df_llm = pd.read_csv(LLM_PREDICTIONS_PATH)
        subject_key = f"{subject}_{condition}"
        result_row = df_llm[df_llm['Subject'] == subject_key]
        if not result_row.empty:
            llm_response = result_row.iloc[0]['LLM Output']
            prediction = result_row.iloc[0]['LLM Decision']
            confidence = str(result_row.iloc[0]['Confidence']).capitalize()

            # Display heading
            st.markdown('<h3>LLM Response</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.markdown(f"**Prediction:** {prediction}")
            col2.markdown(f"**Confidence Level:** {confidence}")

            llm_response_trimmed = llm_response.split('\n', 1)[1].split(':', 1)[1].strip() if '\n' in llm_response and ':' in llm_response.split('\n', 1)[1] else llm_response
            st.text_area("Explanation", value=llm_response_trimmed, height=150)

        else:
            st.error(f"No LLM response found for subject: {subject_key}")
    else:
        st.error("LLM response file not found.")
