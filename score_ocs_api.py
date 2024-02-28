import pandas as pd
import streamlit as st
import numpy as np
from requests import post
import textwrap

# Streamlit page configuration
st.title('Bulk Scoring with Ocsai')
st.markdown("This app allows you to score a large dataset using the OCS API. "
            "See details at <https://openscoring.du.edu/scoringllm>.")

model_options = ["ocsai-davinci2", "ocsai-babbage2", "ocsai-chatgpt"]

def score_file(uploaded_file):
    if uploaded_file is not None:
        filename = uploaded_file.name

        if upload_format == 'auto':
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Only 'csv' and 'xlsx' files supported")
        elif upload_format == 'csv':
            df = pd.read_csv(uploaded_file)
        elif upload_format == 'excel':
            df = pd.read_excel(uploaded_file)

        # SCORING
        chunk_size = 20
        if model in ['ocsai-davinci2', 'ocsai-babbage2']:
            chunk_size = 100
        
        all_results = []
        n_chunks = np.ceil(len(df) / chunk_size).astype(int)
        for i in range(n_chunks):
            i_start = i * chunk_size
            i_end = min(i_start + chunk_size, len(df))
            percent_complete = min(i / n_chunks, 100)
            my_bar.progress(percent_complete, text=f"Scoring in progress ({i+1} of {n_chunks})")
            chunk = df.iloc[i:i_end]
            inputs = chunk[[name_of_prompt_column, name_of_response_column]]
            api_input = inputs.to_csv(None, header=False, index=False).strip()
            params = dict(input=api_input, model=model)
            result = post(ocs_url, data=params, timeout=120)
            all_results += result.json()['scores']

        my_bar.empty()

        # MERGE RESULTS BACK INTO ORIGINAL DATA
        col_mappings = {'prompt': name_of_prompt_column, 'response': name_of_response_column}
        scored = pd.DataFrame(all_results).rename(columns=col_mappings)
        merged = df.merge(scored.drop_duplicates(['prompt', 'response']), on=['prompt', 'response'],
                        how='left').drop(columns=['confidence', 'flags'])
        return merged


with st.sidebar:
    with st.form("scoring_form"):
        # PARAMS
        ocs_url = 'https://openscoring.du.edu/llm'
        model = st.selectbox("Choose model", model_options, index=0)
        upload_format = st.selectbox("Upload format (if auto, extension is used)", ["csv", "excel", "auto"], index=2)
        name_of_prompt_column = st.text_input('Name of prompt column', value='prompt')
        name_of_response_column = st.text_input('Name of response column', value='response')

        uploaded_file = st.file_uploader("Choose a file to upload", type=["csv", "xlsx"])

        my_bar = st.progress(0, text="Scoring in progress")
        my_bar.empty()
        st.form_submit_button('Score my file')

merged = score_file(uploaded_file)
if merged is not None:
    st.write("Your Scored Data. Press the button to download the file.")
    st.dataframe(merged)
else:
    st.markdown(textwrap.dedent('''## Instructions
1. Choose your settings
    - `ocsai-chatgpt` is the slowest, though slightly more accurate than `ocsai-davinci2`.
2. Specify what the name of the prompt and response columns are in your dataset.
3. Upload your file and press the button to score it.
        ''').strip())