import pandas as pd
import streamlit as st
import numpy as np
from requests import post
import textwrap
import yaml
import os
import random
import hashlib
import time
import requests
from PIL import Image
from io import BytesIO

# Generate a random icon hash
def get_random_icon_hash():
    # Use current timestamp and a random number to create a unique string
    random_seed = f"{time.time()}{random.random()}"
    # Create a hash from this string
    hash_object = hashlib.md5(random_seed.encode())
    # Return the first 8 characters of the hash
    return hash_object.hexdigest()[:8]

# Load icon image from URL
def get_icon_image():
    icon_url = f"http://etc.porg.dev/icon/{get_random_icon_hash()}?rounded=80&ocs=true&single=true"
    try:
        response = requests.get(icon_url)
        if response.status_code == 200:
            return response.content.decode('utf-8').strip()
        else:
            return None
    except Exception:
        return None

# Set page configuration
st.set_page_config(
    page_title="Ocsai | Bulk Scoring",
    page_icon=get_icon_image(), # THIS IS DUMB. THE DEPLOYED APP DOESN"T LOAD DIRECTLY FROM URL - BUT DOING THIS ADDS BLOCKING OVERHEAD TO PAGE LOAD
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config file
def load_config():
    config_path = './config.yaml'
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config file: {e}")
        return None

config = load_config()

# Extract model information from config
def get_model_info():
    if not config or 'llmmodels' not in config:
        return [], [], [], [], [], []
    
    llm_models = config['llmmodels']
    
    model_options = [model['name'] for model in llm_models if model.get('production', False)]
    
    legacy_models = []
    chat_models = []
    ocsai1_models = []
    ocsai2_models = []
    ocsai2_langs = set()
    tasks_set = set()
    
    for model in llm_models:
        if not model.get('production', False):
            continue
            
        if model.get('style') == 'completion':
            legacy_models.append(model['name'])
        if model.get('style') == 'chat':
            chat_models.append(model['name'])
            
        if model.get('format') == 'classic' or model.get('format') == 'classic_multi':
            ocsai1_models.append(model['name'])
        elif model.get('format') == 'ocsai2':
            ocsai2_models.append(model['name'])
            
            # Collect languages and tasks for ocsai2 models
            if 'languages' in model:
                for lang in model['languages']:
                    if lang != 'CUSTOM':
                        ocsai2_langs.add(lang)
            
            if 'tasks' in model:
                for task in model['tasks']:
                    if task != 'CUSTOM':
                        tasks_set.add(task)
    return (
        model_options, 
        legacy_models, 
        chat_models, 
        ocsai1_models, 
        ocsai2_models, 
        sorted(list(ocsai2_langs)),
        sorted(list(tasks_set))
    )

# Get model information
model_options, legacy_models, chat_models, ocsai1_models, ocsai2_models, ocsai2_langs, tasks = get_model_info()

# Fallback to hardcoded values if config loading fails
if not model_options:
    raise ValueError("No models found in config")

# Language reference dictionary
langref = {
    "eng": "English",
    "spa": "Spanish",
    "fre": "French",
    "ger": "German",
    "ita": "Italian",
    "dut": "Dutch",
    "pol": "Polish",
    "rus": "Russian",
    "ara": "Arabic",
    "chi": "Chinese",
    "heb": "Hebrew",
}

# Streamlit page configuration
st.title("Bulk Scoring with Ocsai")
st.markdown(
    "This app allows you to score a large dataset using the OCS API. "
    "See details at <https://openscoring.du.edu/scoringllm>."
)

# Get API URL from config or use default
ocs_url = config.get('site', {}).get('api_url', 'https://openscoring.du.edu/')
if ocs_url.endswith('/'):
    ocs_url = ocs_url.rstrip('/') + '/llm'
elif not ocs_url.endswith('/llm'):
    ocs_url += '/llm'

# Debug the URL
st.sidebar.write(f"API URL: [{ocs_url}]({ocs_url.replace('/llm', '/docs')})")

# Find recommended model from config
default_model = next((model['name'] for model in config.get('llmmodels', []) 
                     if model.get('recommended', False) and model.get('production', True)), 
                     "ocsai-1.6")
default_lang = "eng"
default_task = "uses"

verify = True

def lang_formatter(x):
    return langref[x] if x in langref else x


def model_formatter(x):
    for model in config.get('llmmodels', []):
        if model['name'] == x:
            return f"{x} ({model.get('short-description', model.get('description', ''))})"
    
    # Fallback to old style if model not found in config
    style = "English, AUT" if x in ocsai1_models else "multi-lang, multi-task"
    return f"{x} ({style})"


def score_file(uploaded_file):
    if uploaded_file is not None:
        filename = uploaded_file.name

        if upload_format == "auto":
            if filename.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Only 'csv' and 'xlsx' files supported")
        elif upload_format == "csv":
            df = pd.read_csv(uploaded_file)
        elif upload_format == "excel":
            df = pd.read_excel(uploaded_file)

        # SCORING
        chunk_size = 5
        if model in ["ocsai-davinci3"]:
            chunk_size = 100

        all_results = []
        n_chunks = np.ceil(len(df) / chunk_size).astype(int)
        for i in range(n_chunks):
            i_start = i * chunk_size
            i_end = min(i_start + chunk_size, len(df))
            percent_complete = min(i / n_chunks, 100)
            my_bar.progress(
                percent_complete, text=f"Scoring in progress ({i+1} of {n_chunks})"
            )
            chunk = df.iloc[i_start:i_end]
            inputs = chunk[[name_of_prompt_column, name_of_response_column]]
            api_input = inputs.to_csv(None, header=False, index=False).strip()
            params = dict(input=api_input, model=model)

            if language is not None:
                params["language"] = language
            if task is not None:
                params["task"] = task
            if change_question_template:
                params["question_in_input"] = True
                params["prompt_in_input"] = False
            
            headers = {}
            if api_key is not None:
                headers["X-API-KEY"] = api_key
            else:
                headers["X-FORWARDED-FOR"] = "streamlit"
            
            max_retries = 2
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    result = post(ocs_url, data=params, headers=headers, verify=verify, timeout=120)
                    if result.status_code == 200:
                        all_results += result.json()["scores"]
                        success = True
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            # Exponential backoff: wait longer with each retry
                            wait_time = 2 ** retry_count
                            st.toast(f"Request failed (attempt {retry_count}/{max_retries}). Retrying in {wait_time} seconds...", icon="⚠️")
                            time.sleep(wait_time)
                        else:
                            st.error(f"Error code: {result.status_code}")
                            st.error(f"Error: {result.text}")
                            if result.status_code == 401:
                                st.error(
                                    "At usage limit. You may need to provide an API key to access this model, or wait a bit."
                                )
                            
                            # Create empty results with NA scores for this chunk
                            for _, row in chunk.iterrows():
                                empty_result = {
                                    "prompt" if not change_question_template else "question": row[name_of_prompt_column],
                                    "response": row[name_of_response_column],
                                    "score": np.nan,
                                    "confidence": np.nan,
                                    "flags": None,
                                    "language": language,
                                    "type": task
                                }
                                all_results.append(empty_result)
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count
                        st.warning(f"Request failed with exception: {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error(f"Failed after {max_retries} attempts: {str(e)}")
                        
                        # Create empty results with NA scores for this chunk
                        for _, row in chunk.iterrows():
                            empty_result = {
                                "prompt" if not change_question_template else "question": row[name_of_prompt_column],
                                "response": row[name_of_response_column],
                                "score": np.nan,
                                "confidence": np.nan,
                                "flags": None,
                                "language": language,
                                "type": task
                            }
                            all_results.append(empty_result)

        my_bar.empty()

        # MERGE RESULTS BACK INTO ORIGINAL DATA
        # Determine the column names in the API response
        api_prompt_col = "question" if change_question_template else "prompt"
        api_response_col = "response"  # This doesn't change
        
        col_mappings = {
            api_prompt_col: name_of_prompt_column,
            api_response_col: name_of_response_column,
        }
        
        scored = pd.DataFrame(all_results).rename(columns=col_mappings)
        
        # Check if the expected columns exist in the scored dataframe
        merge_cols = [name_of_prompt_column, name_of_response_column]
        
        # Make sure all required columns exist before merging
        if all(col in scored.columns for col in merge_cols):
            merged = df.merge(
                scored.drop_duplicates(merge_cols),
                on=merge_cols,
                how="left",
            )
            # Drop any API-specific columns that aren't needed
            cols_to_drop = ["confidence", "flags", "language", "type"]
            merged = merged.drop(columns=[col for col in cols_to_drop if col in merged.columns])
            return merged
        else:
            st.error(f"API response doesn't contain expected columns. Available columns: {scored.columns.tolist()}")
            st.write("API response sample:", scored.head())
            return None


with st.sidebar:
    icon = get_icon_image()
    model = st.selectbox(
        "Choose model",
        model_options,
        index=(
            model_options.index(default_model) if default_model in model_options else 0
        ),
        format_func=model_formatter,
    )

    input_method = st.radio(
        "Input method",
        ["Upload file", "Paste data"],
        help="Choose whether to upload a file or paste CSV data directly"
    )

    language: str | None = None
    task: str | None = None
    change_question_template: bool = False
    q_template: str | None = None

    if model in ocsai2_models:
        language = st.selectbox(
            "language",
            [None, "CUSTOM"] + ocsai2_langs,
            help=(
                "Select a language. "
                "Use `CUSTOM` to specify your own language, "
                "or `None` to leave it to the model to figure out."
            ),
            index=(
                2 + ocsai2_langs.index(default_lang)
                if default_lang in ocsai2_langs
                else 0
            ),
            format_func=lambda x: langref[x] if x in langref else x,
        )

        if language == "CUSTOM":
            language = st.text_input(
                "Custom 3-character language code "
                "([reference](https://www.loc.gov/standards/iso639-2/php/English_list.php))"
            )

        task = st.selectbox(
            "task",
            [None, "CUSTOM"] + tasks,
            help=(
                "Select a task. "
                "Use `CUSTOM` to specify your own task, "
                "or `None` to leave it to the model to figure out."
            ),
            index=2 + tasks.index(default_task) if default_task in tasks else 0,
        )
        if task == "CUSTOM":
            task = st.text_input("Custom task: type out a description of the goal")

        change_question_template = st.checkbox(
            "Use full question instead of short prompt",
            value=False,
            help=(
                textwrap.dedent('''
                If you want to use the full question from your dataset "
                instead of a short prompt, select this option. This is useful 
                when your dataset already contains complete questions rather 
                than just prompt keywords.

                 For example:

                - Short prompt: "brick"
                - Full question: "What is a surprising use for a brick?"

                The short prompt is all you need most of the time, especially when it
                is clear what the task is (e.g. for the alternate uses task, instances,
                or metaphors).

                For some tasks, you may want to write out the full question, especially
                when defining new tasks that the model doesn't know, or for tasks where
                there's no sensible 'short version' (e.g. a complete the sentence task).

                Not all models support full questions.''')
            ),
        )
        
        # Only show question template if not using full questions from dataset
        if not change_question_template:
            template_defaults = {
                "uses": "What is a surprising use for a {prompt}?",
                "completion": "Complete the sentence: {prompt}",
                "consequences": "What would happen if... {prompt}?",
                "instances": "Was is a surprising that that is {prompt}?",
                "metaphors": "What is a creative metaphor for {prompt}?",
            }
            default_template = ""
            if task in template_defaults:
                default_template = template_defaults[task]

    upload_format = st.selectbox(
        "Upload format",
        help=("If `auto`, the file extension is used to determine the " "format."),
        options=["csv", "excel", "auto"],
        index=2,
        disabled=input_method == "Paste data"
    )
    name_of_prompt_column = "prompt"
    name_of_response_column = "response"
    api_key = st.text_input("API Key", help="Optional API key for higher usage. [More details](https://buttondown.email/creativity/archive/interactive-drawing-assessments-ocsai-is-back/)")
    custom_names = st.checkbox(
        "Use custom column names",
        value=False,
        help=(
            "The first row of your data should name your columns. "
            "If the main data is called something different than "
            "prompt/response, select this option."
        ),
        disabled=input_method == "Paste data"
    )
    if custom_names:
        name_of_prompt_column = st.text_input("Name of prompt column", value="prompt")
        name_of_response_column = st.text_input(
            "Name of response column", value="response"
        )

    with st.form("scoring_form"):
        if input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Choose a file to upload", type=["csv", "xlsx"]
            )
            pasted_data = None
        else:
            uploaded_file = None
            pasted_data = st.text_area(
                "Paste CSV data",
                height=200,
                help="Format: prompt,response (one pair per line)"
            )

        st.form_submit_button("Score my data")

my_bar = st.progress(0, text="Scoring in progress")
my_bar.empty()

# Process pasted data if that method was chosen
if input_method == "Paste data" and pasted_data:
    import io
    # Convert pasted text to a file-like object
    csv_data = io.StringIO(pasted_data)
    # Read as CSV with no header
    df = pd.read_csv(csv_data, header=None, names=["prompt", "response"])
    # Set the column names to match what the scoring function expects
    df = df.rename(columns={"prompt": name_of_prompt_column, "response": name_of_response_column})
    
    # Create a temporary file-like object to pass to score_file
    temp_csv = io.StringIO()
    df.to_csv(temp_csv, index=False)
    temp_csv.seek(0)
    
    # Create a proper file-like object that implements the required interface
    class MockFile:
        def __init__(self, content, filename):
            self.content = content
            self.name = filename
            self._buffer = io.StringIO(content)
            
        def read(self, size=-1):
            return self._buffer.read(size)
            
    mock_file = MockFile(temp_csv.getvalue(), "pasted_data.csv")
    merged = score_file(mock_file)
else:
    merged = score_file(uploaded_file)

if merged is not None:
    st.write("Your Scored Data. Press the button to download the file.")
    st.dataframe(merged)
    
    # Create download button with filename based on the uploaded file or pasted data
    if uploaded_file is not None:
        original_filename = uploaded_file.name
        filename_without_ext, file_extension = os.path.splitext(original_filename)
        download_filename = f"{filename_without_ext}_scored{file_extension}"
        
        # Prepare the file for download based on the extension
        if file_extension.lower() == '.csv':
            csv_data = merged.to_csv(index=False)
            st.download_button(
                label="Download Scored Data",
                data=csv_data,
                file_name=download_filename,
                mime="text/csv"
            )
        elif file_extension.lower() in ['.xlsx', '.xls']:
            # For Excel files, we need to use a BytesIO object
            import io
            buffer = io.BytesIO()
            merged.to_excel(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                label="Download Scored Data",
                data=buffer,
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        # Handle pasted data case - always treat as CSV
        csv_data = merged.to_csv(index=False)
        st.download_button(
            label="Download Scored Data",
            data=csv_data,
            file_name="pasted_data_scored.csv",
            mime="text/csv"
        )
else:
    st.markdown(
        textwrap.dedent("""
            ## Instructions

            *Settings are in the sidebar. If you're on a small screen, you need to expand it.*

            1. Choose your settings.
                - `ocsai-1.6` supported multiple languages and multiple tasks, while `ocsai-4o` is good for English Alternate Uses scoring.
                - If uploading a file, specify what the name of the prompt and response columns are in your dataset.
                - If pasting data, ensure it's in CSV format with prompt in column 1 and response in column 2.
            2. Upload/paste your data and press the button to score it.
                        
            ## Citing OCS
            
            To cite the work which introduced automated scoring of divergent thinking with large language models:

            > "Organisciak, P., Acar, S., Dumas, D., & Berthiaume, K. (2023). Beyond semantic distance: Automated scoring of divergent thinking greatly improves with large language models. *Thinking Skills and Creativity*, 49, 101356. <https://doi.org/10.1016/j.tsc.2023.101356>"
                        
            A [pre-print version](http://dx.doi.org/10.13140/RG.2.2.32393.31840) is also available.
        """
        ).strip()
    )
