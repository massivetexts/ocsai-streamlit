import pandas as pd
import streamlit as st
import numpy as np
from requests import post
import textwrap
import yaml
import os

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
st.sidebar.write(f"API URL: {ocs_url}")

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
        chunk_size = 20
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
            chunk = df.iloc[i:i_end]
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
                headers["X-FORWARDED-FOR"] = "streamlit"
            else:
                headers["X-API-KEY"] = api_key
            result = post(ocs_url, data=params, headers=headers, verify=verify, timeout=120)
            if result.status_code != 200:
                st.error(f"Error code: {result.status_code}")
                st.error(f"Error: {result.text}")
                if result.status_code == 401:
                    st.error(
                        "At usage limit. You may need to provide an API key to access this model, or wait a bit."
                    )
                return
            else:
                all_results += result.json()["scores"]

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
    model = st.selectbox(
        "Choose model",
        model_options,
        index=(
            model_options.index(default_model) if default_model in model_options else 0
        ),
        format_func=model_formatter,
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
                "If you want to use the full question from your dataset "
                "instead of a short prompt, select this option. This is useful "
                "when your dataset already contains complete questions rather "
                "than just prompt keywords."
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
    )
    if custom_names:
        name_of_prompt_column = st.text_input("Name of prompt column", value="prompt")
        name_of_response_column = st.text_input(
            "Name of response column", value="response"
        )

    with st.form("scoring_form"):
        uploaded_file = st.file_uploader(
            "Choose a file to upload", type=["csv", "xlsx"]
        )

        st.form_submit_button("Score my file")

my_bar = st.progress(0, text="Scoring in progress")
my_bar.empty()
merged = score_file(uploaded_file)

if merged is not None:
    st.write("Your Scored Data. Press the button to download the file.")
    st.dataframe(merged)
else:
    st.markdown(
        textwrap.dedent(
            """## Instructions
1. Choose your settings
2. Specify what the name of the prompt and response columns are in your dataset.
3. Upload your file and press the button to score it.
        """
        ).strip()
    )
