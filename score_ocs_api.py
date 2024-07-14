import pandas as pd
import streamlit as st
import numpy as np
from requests import post
import textwrap

# Streamlit page configuration
st.title("Bulk Scoring with Ocsai")
st.markdown(
    "This app allows you to score a large dataset using the OCS API. "
    "See details at <https://openscoring.du.edu/scoringllm>."
)

model_options = ["ocsai-davinci3", "ocsai-chatgpt2", "ocsai-1.5"]

legacy_models = ["ocsai-davinci3"]
chat_models = ["ocsai-chatgpt2", "ocsai-1.5"]
ocsai1_models = ["ocsai-davinci3", "ocsai-chatgpt2"]
ocsai2_models = ["ocsai-1.5"]
ocsai2_langs = [
    "ara",
    "chi",
    "dut",
    "eng",
    "fre",
    "ger",
    "heb",
    "ita",
    "pol",
    "rus",
    "spa",
]
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
tasks = ["uses", "completion", "consequences", "instances", "metaphors"]

ocs_url = "https://openscoring.du.edu/llm"
#ocs_url = 'http://127.0.0.1:5000/llm'
default_model = "ocsai-1.5"
default_lang = "eng"
default_task = "uses"

verify = True

def lang_formatter(x):
    return langref[x] if x in langref else x


def model_formatter(x):
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
            if q_template is not None:
                chunk["question"] = chunk["prompt"].apply(
                    lambda x: q_template.format(prompt=x)
                )
                inputs = chunk[
                    [name_of_prompt_column, "question", name_of_response_column]
                ]
            else:
                inputs = chunk[[name_of_prompt_column, name_of_response_column]]
            api_input = inputs.to_csv(None, header=False, index=False).strip()
            params = dict(input=api_input, model=model)

            if language is not None:
                params["language"] = language
            if task is not None:
                params["task"] = task
            if change_question_template:
                params["question_in_input"] = True
            
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
        col_mappings = {
            "prompt": name_of_prompt_column,
            "response": name_of_response_column,
        }
        scored = pd.DataFrame(all_results).rename(columns=col_mappings)
        merged = df.merge(
            scored.drop_duplicates(["prompt", "response"]),
            on=["prompt", "response"],
            how="left",
        ).drop(columns=["confidence", "flags", "language", "type"])
        return merged


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
            "Change question template",
            value=False,
            help=(
                "If you want to specify a custom question "
                "template, select this option. This is useful"
                " if a short 'prompt' is insufficient for the"
                " model to understand what is being scored -"
                " especially if you're scoring a task that the"
                " model wasn't trained on"
            ),
        )
        if change_question_template:
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

            q_template = st.text_input(
                "Question template",
                value=default_template,
                help=(
                    "This is the question template that will be used to "
                    "guide the model. It *needs* `{prompt}` included, where "
                    "the the prompt is inserted. For example, \n- "
                    "\n- ".join(template_defaults.values())
                ),
            )
            if "{prompt}" not in q_template:
                st.error("Your question template must include `{prompt}`")

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
    - `ocsai-chatgpt` is the slowest, though slightly more accurate than `ocsai-davinci2`.
2. Specify what the name of the prompt and response columns are in your dataset.
3. Upload your file and press the button to score it.
        """
        ).strip()
    )
