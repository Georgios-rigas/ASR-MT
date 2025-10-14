# Translation Analysis Dashboard for SageMaker

# FORCE PYTHON PATH FOR SAGEMAKER
import sys
sys.path.insert(0, '/opt/conda/lib/python3.12/site-packages')

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re # Import the regular expression module for cleaning
import os, shutil



# Import jiwer for Word Error Rate and accuracy calculation
try:
    import jiwer
    jiwer_available = True
except ImportError:
    jiwer_available = False

# Set page config
st.set_page_config(page_title="ASR-MT DEMO", layout="wide")

# Title
st.title("🌐 ASR-MT DEMO")

FINETUNING_METRICS = {
    "Icelandic": {"hours_train": 2.85, "base_wer": 57.52, "ft_wer": 37.99},
    "Belarusian": {"hours_train": 9.52, "base_wer": 71.70, "ft_wer": 17.72},
    "Nepali": {"hours_train": 12.51, "base_wer": 85.14, "ft_wer": 34.68}
}

# Helper function to clean strings for accurate comparison
def clean_string(text):
    """
    Cleans a string by making it lowercase, stripping whitespace, removing punctuation,
    and normalizing all internal whitespace to a single space.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Replace multiple whitespace characters with a single space
    return re.sub(r'\s+', ' ', text)

def calculate_word_match_ratio(reference, generated):
    """
    Calculates the ratio of correctly matched words (higher is better).
    Uses the jiwer library to ensure accurate, word-level comparison.
    """
    if not jiwer_available:
        return 0.0

    ref_clean = clean_string(reference)
    gen_clean = clean_string(generated)

    if not ref_clean:
        return 1.0 if not gen_clean else 0.0

    try:
        measures = jiwer.compute_measures(ref_clean, gen_clean)
        total_words_in_ref = measures['hits'] + measures['substitutions'] + measures['deletions']
        
        if total_words_in_ref == 0:
            return 1.0

        word_match_ratio = measures['hits'] / total_words_in_ref
        return word_match_ratio
    except Exception:
        return 0.0

# --- Load external data files ---
@st.cache_data
def load_data():
    """Loads all required Excel files for the dashboard."""
    try:
        metrics_df = pd.read_excel('metrics.xlsx')
        translations_df = pd.read_excel('translations.xlsx')
        # We only load the examples file for the fine-tuning tab now
        finetuning_examples_df = pd.read_excel('finetuning_examples.xlsx')
        return metrics_df, translations_df, finetuning_examples_df
    except FileNotFoundError as e:
        st.error(f"Error: A required data file is missing. Please ensure 'metrics.xlsx', 'translations.xlsx', and 'finetuning_examples.xlsx' are all present. Missing file: {e.filename}")
        return None, None, None

metrics_df, translations_df, finetuning_examples_df = load_data()


# --- Main App Body ---
if all(df is not None for df in [metrics_df, translations_df, finetuning_examples_df]):
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Audio Transcription & Translation Inference",
        "Translation Performance Overview",
        "Translation Models Inference",
        "Low-Resource Fine-Tuning"
    ])

    # ==========================
    # VIEW 3: Audio Transcription & Translation
    # ==========================
    try:
        if shutil.which("ffmpeg") is None:
            import imageio_ffmpeg as ioff
        os.environ["IMAGEIO_FFMPEG_EXE"] = ioff.get_ffmpeg_exe()
    except Exception as e:
        print("FFmpeg setup error:", e)
    with tab1:
        st.header("🎤 Audio Transcription & Translation")

        # --- (Your setup code for libraries) ---
        try:
            import whisper
            from transformers import pipeline
            import torch
            libraries_available = True
        except ImportError:
            st.error("Required libraries (whisper, transformers, torch) are not installed.")
            libraries_available = False

        if libraries_available:
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                whisper_model_size = st.selectbox("Select Whisper Model Size", ["base", "small", "medium", "large"])
            with col2:
                translation_model = st.selectbox("Select Translation Model", ["Helsinki-NLP/opus-mt-es-en", "facebook/mbart-large-50-many-to-many-mmt"])

            # File upload
            uploaded_file = st.file_uploader("Upload a WAV audio file", type=['wav'])
            st.markdown("---")

            # Select from local corpus
            col1, _ = st.columns(2)
            with col1:
                st.subheader("Or select from local corpus")
                local_audio_path = "local_corpus/audio"
                audio_files = []
                if os.path.exists(local_audio_path):
                    audio_files = [f for f in os.listdir(local_audio_path) if f.endswith('.wav')]
                selected_audio = st.selectbox("Select an audio file from corpus", ["None"] + audio_files)

            # Buttons
            col1, col2, _ = st.columns([1, 1, 2])
            with col1:
                transcribe_button = st.button("🎙️ Transcribe Audio", type="primary", use_container_width=True)
            with col2:
                translate_button = st.button("🌐 Translate", type="secondary", use_container_width=True, 
                                           disabled=not st.session_state.get('transcription_done', False))

            # Initialize session state
            if 'transcription_done' not in st.session_state:
                st.session_state.transcription_done = False
                st.session_state.transcription = ""
                st.session_state.translation = ""
                st.session_state.audio_for_playback = None # Path to the processed audio file

            # Process transcription
            if transcribe_button:
                st.session_state.translation = "" # Clear previous
                audio_to_process = None

                if uploaded_file is not None:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_to_process = temp_path
                elif selected_audio != "None":
                    audio_to_process = os.path.join(local_audio_path, selected_audio)

                if audio_to_process:
                    # ONLY CHANGE 1: Save the path for the audio player
                    st.session_state.audio_for_playback = audio_to_process
                    
                    with st.spinner("Transcribing audio..."):
                        @st.cache_resource
                        def load_whisper(model_size): return whisper.load_model(model_size)
                        
                        whisper_model = load_whisper(whisper_model_size)
                        result = whisper_model.transcribe(audio_to_process, language="es")
                        st.session_state.transcription = result["text"]
                        st.session_state.transcription_done = True
                        st.rerun()
                else:
                    st.warning("Please upload or select an audio file.")

            # Process translation
            if translate_button and st.session_state.transcription_done:
                with st.spinner("Translating..."):
                    @st.cache_resource
                    def load_translator(model_name):
                        if "mbart" in model_name:
                            return pipeline("translation", model=model_name, src_lang="es_XX", tgt_lang="en_XX")
                        return pipeline("translation", model=model_name)
                    
                    translator = load_translator(translation_model)
                    translation_result = translator(st.session_state.transcription)
                    st.session_state.translation = translation_result[0]['translation_text']
                    st.rerun()

            # Display results if transcription is done
            if st.session_state.transcription_done:
                st.markdown("---")
                st.subheader("📊 Results")

                # ONLY CHANGE 2: Add the st.audio widget here
                if st.session_state.audio_for_playback and os.path.exists(st.session_state.audio_for_playback):
                    with open(st.session_state.audio_for_playback, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav')

                # Display columns with generated text (your original logic)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Spanish Transcription")
                    st.markdown("**Generated:**")
                    st.info(st.session_state.transcription)
                
                with col2:
                    st.markdown("### English Translation")
                    st.markdown("**Generated:**")
                    if st.session_state.translation:
                        st.info(st.session_state.translation)
                    else:
                        st.info("Click 'Translate' to generate.")

      # ==========================
    # VIEW 1: Model Metrics Overview
    # ==========================
    with tab2:
        st.header("Model Performance Metrics")
        
        # Language filter for View 1
        col1, col2 = st.columns([1, 3])
        with col1:
            languages_available = metrics_df['Language'].unique()
            selected_language_metrics = st.selectbox(
                "Select Language",
                options=['All'] + list(languages_available),
                key='lang_metrics'
            )
        
        # Filter data based on selection
        if selected_language_metrics == 'All':
            filtered_metrics = metrics_df
        else:
            filtered_metrics = metrics_df[metrics_df['Language'] == selected_language_metrics]
        
        # Display key metrics
        st.subheader("📈 Key Performance Indicators")
        
        # Create metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_wer = filtered_metrics['avg_wer'].mean()
            st.metric("Average WER", f"{avg_wer:.3f}")
        
        with col2:
            avg_bleu = filtered_metrics['avg_bleu'].mean()
            st.metric("Average BLEU", f"{avg_bleu:.3f}")
        
        with col3:
            avg_time = filtered_metrics['avg_time_s'].mean()
            st.metric("Average Time (s)", f"{avg_time:.2f}")
        
        # Create comparison charts
        st.subheader("📊 Model Comparisons")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # WER comparison chart
            fig_wer = px.bar(
                filtered_metrics,
                x='model',
                y='avg_wer',
                color='Language',
                title='WER by Model (Lower is Better)',
                labels={'avg_wer': 'Average WER', 'model': 'Model'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_wer.update_layout(height=400)
            st.plotly_chart(fig_wer, use_container_width=True)
        
        with col2:
            # BLEU comparison chart
            fig_bleu = px.bar(
                filtered_metrics,
                x='model',
                y='avg_bleu',
                color='Language',
                title='BLEU Score by Model (Higher is Better)',
                labels={'avg_bleu': 'Average BLEU', 'model': 'Model'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_bleu.update_layout(height=400)
            st.plotly_chart(fig_bleu, use_container_width=True)
        
        # Time performance chart
        fig_time = px.scatter(
            filtered_metrics,
            x='avg_time_s',
            y='avg_bleu',
            size='avg_wer',
            color='model',
            hover_data=['Language'],
            title='Performance vs Speed Trade-off',
            labels={'avg_time_s': 'Average Time (seconds)', 'avg_bleu': 'BLEU Score'},
            size_max=30
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics Table")
        st.dataframe(
            filtered_metrics.style.format({
                'avg_wer': '{:.3f}',
                'avg_bleu': '{:.3f}',
                'avg_time_s': '{:.2f}'
            }),
            use_container_width=True
        )
    
    # ==========================
    # VIEW 2: Translation Comparison
    # ==========================
    with tab3:
        st.header("Translation Comparison")
        
        # Filters for View 2
        col1, col2 = st.columns(2)
        
        with col1:
            # Language filter
            languages_trans = translations_df['Language'].unique()
            selected_language_trans = st.selectbox(
                "Select Language",
                options=languages_trans,
                key='lang_trans'
            )
        
        # Filter translations by language
        filtered_translations = translations_df[translations_df['Language'] == selected_language_trans]
        
        with col2:
            # Original text filter
            original_texts = filtered_translations['original text'].unique()
            selected_text = st.selectbox(
                "Select Original Text",
                options=original_texts,
                key='orig_text'
            )
        
        # Filter by selected original text
        text_translations = filtered_translations[filtered_translations['original text'] == selected_text]
        
        if not text_translations.empty:
            st.subheader("📝 Translation Results")
            
            # Display original text
            st.info(f"**Original Text:** {selected_text}")
            
            # Get unique models for this text
            models = text_translations['model'].unique()
            
            # Create columns for side-by-side comparison
            cols = st.columns(len(models) + 1)  # +1 for expected translation
            
            # Display expected translation first
            with cols[0]:
                st.markdown("**Expected Translation**")
                expected = text_translations.iloc[0]['expected']
                st.success(expected)
            
            # Display translations from each model
            for idx, model in enumerate(models, 1):
                with cols[idx]:
                    model_data = text_translations[text_translations['model'] == model].iloc[0]
                    st.markdown(f"**{model}**")
                    st.info(model_data['translated'])
                    
                    # Display metrics for this translation
                    st.caption(f"WER: {model_data['wer']:.3f}")
                    st.caption(f"BLEU: {model_data['bleu']:.3f}")
                    st.caption(f"Time: {model_data['time']:.2f}s")
            
            # Metrics comparison chart for selected text
            st.subheader("📊 Metrics Comparison for Selected Text")
            
            # Prepare data for comparison
            comparison_data = text_translations[['model', 'wer', 'bleu', 'time']].copy()
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('WER (Lower is Better)', 'BLEU (Higher is Better)', 'Translation Time (s)')
            )
            
            # WER chart
            fig.add_trace(
                go.Bar(x=comparison_data['model'], y=comparison_data['wer'], 
                       name='WER', marker_color='coral'),
                row=1, col=1
            )
            
            # BLEU chart
            fig.add_trace(
                go.Bar(x=comparison_data['model'], y=comparison_data['bleu'], 
                       name='BLEU', marker_color='lightblue'),
                row=1, col=2
            )
            
            # Time chart
            fig.add_trace(
                go.Bar(x=comparison_data['model'], y=comparison_data['time'], 
                       name='Time', marker_color='lightgreen'),
                row=1, col=3
            )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed comparison table
            st.subheader("📋 Detailed Comparison")
            comparison_table = text_translations[['model', 'translated', 'expected', 'wer', 'bleu', 'time']].copy()
            comparison_table = comparison_table.rename(columns={
                'translated': 'Model Translation',
                'expected': 'Expected Translation',
                'wer': 'WER Score',
                'bleu': 'BLEU Score',
                'time': 'Time (s)'
            })
            
            st.dataframe(
                comparison_table.style.format({
                    'WER Score': '{:.3f}',
                    'BLEU Score': '{:.3f}',
                    'Time (s)': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Best performing model for this text
            best_bleu_model = text_translations.loc[text_translations['bleu'].idxmax(), 'model']
            best_wer_model = text_translations.loc[text_translations['wer'].idxmin(), 'model']
            fastest_model = text_translations.loc[text_translations['time'].idxmin(), 'model']
            
            st.subheader("🏆 Best Performers for This Text")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best BLEU Score", best_bleu_model)
            with col2:
                st.metric("Best WER Score", best_wer_model)
            with col3:
                st.metric("Fastest Translation", fastest_model)

 # ==============================================================
    # VIEW 4: Low-Resource Fine-Tuning Analysis (from Excel files)
    # ==============================================================
    with tab4:
        st.header("🔬 Low-Resource Fine-Tuning Analysis")

        language_names = list(FINETUNING_METRICS.keys())
        
        selected_language_name = st.selectbox(
            "Select a Fine-Tuned Language",
            options=language_names,
            key='lang_ft_select'
        )

        # Get the hardcoded metrics for the selected language
        metrics_data = FINETUNING_METRICS[selected_language_name]
        
        # Filter the loaded examples dataframe for the selected language
        examples_data = finetuning_examples_df[finetuning_examples_df['Language'] == selected_language_name]

        st.subheader(f"Performance for {selected_language_name}")

        # --- High-Level Metrics (from hardcoded data) ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Hours", f"{metrics_data['hours_train']:.2f} hrs")
        with col2:
            st.metric("Base Model WER", f"{metrics_data['base_wer']:.2f}%")
        with col3:
            st.metric("Fine-Tuned WER", f"{metrics_data['ft_wer']:.2f}%", 
                      delta=f"{metrics_data['base_wer'] - metrics_data['ft_wer']:.2f}% improvement",
                      delta_color="inverse")

        # --- WER Improvement Visualization (from hardcoded data) ---
        st.subheader("WER Improvement")
        fig_wer_ft = go.Figure(data=[
            go.Bar(name='Base Model', x=[selected_language_name], y=[metrics_data['base_wer']], text=f"{metrics_data['base_wer']:.2f}%", textposition='auto'),
            go.Bar(name='Fine-Tuned Model', x=[selected_language_name], y=[metrics_data['ft_wer']], text=f"{metrics_data['ft_wer']:.2f}%", textposition='auto')
        ])
        fig_wer_ft.update_layout(
            barmode='group',
            title=f'Word Error Rate (WER) Comparison for {selected_language_name}',
            yaxis_title='WER (%) - Lower is Better',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_wer_ft, use_container_width=True)

        # --- Transcription Examples Table (from loaded Excel file) ---
        st.subheader("Transcription Examples")
        
        st.dataframe(
            examples_data.drop(columns=['Language']),
            use_container_width=True,
            height=210
        )

# --- Sidebar ---
with st.sidebar:
    st.header("About")
    st.info("""
    This dashboard analyzes ASR and MT model performance.
    
    **Metrics Explained:**
    - **WER (Word Error Rate):** Lower is better
    - **BLEU Score:** Higher is better (0-1 scale)
    - **Time:** Inference time in seconds
    """)
    if metrics_df is not None:
        st.header("Dataset Summary")
        st.metric("Total Models Tested", metrics_df['model'].nunique())
        st.metric("Languages Tested", metrics_df['Language'].nunique())