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

# Load data function
@st.cache_data
def load_data():
    """Load both Excel files"""
    try:
        # Load metrics data (first Excel)
        metrics_df = pd.read_excel('metrics.xlsx')
        # Load translations data (second Excel)
        translations_df = pd.read_excel('translations.xlsx')
        return metrics_df, translations_df
    except FileNotFoundError:
        st.error("Please ensure both 'metrics.xlsx' and 'translations.xlsx' are in the same directory as this script.")
        return None, None

# Load the data
metrics_df, translations_df = load_data()

if metrics_df is not None and translations_df is not None:
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Audio Transcription & Translation Inference","Translation Performance Overview", "Translation Models Inference"])

    

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
        
        # Force module loading with explicit path
        import sys
        sys.path.insert(0, '/opt/conda/lib/python3.12/site-packages')
        
        # Try to load train.csv for expected values
        @st.cache_data
        def load_train_data():
            """Load train.csv for expected transcriptions and translations"""
            try:
                train_df = pd.read_csv('train.csv')
                # Check if it has the expected columns
                if 'path' in train_df.columns and 'text' in train_df.columns and 'text_en' in train_df.columns:
                    return train_df
                else:
                    st.warning("train.csv doesn't have expected columns (path, text, text_en)")
                    return None
            except FileNotFoundError:
                st.warning("train.csv not found. Expected values won't be available.")
                return None
        
        train_df = load_train_data()
        
        # Test if Whisper is available
        whisper_available = False
        try:
            import whisper
            whisper_available = True
            st.success("✅ Whisper loaded successfully")
        except ImportError as e:
            st.error(f"❌ Whisper import error: {e}")
        
        # Test if transformers is available
        transformers_available = False
        try:
            from transformers import pipeline
            import torch
            transformers_available = True
            st.success("✅ Transformers loaded successfully")
        except ImportError as e:
            st.error(f"❌ Transformers import error: {e}")
        
        if whisper_available and transformers_available:
            import whisper
            from transformers import pipeline
            import torch
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                whisper_model_size = st.selectbox(
                    "Select Whisper Model Size",
                    options=["base", "small", "medium", "large"],
                    help="Larger models are more accurate but slower"
                )
            
            with col2:
                translation_model = st.selectbox(
                    "Select Translation Model",
                    options=["opus-mt-es-en", "mbart-large-50-many-to-many-mmt"],
                    help="Model for Spanish to English translation"
                )
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload a WAV audio file",
                type=['wav'],
                help="Upload a Spanish audio file for transcription and translation"
            )
            
            # Alternative: Select from local_corpus
            st.markdown("---")
            
            # MODIFICATION: Place selection in a smaller column
            col1, _ = st.columns(2)
            with col1:
                st.subheader("Or select from local corpus")
                # Get the directory where app.py is located
                #base_dir = os.path.dirname(os.path.abspath(__file__))
                # local_audio_path = os.path.join(base_dir, "local_corpus", "audio")
                import os
                # Get the directory where app.py is located
                base_dir = os.path.dirname(os.path.abspath(__file__))
                local_audio_path = os.path.join(base_dir, "local_corpus", "audio")

                audio_files = []
                
                if os.path.exists(local_audio_path):
                    audio_files = [f for f in os.listdir(local_audio_path) if f.endswith('.wav')]
                    if audio_files:
                        selected_audio = st.selectbox(
                            "Select an audio file from corpus",
                            options=["None"] + audio_files
                        )
                    else:
                        st.info("No WAV files found in local_corpus/audio/")
                        selected_audio = "None"
                else:
                    st.info("local_corpus/audio/ directory not found")
                    selected_audio = "None"
            
            # Process buttons - sized to be smaller
            col1, col2, _ = st.columns([1, 1, 2])
            
            with col1:
                transcribe_button = st.button("🎙️ Transcribe Audio", type="primary", use_container_width=True)
            
            with col2:
                translate_button = st.button("🌐 Translate", type="secondary", use_container_width=True, 
                                           disabled=not st.session_state.get('transcription_done', False))
            
            # Initialize session state for transcription
            if 'transcription_done' not in st.session_state:
                st.session_state.transcription_done = False
                st.session_state.transcription = ""
                st.session_state.translation = ""
                st.session_state.audio_path = ""
            
            # Process transcription
            if transcribe_button:
                st.session_state.translation = "" # Clear previous translation
                
                audio_to_process = None
                audio_path = None
                
                # Determine which audio to process
                if uploaded_file is not None:
                    # Save uploaded file temporarily
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    audio_to_process = temp_path
                    audio_path = uploaded_file.name
                elif selected_audio != "None":
                    audio_to_process = os.path.join(local_audio_path, selected_audio)
                    audio_path = f"local_corpus/audio/{selected_audio}"
                
                if audio_to_process:
                    with st.spinner("Transcribing audio with Whisper..."):
                        # Load Whisper model
                        @st.cache_resource
                        def load_whisper(model_size):
                            return whisper.load_model(model_size)
                        
                        try:
                            whisper_model = load_whisper(whisper_model_size)
                            result = whisper_model.transcribe(audio_to_process, language="es")
                            st.session_state.transcription = result["text"]
                            st.session_state.transcription_done = True
                            st.session_state.audio_path = audio_path
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error transcribing audio: {str(e)}")
                            
                        finally:
                            # Clean up temp file if it was uploaded
                            if uploaded_file is not None and os.path.exists(temp_path):
                                os.remove(temp_path)
                else:
                    st.warning("Please upload an audio file or select one from the corpus.")
            
            # Process translation
            if translate_button and st.session_state.transcription_done:
                with st.spinner("Translating to English..."):
                    # Load translation model
                    @st.cache_resource
                    def load_translator(model_name):
                        if model_name == "opus-mt-es-en":
                            return pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
                        else:
                            return pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt", 
                                          src_lang="es_XX", tgt_lang="en_XX")
                    
                    try:
                        translator = load_translator(translation_model)
                        translation_result = translator(st.session_state.transcription, max_length=512)
                        st.session_state.translation = translation_result[0]['translation_text']
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error translating: {str(e)}")
            
            # Display results if we have them
            if st.session_state.transcription_done:
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Get expected values if available
                expected_text = ""
                expected_translation = ""
                
                if train_df is not None and st.session_state.audio_path:
                    matching_row = train_df[train_df['path'].str.contains(
                        os.path.basename(st.session_state.audio_path).replace('.wav', ''), 
                        case=False, na=False
                    )]
                    
                    if not matching_row.empty:
                        expected_text = matching_row.iloc[0]['text']
                        expected_translation = matching_row.iloc[0]['text_en']
                
                # Display transcription and translation with expected values
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Spanish Transcription")
                    
                    # Predicted transcription
                    st.markdown("**Generated:**")
                    st.info(st.session_state.transcription)
                    
                    # Expected transcription (if available)
                    if expected_text:
                        st.markdown("**Expected:**")
                        st.success(expected_text)
                        
                        # Calculate similarity
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, 
                                                    st.session_state.transcription.lower(), 
                                                    expected_text.lower()).ratio()
                        st.metric("Similarity", f"{similarity:.2%}")
                
            # CORRECTED CODE BLOCK

            with col2:
                st.markdown("###  English Translation")
                
                if st.session_state.translation:
                    # Predicted translation
                    st.markdown("**Generated:**")
                    st.info(st.session_state.translation)
                    
                    # Expected translation (if available)
                    if expected_translation:
                        st.markdown("**Expected:**")
                        # FIX: Use the correct variable that holds the English text
                        st.success(expected_translation)
                        
                        # Calculate similarity
                        from difflib import SequenceMatcher
                        similarity = SequenceMatcher(None, 
                                                    st.session_state.translation.lower(), 
                                                    expected_translation.lower()).ratio()
                        st.metric("Similarity", f"{similarity:.2%}")
                else:
                    st.markdown("**Generated:**")
                    st.info("Click 'Translate' button to generate translation.")
                        
        else:
            st.warning("⚠️ Waiting for required libraries to be installed...")
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


           
else:
    st.error("Unable to load data files. Please check that both Excel files are present.")

# Add sidebar information
with st.sidebar:
    st.header("About")
    st.info("""
    This dashboard analyzes translation model performance across different languages.
    
    **Metrics Explained:**
    - **WER (Word Error Rate):** Lower is better
    - **BLEU Score:** Higher is better (0-1 scale)
    - **Time:** Translation time in seconds
    
    **Data Sources:**
    - metrics.xlsx: Overall model performance
    - translations.xlsx: Individual translation examples
    """)
    
    if metrics_df is not None:
        st.header("Dataset Summary")
        st.metric("Total Models", metrics_df['model'].nunique())
        st.metric("Languages", metrics_df['Language'].nunique())
        st.metric("Test Samples per Model", metrics_df['samples_tested'].iloc[0])

