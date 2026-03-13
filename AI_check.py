"""
Audio Quality Checker using YAMNet
A Streamlit dashboard for recording and analyzing audio quality for specific tasks.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import io
import tempfile
import os
from st_audiorec import st_audiorec
import resampy

# ============================================
# CONSTANTS - YAMNet Class Mappings
# ============================================

TASK_CLASSES = {
    "Breathing": ["Breathing", "Wheeze", "Gasp", "Pant", "Snort", "Snoring"],
    "Cough": ["Cough", "Throat clearing", "Sneeze"],
    "Vowel": ["Speech", "Singing", "Humming", "Mantra", "Chant"],
    "Speech": ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Whispering"]
}

NOISE_CLASSES = [
    "Noise", "Environmental noise", "Static", "Hum", "Buzz",
    "Reverberation", "Echo", "White noise", "Pink noise"
]

SILENCE_CLASS = "Silence"

# YAMNet expects 16kHz sample rate
YAMNET_SAMPLE_RATE = 16000

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_yamnet_model():
    """Load YAMNet model from TensorFlow Hub."""
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    return model

@st.cache_resource
def load_class_names():
    """Load YAMNet class names."""
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    with open(class_map_path, 'r') as f:
        lines = f.readlines()

    class_names = {}
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) >= 3:
            index = int(parts[0])
            display_name = parts[2].strip('"')
            class_names[index] = display_name

    return class_names

# ============================================
# AUDIO ANALYSIS FUNCTIONS
# ============================================

def preprocess_audio(audio_bytes):
    """
    Convert recorded audio bytes to WAV format suitable for YAMNet.
    Returns: (waveform, sample_rate) - mono audio at 16kHz
    """
    # Read audio from bytes
    audio_file = io.BytesIO(audio_bytes)

    try:
        waveform, sr = sf.read(audio_file)
    except Exception as e:
        # Try saving to temp file first (some formats need this)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        waveform, sr = sf.read(tmp_path)
        os.unlink(tmp_path)

    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample to 16kHz if needed
    if sr != YAMNET_SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, YAMNET_SAMPLE_RATE)

    # Normalize
    waveform = waveform.astype(np.float32)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    return waveform, YAMNET_SAMPLE_RATE

def analyze_audio_with_yamnet(waveform, model, class_names):
    """
    Run YAMNet inference on the audio waveform.
    Returns: Dictionary with scores for each class
    """
    # Run model
    scores, embeddings, spectrogram = model(waveform)

    # Average scores across all frames
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Create results dictionary
    results = {}
    for idx, score in enumerate(mean_scores):
        if idx in class_names:
            results[class_names[idx]] = float(score)

    return results, scores.numpy()

def calculate_silence_percentage(waveform, threshold=0.02):
    """
    Calculate the percentage of silence in the audio.
    Uses RMS energy to detect silent frames.
    """
    frame_length = int(YAMNET_SAMPLE_RATE * 0.025)  # 25ms frames
    hop_length = int(YAMNET_SAMPLE_RATE * 0.010)    # 10ms hop

    silent_frames = 0
    total_frames = 0

    for i in range(0, len(waveform) - frame_length, hop_length):
        frame = waveform[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))

        if rms < threshold:
            silent_frames += 1
        total_frames += 1

    if total_frames == 0:
        return 0.0

    return (silent_frames / total_frames) * 100

def calculate_noise_level(results):
    """
    Calculate background noise level from YAMNet results.
    Returns: (level_string, score)
    """
    noise_scores = []
    for noise_class in NOISE_CLASSES:
        if noise_class in results:
            noise_scores.append(results[noise_class])

    if not noise_scores:
        return "Unknown", 0.0

    max_noise_score = max(noise_scores)
    avg_noise_score = np.mean(noise_scores)

    # Determine noise level
    if max_noise_score > 0.5:
        return "High", max_noise_score
    elif max_noise_score > 0.2:
        return "Medium", max_noise_score
    elif max_noise_score > 0.05:
        return "Low", max_noise_score
    else:
        return "Minimal", max_noise_score

def check_task_audio(results, task, frame_scores, class_names):
    """
    Check if the audio contains the expected sound for the task.
    Returns: (detected: bool, confidence: float, detected_classes: list)
    """
    task_class_names = TASK_CLASSES.get(task, [])

    detected_classes = []
    max_confidence = 0.0

    for class_name in task_class_names:
        if class_name in results:
            score = results[class_name]
            if score > 0.1:  # Detection threshold
                detected_classes.append((class_name, score))
            if score > max_confidence:
                max_confidence = score

    # Sort by confidence
    detected_classes.sort(key=lambda x: x[1], reverse=True)

    # Consider detected if any relevant class has score > 0.15
    detected = max_confidence > 0.15

    return detected, max_confidence, detected_classes

def get_top_predictions(results, top_k=10):
    """Get top K predictions from results."""
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.set_page_config(
        page_title="Audio Quality Checker",
        page_icon=":microphone:",
        layout="wide"
    )

    st.title("Audio Quality Checker")
    st.markdown("Record audio and analyze its quality using YAMNet AI model")

    # Load model
    with st.spinner("Loading YAMNet model..."):
        model = load_yamnet_model()
        class_names = load_class_names()

    st.success("Model loaded successfully!")

    # Sidebar - Task Selection
    st.sidebar.header("Task Selection")
    selected_task = st.sidebar.selectbox(
        "Select the audio task:",
        options=list(TASK_CLASSES.keys()),
        help="Choose what type of audio you're recording"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Expected sounds for {selected_task}:**")
    for cls in TASK_CLASSES[selected_task]:
        st.sidebar.markdown(f"- {cls}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Record Audio")
        st.markdown(f"**Current Task:** {selected_task}")
        st.markdown("Click to start/stop recording (waveform shown while recording):")

        # Audio recorder with waveform visualization
        audio_bytes = st_audiorec()

        # Playback recorded audio
        if audio_bytes:
            st.markdown("---")
            st.markdown("**Playback recorded audio:**")
            st.audio(audio_bytes, format="audio/wav")

        # Also allow file upload
        st.markdown("---")
        st.markdown("**Or upload an audio file:**")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an existing audio file for analysis"
        )

        # Playback uploaded audio
        if uploaded_file:
            st.markdown("**Playback uploaded audio:**")
            st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")

    with col2:
        st.header("Analysis Results")

        # Determine which audio to analyze
        audio_to_analyze = None
        if audio_bytes:
            audio_to_analyze = audio_bytes
            st.info("Analyzing recorded audio...")
        elif uploaded_file:
            audio_to_analyze = uploaded_file.read()
            st.info("Analyzing uploaded audio...")

        if audio_to_analyze:
            try:
                # Preprocess audio
                waveform, sr = preprocess_audio(audio_to_analyze)

                # Check if audio is too short
                duration = len(waveform) / sr
                if duration < 0.5:
                    st.warning("Audio is too short (< 0.5 seconds). Please record a longer sample.")
                else:
                    # Run analysis
                    with st.spinner("Analyzing audio..."):
                        results, frame_scores = analyze_audio_with_yamnet(waveform, model, class_names)
                        silence_pct = calculate_silence_percentage(waveform)
                        noise_level, noise_score = calculate_noise_level(results)
                        task_detected, task_confidence, detected_classes = check_task_audio(
                            results, selected_task, frame_scores, class_names
                        )

                    # Display results
                    st.markdown("---")

                    # Audio Info
                    st.subheader("Audio Information")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.metric("Duration", f"{duration:.2f} sec")
                    with info_col2:
                        st.metric("Sample Rate", f"{sr} Hz")

                    st.markdown("---")

                    # Task Detection
                    st.subheader(f"Task Detection: {selected_task}")
                    if task_detected:
                        st.success(f"{selected_task} sound DETECTED!")
                        st.metric("Confidence", f"{task_confidence:.1%}")

                        if detected_classes:
                            st.markdown("**Detected sounds:**")
                            for cls, score in detected_classes[:5]:
                                st.markdown(f"- {cls}: {score:.1%}")
                    else:
                        st.error(f"{selected_task} sound NOT detected")
                        st.markdown("The audio doesn't seem to contain the expected sound.")
                        st.markdown("**Tip:** Make sure you're recording the correct audio type.")

                    st.markdown("---")

                    # Noise Level
                    st.subheader("Background Noise")
                    noise_indicators = {
                        "Minimal": "[OK]",
                        "Low": "[LOW]",
                        "Medium": "[MED]",
                        "High": "[HIGH]",
                        "Unknown": "[?]"
                    }
                    st.markdown(f"**Noise Level:** {noise_indicators.get(noise_level, '[?]')} {noise_level}")
                    st.progress(min(noise_score, 1.0))

                    if noise_level in ["High", "Medium"]:
                        st.warning("Consider recording in a quieter environment")

                    st.markdown("---")

                    # Silence Check
                    st.subheader("Silence Analysis")
                    st.metric("Silence Percentage", f"{silence_pct:.1f}%")
                    st.progress(silence_pct / 100)

                    if silence_pct > 70:
                        st.error("Too much silence in the recording!")
                    elif silence_pct > 50:
                        st.warning("Recording has significant silence")
                    else:
                        st.success("Good audio activity level")

                    st.markdown("---")

                    # Top Predictions
                    st.subheader("Top Detected Sounds")
                    top_preds = get_top_predictions(results, top_k=10)

                    for class_name, score in top_preds:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(score)
                        with col_b:
                            st.markdown(f"**{class_name}**")
                            st.caption(f"{score:.1%}")

                    st.markdown("---")

                    # Overall Assessment
                    st.subheader("Overall Assessment")

                    issues = []
                    if not task_detected:
                        issues.append(f"{selected_task} not detected in audio")
                    if noise_level in ["High", "Medium"]:
                        issues.append(f"{noise_level} background noise")
                    if silence_pct > 50:
                        issues.append(f"High silence ({silence_pct:.1f}%)")

                    if not issues:
                        st.success("Audio quality is GOOD for the selected task!")
                    else:
                        st.warning("Issues found:")
                        for issue in issues:
                            st.markdown(f"- {issue}")

            except Exception as e:
                st.error(f"Error analyzing audio: {str(e)}")
                st.exception(e)
        else:
            st.info("Record or upload audio to see analysis results")

if __name__ == "__main__":
    main()
