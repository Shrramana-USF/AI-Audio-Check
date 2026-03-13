"""
Audio Quality Checker
Record and analyze audio using YAMNet to verify quality for specific tasks.
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
import plotly.express as px

# What sounds we expect for each task
TASK_CLASSES = {
    "Breathing": ["Breathing", "Wheeze", "Gasp", "Pant", "Snort", "Snoring"],
    "Cough": ["Cough", "Throat clearing", "Sneeze"],
    "Vowel": ["Speech", "Singing", "Humming", "Mantra", "Chant"],
    "Speech": ["Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue", "Whispering"]
}

# Sounds that indicate background noise
NOISE_CLASSES = [
    "Noise", "Environmental noise", "Static", "Hum", "Buzz",
    "Reverberation", "Echo", "White noise", "Pink noise"
]

YAMNET_SAMPLE_RATE = 16000


@st.cache_resource
def load_yamnet_model():
    """Load YAMNet from TensorFlow Hub."""
    return hub.load('https://tfhub.dev/google/yamnet/1')


@st.cache_resource
def load_class_names():
    """Fetch the list of sound classes YAMNet can recognize."""
    class_map_path = tf.keras.utils.get_file(
        'yamnet_class_map.csv',
        'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    )
    with open(class_map_path, 'r') as f:
        lines = f.readlines()

    class_names = {}
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 3:
            index = int(parts[0])
            display_name = parts[2].strip('"')
            class_names[index] = display_name

    return class_names


def preprocess_audio(audio_bytes):
    """Convert audio to mono 16kHz format for YAMNet."""
    audio_file = io.BytesIO(audio_bytes)

    try:
        waveform, sr = sf.read(audio_file)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        waveform, sr = sf.read(tmp_path)
        os.unlink(tmp_path)

    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != YAMNET_SAMPLE_RATE:
        waveform = resampy.resample(waveform, sr, YAMNET_SAMPLE_RATE)

    waveform = waveform.astype(np.float32)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))

    return waveform, YAMNET_SAMPLE_RATE


def analyze_audio_with_yamnet(waveform, model, class_names):
    """Run YAMNet and get confidence scores for each sound class."""
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)

    results = {}
    for idx, score in enumerate(mean_scores):
        if idx in class_names:
            results[class_names[idx]] = float(score)

    return results, scores.numpy()


def calculate_silence_percentage(waveform, threshold=0.02):
    """Check how much of the audio is silent."""
    frame_length = int(YAMNET_SAMPLE_RATE * 0.025)
    hop_length = int(YAMNET_SAMPLE_RATE * 0.010)

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
    """Determine how noisy the recording is."""
    noise_scores = []
    for noise_class in NOISE_CLASSES:
        if noise_class in results:
            noise_scores.append(results[noise_class])

    if not noise_scores:
        return "Unknown", 0.0

    max_noise_score = max(noise_scores)

    if max_noise_score > 0.5:
        return "High", max_noise_score
    elif max_noise_score > 0.2:
        return "Medium", max_noise_score
    elif max_noise_score > 0.05:
        return "Low", max_noise_score
    else:
        return "Minimal", max_noise_score


def check_task_audio(results, task):
    """Check if the expected sound for the task was detected."""
    task_class_names = TASK_CLASSES.get(task, [])

    detected_classes = []
    max_confidence = 0.0

    for class_name in task_class_names:
        if class_name in results:
            score = results[class_name]
            if score > 0.1:
                detected_classes.append((class_name, score))
            if score > max_confidence:
                max_confidence = score

    detected_classes.sort(key=lambda x: x[1], reverse=True)
    detected = max_confidence > 0.15

    return detected, max_confidence, detected_classes


def get_top_predictions(results, top_k=10):
    """Get the most confident sound predictions."""
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def main():
    st.set_page_config(
        page_title="Audio Quality Checker",
        page_icon=":microphone:",
        layout="wide"
    )

    st.title("Audio Quality Checker")
    st.markdown("Record audio and analyze its quality using YAMNet AI model")

    with st.spinner("Loading YAMNet model..."):
        model = load_yamnet_model()
        class_names = load_class_names()

    st.success("Model loaded successfully!")

    # Sidebar for task selection
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

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Record Audio")
        st.markdown(f"**Current Task:** {selected_task}")
        st.markdown("Click to start/stop recording:")

        audio_bytes = st_audiorec()

        st.markdown("---")
        st.markdown("**Or upload an audio file:**")
        uploaded_file = st.file_uploader(
            "Choose a WAV file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an existing audio file for analysis"
        )

        if uploaded_file:
            st.markdown("**Playback:**")
            st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")

    with col2:
        st.header("Analysis Results")

        audio_to_analyze = None
        if audio_bytes:
            audio_to_analyze = audio_bytes
            st.info("Analyzing recorded audio...")
        elif uploaded_file:
            audio_to_analyze = uploaded_file.read()
            st.info("Analyzing uploaded audio...")

        if audio_to_analyze:
            try:
                waveform, sr = preprocess_audio(audio_to_analyze)
                duration = len(waveform) / sr

                if duration < 0.5:
                    st.warning("Audio is too short. Please record at least 0.5 seconds.")
                else:
                    with st.spinner("Analyzing audio..."):
                        results, frame_scores = analyze_audio_with_yamnet(waveform, model, class_names)
                        silence_pct = calculate_silence_percentage(waveform)
                        noise_level, noise_score = calculate_noise_level(results)
                        task_detected, task_confidence, detected_classes = check_task_audio(results, selected_task)

                    st.markdown("---")

                    # Basic info
                    st.subheader("Audio Information")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.metric("Duration", f"{duration:.2f} sec")
                    with info_col2:
                        st.metric("Sample Rate", f"{sr} Hz")

                    st.markdown("---")

                    # Did we detect the expected sound?
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
                        st.markdown("The audio doesn't contain the expected sound.")
                        st.markdown("**Tip:** Make sure you're recording the correct audio type.")

                    st.markdown("---")

                    # How noisy is it?
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

                    # How much silence?
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

                    # What sounds were detected?
                    st.subheader("Top Detected Sounds")
                    top_preds = get_top_predictions(results, top_k=10)

                    labels = [pred[0] for pred in top_preds]
                    values = [pred[1] for pred in top_preds]

                    fig = px.pie(
                        names=labels,
                        values=values,
                        title="Audio Class Distribution"
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch')

                    for class_name, score in top_preds:
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.progress(score)
                        with col_b:
                            st.markdown(f"**{class_name}**")
                            st.caption(f"{score:.1%}")

                    st.markdown("---")

                    # Final verdict
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
