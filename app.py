import streamlit as st
import whisper
import soundfile as sf
import io
import numpy as np
import gc


# Set up the Streamlit app with custom theme
st.set_page_config(
    page_title="Audio Transcription App",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff7e5f, #feb47b);
        color: white;
        border-radius: 25px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown h1 {
        color: #ff4757;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Audio Transcription App")
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.9); padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
    Upload an audio file to get its transcription
</div>
""", unsafe_allow_html=True)


# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")


model = load_model()

# Function to process audio in chunks
def process_audio_chunks(audio, samplerate):
    # Convert to mono if stereo and resample if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if needed (Whisper's optimal sample rate)
    if samplerate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)
        samplerate = 16000

    
    # Validate audio quality
    if np.max(np.abs(audio)) < 0.01:  # Check if audio is too quiet
        raise ValueError("Audio signal is too weak. Please use a louder recording.")
    
    chunk_size = samplerate * 30  # 30 seconds per chunk
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
    transcriptions = []
    
    for chunk in chunks:
        if len(chunk) > 0:
            # Convert to float32 and normalize
            chunk = chunk.astype(np.float32)
            chunk /= np.max(np.abs(chunk))
            
            # Add initial prompt for better context
            result = model.transcribe(chunk, initial_prompt="Transcribe the following audio clearly and accurately.")
            # Clean and encode text, remove extra spaces
            text = " ".join(result["text"].encode('utf-8', errors='ignore').decode('utf-8').split())

            transcriptions.append(text)
            
            # Clean up memory
            del chunk
            gc.collect()
    
    return " ".join(transcriptions)




# File uploader with colorful styling
audio_file = st.file_uploader(
    "🎧 Upload Audio", 
    type=["wav", "mp3", "ogg"],
    help="Supported formats: WAV, MP3, OGG"
)


if audio_file is not None:
    # Read audio file
    audio_bytes = audio_file.read()
    try:
        audio, samplerate = sf.read(io.BytesIO(audio_bytes))
        
        # Check file size
        if len(audio) > samplerate * 300:  # 5 minutes max
            st.error("File too large. Please upload audio shorter than 5 minutes.")

        else:
            # Transcribe audio in chunks
            with st.spinner("Transcribing audio..."):
                transcription = process_audio_chunks(audio, samplerate)


    
    except ValueError as e:
        st.error(f"Audio quality issue: {str(e)}")
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}. Please ensure the audio file is clear and not corrupted.")

    else:
        # Display results with colorful styling
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #ff4757;">🎤 Transcription</h3>
            <p style="color: #2f3542;">{transcription}</p>
        </div>
        """.format(transcription=transcription), unsafe_allow_html=True)

        # Download button with colorful styling
        st.download_button(
            label="📥 Download Transcription",
            data=transcription,
            file_name="transcription.txt",
            mime="text/plain",
            help="Save your transcription as a text file"
        )
