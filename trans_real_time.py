# import base64
import io
import os

import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample
import sounddevice as sd
import streamlit as st
# import streamlit.components.v1 as components
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def check_system_accelleration():
    """
    Check if MPS (Metal Performance Shaders) or CUDA is available for GPU
    acceleration on the current system.
    Returns:
    - device (torch.device): The device to be used for GPU acceleration.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS for GPU acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU. Consider using a GPU for faster processing.")
    return device


def load_processor_and_model(
        _model_type: str,
        _model_path: str, 
        _device: torch.device
        ) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """
    Load or download a Whisper model and processor from Hugging Face.

    This function checks if the specified model exists locally. If not, it downloads
    the model from Hugging Face and saves it. The model and processor are then loaded
    either from the local directory or from the downloaded files.

    Args:
        _model_type: The Hugging Face model identifier (e.g. "openai/whisper-small")
        _model_path: Local directory path to save/load the model
        _device: PyTorch device to load the model onto (CPU/GPU)

    Returns:
        tuple: (WhisperProcessor, WhisperForConditionalGeneration)
            - The loaded processor
            - The loaded model on the specified device
    """

    # Check if the model exists locally. If not, download and save it.
    if not os.path.exists(_model_path):
        os.makedirs(_model_path, exist_ok=True)
        
        # Load and save the model from Hugging Face
        _processor = WhisperProcessor.from_pretrained(_model_type)
        _model = WhisperForConditionalGeneration.from_pretrained(_model_type).to(_device)

        _processor.save_pretrained(_model_path)
        _model.save_pretrained(_model_path)

        print(f"\n{_model_type} model and processor downloaded and saved to {_model_path}")
    else:
        print(f"\nLoading {_model_type} model and processor from {_model_path}")

    # Load from local directory
    _processor = WhisperProcessor.from_pretrained(_model_path)
    _model = WhisperForConditionalGeneration.from_pretrained(_model_path).to(_device)
        
    return _processor, _model


def generate_transcription(
        _input_features,
        _attention_mask,
        ):
    # Generate token ids
    predicted_ids = model.generate(
        input_features=_input_features,
        attention_mask=_attention_mask,
        )

    # Decode token ids to text
    _transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
        )
    
    return _transcription


def process_audio_input(
        audio_file, 
        sampling_rate: int = 16000
        ) -> list[dict] | None:
    """
    Process audio input from a file-like object.

    Parameters:
    - audio_file: A file-like object containing audio data.
    - sampling_rate: Target sampling rate (e.g., 16000 for Whisper).

    Returns:
    - A list containing a single dictionary with the processed audio data (numpy array) and sampling rate.
      Example: [{'audio': {'array': np.array, 'sampling_rate': int}}] or None on failure.
    """
    if audio_file is None:
        return None

    try:
        # Read the audio file as bytes
        audio_bytes = audio_file.read()

        # Create a BytesIO object
        byte_io = io.BytesIO(audio_bytes)

        # Read the WAV file
        orig_sr, audio_array = read(byte_io)

        # Convert to float32
        audio_array = audio_array.astype(np.float32)

        # Normalize to [-1, 1] if necessary (assuming 16-bit PCM)
        if audio_array.max() > 1.0:
            audio_array = audio_array / 32768.0  # 32768 is the max value for 16-bit PCM

        # Resample if necessary
        if orig_sr != sampling_rate:
            num_samples = int(len(audio_array) * sampling_rate / orig_sr)
            audio_array = resample(audio_array, num_samples)

        return [{'audio': {
            'array': audio_array,
            'sampling_rate': sampling_rate
        }}]

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def play_audio_sample(
        _audio_sample: dict
        ):
    """
    Play an audio sample.

    Parameters:
    - _audio_sample (dict): Dictionary containing 'array' and 'sampling_rate' of the audio.
    """

    try:
        sd.play(
            _audio_sample['array'], 
            _audio_sample['sampling_rate']
            )
        sd.wait()  # Wait until sound has finished playing
    except Exception as e:
        print(f"Error playing audio: {e}")


## Work for replaying audio, but only once.
# def autoplay_audio(audio_data):
#     if audio_data is not None:
#         audio_bytes = audio_data.getvalue()  # Get the bytes from the UploadedFile object
#         b64 = base64.b64encode(audio_bytes).decode('utf-8')
#         md = f"""
#         <audio autoplay="true" style="display:none">
#         <source src="data:audio/wav;base64,{b64}" type="audio/wav">
#         </audio>
#         """
#         st.markdown(md, unsafe_allow_html=True)
#     else:
#         pass


## Not working
# def autoplay_audio(audio_data):
#     audio_bytes = audio_data.getvalue()  # Get the bytes from the UploadedFile object
#     b64 = base64.b64encode(audio_bytes).decode('utf-8')
#     md = f"""
#     <script>
#     var audio = new Audio('data:audio/wav;base64,{b64}');
#     audio.play();
#     </script>
#     """
#     st.markdown(md, unsafe_allow_html=True)


st.title("Real-Time Speech Transcription")

# Check system accelleration
device = check_system_accelleration()

# Define model type and model path
model_type = "openai/whisper-small.en"
model_path = model_type.split("/")[-1]  # "whisper-small.en"

# Load processor and model
(processor, 
model) = load_processor_and_model(
                                _model_type=model_type,
                                _model_path=model_path,
                                _device=device
                                )

audio_file = st.audio_input(label="Record a voice message to transcribe")

if audio_file:

    # Process the audio
    dataset = process_audio_input(audio_file)

    # Play the audio sample
    audio_sample = dataset[0]["audio"]
    # play_audio_sample(audio_sample)  # Read out the audio clip. Not working in Docker container.

    audio = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    # Create input features (Pytorch tensor)
    # Processor() returns a dict with only one key: "input features". 
    # The value of this key is a PyTorch tensor.
    input_features = processor(
        audio, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).input_features.to(device)  # Move the tensor to the GPU

    # Create the attention mask
    attention_mask = torch.ones_like(input_features)  # Initialize with ones
    attention_mask[input_features == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to zero

    transcription = generate_transcription(
        input_features, 
        attention_mask,
        )

    st.divider()
    st.markdown("**Whisper Small Model Transcription:**")
    st.markdown("*"+ transcription[0])