import os
import pdb
import re
import sys
import time

from datasets import load_dataset
import evaluate
import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import sounddevice as sd


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


def generate_transcription(
        _input_features,
        _attention_mask = None,
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


def get_integer_range(_range_input, _dataset_length):

    try:
        range_list = _range_input.split(",")

        if len(range_list) == 2:

            try:
                start = int(range_list[0])
                end = int(range_list[1])
            except ValueError:
                raise ValueError("Please enter valid integers separated by a comma.")

            if start >=0 and end <= _dataset_length:
                return start - 1, end
            else:
                raise ValueError(f"The numbers must be betwee 1 and {_dataset_length}.")

        else:
            raise ValueError("Invalid format. Please enter 'start,end'.")
    except ValueError as e:
        with placeholder_get_integer_range_sesson.container():  # To prevent the grays-out display
            st.error(e) 
        st.stop()


# def preprocess(text):
#     text = re.sub(r'[^\w\s]', '', text)  # Removes anything that's not a word or space
#     return text.lower()  # Convert to lowercase


st.title("Whisper Transcription")

# # Select dataset in Hugging Face
# dataset_path = "hf-internal-testing/librispeech_asr_dummy"
# dataset_name = "clean"
# dataset_split = "validation"

dataset_path = "tonypeng/whisper-finetuning"
# dataset_name = "clean"
# dataset_split = "train"
dataset_split = "test"

st.markdown(f'**<span style="font-size: 18px;">:green[Hugging Face data repository: {dataset_path}] \
            <br>:green[Data set split: {dataset_split}]</span>**<br>', \
            unsafe_allow_html=True,
            )

# Check system accelleration
device = check_system_accelleration()

# Define model type and model path
model_type = "openai/whisper-small.en"
# model_path = model_type.split("/")[-1]  # "whisper-small.en"
model_path = "whisper-small-finetuned"

# Load processor and model
(processor, 
 model) = load_processor_and_model(
                                _model_type=model_type,
                                _model_path=model_path,
                                _device=device
                                )

# # Select an audio file from the dataset
# dataset = load_dataset(
#     path=dataset_path, 
#     name=dataset_name, 
#     split=dataset_split,
#     )

# Select an audio file from the dataset
dataset = load_dataset(
    path=dataset_path, 
    split=dataset_split,
    )

dataset_length = len(dataset)
# st.write(dataset)
st.sidebar.write(f"Number of clips in the audio repository: {dataset_length}")

if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []

if "text_input" not in st.session_state:
    st.session_state.text_input = 0

range_input = st.sidebar.text_input(
    label=f"Enter integer range (start,end) of audio clips to play:",
    key=f"range_inpuut_{st.session_state.text_input}"
    )

# Create a container for transcriptions at the top
# transcription_container = st.container()
# This following 2 lines have to be before the iteration below to remove the grayed-out display
placeholder_transcription_sesson = st.empty()  
placeholder_get_integer_range_sesson = st.empty()  

# Load the WER metric
wer_metric = evaluate.load("wer")

ground_truth_list = []
transcription_list = []

# Initialize the normalizer before loading the dataset
normalizer = BasicTextNormalizer(remove_diacritics=True)

if range_input:

    st.session_state.transcriptions = []

    (clip_start, clip_end) = get_integer_range(range_input, dataset_length)
    # st.write(f"clip start: {clip_start}, clip end: {clip_end}")

    # Initialize counters for total word errors and total words
    total_word_errors = 0
    total_words = 0

    # Process audio input and generate transcription
    for i in range(clip_start, clip_end):

        # Play the audio sample
        audio_sample = dataset[i]["audio"]
        # audio_sample = dataset[i]
        # st.write(audio_sample)
        play_audio_sample(audio_sample)  # Read out the audio clip

        audio = audio_sample["array"]
        sampling_rate = audio_sample["sampling_rate"]

        # pdb.set_trace() # Set a break point here to inspect the variables
        # Create input features (Pytorch tensor)
        # Processor() returns a dict with only one key: "input features". 
        # The value of this key is a PyTorch tensor.
        input_features = processor(
            audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features.to(device)  # Move the tensor to the GPU

        # # Create the attention mask
        # attention_mask = torch.ones_like(input_features)  # Initialize with ones
        # attention_mask[input_features == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to zero

        transcription = generate_transcription(
            input_features,
            # attention_mask,
            )

        # Get the ground truth text
        ground_truth = dataset[i]["transcription"]

        # # Preprocess the transcription and ground truth to remove punctuation and convert to lowercase
        # transcription[0] = preprocess(transcription[0])
        # ground_truth = preprocess(ground_truth)

        # Use the Whisper normalizer instead of the custom preprocess function
        transcription[0] = normalizer(transcription[0])
        ground_truth = normalizer(ground_truth)

        transcription_list.append(transcription[0])
        ground_truth_list.append(ground_truth)

        # Compute WER
        wer = 100 * wer_metric.compute(predictions=[transcription[0]], references=[ground_truth])

        # Compute accumulated WER
        accumulated_wer = 100 * wer_metric.compute(predictions=transcription_list, references=ground_truth_list)

        # Store transcription in session state
        st.session_state.transcriptions.append({
            'index': i + 1,
            'text': transcription[0],
            'ground_truth': ground_truth,
            'wer': wer,
            'accumulated_wer': accumulated_wer
        })

        # st.write(f"\nAudio file index {i+1} transcript: {transcription[0]}")
        st.markdown(f'***<span style="font-size: 18px;"> \
                    Audio file {i+1}: \
                    <br>Transcription: :blue[{transcription[0]}] \
                    <br>Ground Truth: :green[{ground_truth}] \
                    <br>Word Error Rate (WER): :red[{wer:.1f}] \
                    <br>Accumulated WER: :red[{accumulated_wer:.1f}] \
                    </span>***',
                    unsafe_allow_html=True,
                    )

    st.session_state.text_input += 1
    time.sleep(0.2)  # Add a small delay
    st.rerun()


# Display all transcriptions from session state
# if st.session_state.should_display:
with placeholder_transcription_sesson.container():
    for trans in st.session_state.transcriptions:
        st.markdown(f'***<span style="font-size: 18px;"> Audio file index {trans["index"]}: \
                    <br>Transcription: :blue[{trans["text"]}] \
                    <br>Ground Truth: :green[{trans["ground_truth"]}] \
                    <br>Word Error Rate (WER): :red[{trans["wer"]:.1f}] \
                    <br>Accumulated WER: :red[{trans["accumulated_wer"]:.1f}] \
                    </span>***',
                    unsafe_allow_html=True,
                    )
