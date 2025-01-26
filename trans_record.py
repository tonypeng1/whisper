import json
import io
import os
import time 
import wave

# from datasets import Audio, Dataset, load_dataset
# import numpy as np
# from scipy.io.wavfile import read
# from scipy.signal import resample
import streamlit as st
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration


# def check_system_accelleration():
#     """
#     Check if MPS (Metal Performance Shaders) or CUDA is available for GPU
#     acceleration on the current system.
#     Returns:
#     - device (torch.device): The device to be used for GPU acceleration.
#     """
#     if torch.backends.mps.is_available():
#         device = torch.device("mps")
#         print("Using MPS for GPU acceleration.")
#     elif torch.cuda.is_available():
#         device = torch.device("cuda")
#         print("Using CUDA for GPU acceleration.")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU. Consider using a GPU for faster processing.")
#     return device


# def load_processor_and_model(
#         _model_type: str,
#         _model_path: str, 
#         _device: torch.device
#         ) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
#     """
#     Load or download a Whisper model and processor from Hugging Face.

#     This function checks if the specified model exists locally. If not, it downloads
#     the model from Hugging Face and saves it. The model and processor are then loaded
#     either from the local directory or from the downloaded files.

#     Args:
#         _model_type: The Hugging Face model identifier (e.g. "openai/whisper-small")
#         _model_path: Local directory path to save/load the model
#         _device: PyTorch device to load the model onto (CPU/GPU)

#     Returns:
#         tuple: (WhisperProcessor, WhisperForConditionalGeneration)
#             - The loaded processor
#             - The loaded model on the specified device
#     """

#     # Check if the model exists locally. If not, download and save it.
#     if not os.path.exists(_model_path):
#         os.makedirs(_model_path, exist_ok=True)
        
#         # Load and save the model from Hugging Face
#         _processor = WhisperProcessor.from_pretrained(_model_type)
#         _model = WhisperForConditionalGeneration.from_pretrained(_model_type).to(_device)

#         _processor.save_pretrained(_model_path)
#         _model.save_pretrained(_model_path)

#         print(f"\n{_model_type} model and processor downloaded and saved to {_model_path}")
#     else:
#         print(f"\nLoading {_model_type} model and processor from {_model_path}")

#     # Load from local directory
#     _processor = WhisperProcessor.from_pretrained(_model_path)
#     _model = WhisperForConditionalGeneration.from_pretrained(_model_path).to(_device)
        
#     return _processor, _model


# def generate_transcription(
#         _input_features,
#         _attention_mask,
#         ):
#     # Generate token ids
#     predicted_ids = model.generate(
#         input_features=_input_features,
#         attention_mask=_attention_mask,
#         )

#     # Decode token ids to text
#     _transcription = processor.batch_decode(
#         predicted_ids, 
#         skip_special_tokens=True
#         )
    
#     return _transcription


# def process_audio_input(
#         audio_file, 
#         sampling_rate: int = 16000
#         ) -> list[dict] | None:
#     """
#     Process audio input from a file-like object.

#     Parameters:
#     - audio_file: A file-like object containing audio data.
#     - sampling_rate: Target sampling rate (e.g., 16000 for Whisper).

#     Returns:
#     - A list containing a single dictionary with the processed audio data (numpy array) and sampling rate.
#       Example: [{'audio': {'array': np.array, 'sampling_rate': int}}] or None on failure.
#     """
#     if audio_file is None:
#         return None

#     try:
#         # Read the audio file as bytes
#         audio_bytes = audio_file.read()

#         # Create a BytesIO object
#         byte_io = io.BytesIO(audio_bytes)

#         # Read the WAV file
#         orig_sr, audio_array = read(byte_io)

#         # Convert to float32
#         audio_array = audio_array.astype(np.float32)

#         # Normalize to [-1, 1] if necessary (assuming 16-bit PCM)
#         if audio_array.max() > 1.0:
#             audio_array = audio_array / 32768.0  # 32768 is the max value for 16-bit PCM

#         # Resample if necessary
#         if orig_sr != sampling_rate:
#             num_samples = int(len(audio_array) * sampling_rate / orig_sr)
#             audio_array = resample(audio_array, num_samples)

#         return [{'audio': {
#             'array': audio_array,
#             'sampling_rate': sampling_rate
#         }}]

#     except Exception as e:
#         print(f"Error processing audio: {e}")
#         return None


# def init_session_states():
#     if "test_audio_finish" not in st.session_state:
#         st.session_state.test_audio_finish = False
#     if "train_audio_finish" not in st.session_state:
#         st.session_state.train_audio_finish = False


# # Python dictionary
# python_dict = {
#     "name": "John Doe",
#     "age": 30,
#     "is_active": True,
#     "hobbies": ["reading", "coding"],
#     "address": {"street": "123 Main St", "city": "Anytown"}
# }

# # Convert Python dictionary to JSON string
# json_string = json.dumps(python_dict)
# print("JSON String:", json_string)

# st.title("Real-Time Speech Transcription")

# loaded_dict = json.loads(json_string)
# print("Python Dictionary:", loaded_dict)

# init_session_states()


def get_transcription_from_jsonl_file(_data_folder, 
                                      _audio_file_name):

    _transcription = None

    # st.write(f"Data folder: {_data_folder}")
    # st.write(f"Audio file name: {_audio_file_name}")
    jsonl_file_path = f"{_data_folder}/metadata.jsonl"
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['file_name'] == _audio_file_name:
                _transcription = data['transcription']
                break

    if _transcription:
        pass
    else:
        st.error("File not found in the JSONL file.")
    
    return _transcription


train_folder = "dataset/data/train"
train_number_of_records = 794

test_folder = "dataset/data/test"
test_number_of_records = 265

if 'audio_key' not in st.session_state:
    st.session_state.audio_key = 0

if 'data_group' not in st.session_state:
    st.session_state.data_group = "Train"

if st.session_state.data_group == "Train":
    index = 0
else:
    index = 1

data_group = st.sidebar.radio(label="Choose dataset group:",
                 options=("Train", "Test"),
                 index=index,
                 key="dataset",
                 )

if data_group == "Train":
    data_folder = train_folder
    group = "train"
    number_of_records = train_number_of_records
else:
    data_folder = test_folder
    group = "test"
    number_of_records = test_number_of_records

try:
    # Find next available file name
    audio_file_number = 1
    audio_file_name = f"audio_{group}_{audio_file_number}.wav"
    audio_file_path = os.path.join(data_folder, audio_file_name)

    while os.path.exists(audio_file_path) and audio_file_number < number_of_records:
        audio_file_number += 1
        audio_file_name = f"audio_{group}_{audio_file_number}.wav"
        audio_file_path = os.path.join(data_folder, audio_file_name)

except Exception as e:
    st.error(f"Error getting next available audio file number: {e}")

st.title(f"Record Audio File")

if audio_file_number == number_of_records:
    # st.write(f"audio file number is: {audio_file_number}")
    # st.write(f"audio file path is: {audio_file_path}")
    # st.write(f"Next available audio file number: {audio_file_number}")
    st.success(f"{data_group} data audio recording completed: {number_of_records} records")
else:

    # # Check system accelleration
    # device = check_system_accelleration()

    # # Define model type and model path
    # model_type = "openai/whisper-small.en"
    # model_path = model_type.split("/")[-1]  # "whisper-small.en"

    # # Load processor and model
    # (processor, 
    # model) = load_processor_and_model(
    #                                 _model_type=model_type,
    #                                 _model_path=model_path,
    #                                 _device=device
    #                                 )

    message_1 = f'{audio_file_number - 1} audio files saved in the "{data_group}" data set.'
    message_2 = "Here is the text of the next audio clip: "
    st.markdown(f'**<span style="font-size: 18px;">:green[{message_1}]<br>:green[Great job!]<br><br>:green[{message_2}]</span>**', unsafe_allow_html=True)

    transcription = get_transcription_from_jsonl_file(data_folder, audio_file_name)
    st.markdown(f'***<span style="font-size: 35px;">:blue[{transcription}]</span>***', unsafe_allow_html=True)

    # audio_container = st.empty()
    # audio_file = audio_container.audio_input(label=f"Click to record the next audio file")

    # audio_file = audio_container.audio_input(
    #     label=f"Click to record the next audio file",
    #     key=f"audio_input_{st.session_state.audio_key}"
    # )

    audio_file = st.audio_input(
        label=f"Click to record the next audio file",
        key=f"audio_input_{st.session_state.audio_key}"
    )

    # audio_file = audio_container.audio_input(
    #     label=f"Click to record the next audio file",
    #     # key=f"audio_input_{int(time.time())}"  # Add a timestamp-based unique key
    #     )

    # audio_file = st.audio_input(label=f"Click to record the next audio file")

    # if "record" not in st.session_state:
    #     st.session_state.record = False

    # if audio_file is not None:
    #     st.session_state.record = True

    if audio_file is not None:
        # st.write("Audio file received")
        confirmation = st.button(
                label="CLICK HERE TO SAVE THE AUDIO FILE",
            )
        
        if confirmation:
            # Save the audio file as .wav
            with open(audio_file_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.success(f"Audio file saved as {audio_file_name}")
            st.session_state.audio_key += 1 
            st.rerun()
            

        # audio_file = audio_container.audio_input(label=f"Click to record the next audio file")

        # # Process the audio
        # dataset = process_audio_input(audio_file)

        # # Play the audio sample
        # audio_sample = dataset[0]["audio"]

        # audio = audio_sample["array"]
        # sampling_rate = audio_sample["sampling_rate"]

        # # Create input features (Pytorch tensor)
        # # Processor() returns a dict with only one key: "input features". 
        # # The value of this key is a PyTorch tensor.
        # input_features = processor(
        #     audio, 
        #     sampling_rate=sampling_rate, 
        #     return_tensors="pt"
        # ).input_features.to(device)  # Move the tensor to the GPU

        # # Create the attention mask
        # attention_mask = torch.ones_like(input_features)  # Initialize with ones
        # attention_mask[input_features == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to zero

        # transcription = generate_transcription(
        #     input_features, 
        #     attention_mask,
        #     )

        # st.divider()
        # st.markdown("#### Whisper Small Model Transcription:")
        # st.markdown(f'***<span style="font-size: 24px;">:blue[{transcription[0]}]</span>***', unsafe_allow_html=True)