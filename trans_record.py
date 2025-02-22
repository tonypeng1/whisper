import json
import glob
import io
import os
import pdb
import time 

from datasets import Audio, Dataset, DatasetDict, Features, Value
from huggingface_hub import login, HfApi
# import librosa  # also an library for audio processing
import soundfile as sf
import streamlit as st


def get_transcription_from_jsonl_file(_data_folder: str, 
                                      _audio_file_name: str) -> str:
    """
    Retrieves the transcription for a given audio file from a JSONL metadata file.

    Args:
        data_folder (str): Path to the folder containing the metadata.jsonl file
        audio_file_name (str): Name of the audio file to look up

    Returns:
        str: The transcription text if found, None otherwise

    Raises:
        FileNotFoundError: If the metadata.jsonl file does not exist
        json.JSONDecodeError: If the JSONL file contains invalid JSON
    """

    jsonl_file_path = f"{_data_folder}/metadata.jsonl"

    try:
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if data['file_name'] == _audio_file_name:
                    return data['transcription']

    except FileNotFoundError:
        st.error(f"Metadata file not found at: {jsonl_file_path}")
        st.stop()
    except json.JSONDecodeError:
        st.error("Invalid JSON format in metadata file")
        st.stop()


def construct_dataset_from_metadata_file(metadata_file: str, 
                                         path: str,
                                         number: int) -> tuple:
    """
    Constructs dataset lists from a metadata file containing audio transcription data.

    Args:
        metadata_file (str): Path to the metadata JSONL file containing transcription info
        path (str): Base directory path where audio files are located
        number (int): Number of audio files to process

    Returns:
        tuple: A tuple containing three lists:
            - data_id (List[str]): List of unique IDs for each transcription
            - data_audio (List[str]): List of full paths to audio files
            - data_transcription (List[str]): List of transcription texts

    Raises:
        FileNotFoundError: If metadata_file does not exist
    """

    if not os.path.exists(metadata_file):
        error_msg = f"Warning: No metadata.jsonl found in {path}"
        st.error(error_msg)
        st.stop()
        raise FileNotFoundError(error_msg)

    data_id = []
    data_audio = []
    data_transcription = []

    with open(metadata_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= number:
                break

            item = json.loads(line)
            audio_file = os.path.join(path, item['file_name'])
            transcrip = item['transcription']
            
            data_id.append(str(i + 1))
            data_audio.append(audio_file)
            data_transcription.append(transcrip)
            # st.write(f"Processing audio file {i + 1}: {audio_file}")

        # pdb.set_trace() # Set a break point here
        return data_id, data_audio, data_transcription


def process_dataset(_base_path: str, 
                    split: str,
                    number: int) -> Dataset:
    """Process audio dataset and create a Hugging Face Dataset object.
    
    Args:
        _base_path (str): Base directory path containing the dataset
        split (str): Dataset split to process (e.g. 'train', 'test')
        
    Returns:
        Dataset: Hugging Face Dataset containing audio files and transcriptions 
        with features:
            - id (str): Unique identifier for each audio sample
            - path (str): Path to the audio file
            - audio (Audio): Audio array with 16kHz sampling rate
            - transcription (str): Text transcription of the audio
            
    Raises:
        FileNotFoundError: If metadata.jsonl file or audio directory not found
    """

    # Process dataset files
    path = os.path.join(_base_path, split)
    metadata_file = os.path.join(path, "metadata.jsonl")
 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset directory not found at {path}")

    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
 
    # Construct data lists from metadata file
    (_id,
     data_audio, 
     data_transcription) = construct_dataset_from_metadata_file(metadata_file, 
                                                                path,
                                                                number)

    # Define the feature schema
    features = Features({
        'id': Value('string'),
        'path': Value('string'),
        'audio': Audio(sampling_rate=16000),
        'transcription': Value('string')
    })

    # Create and return dataset
    dataset = Dataset.from_dict({
        'id': _id,
        'path': data_audio,
        'audio': data_audio,
        'transcription': data_transcription
    }, features=features)

    return dataset


def init_streamlit_session_state():

    if 'audio_key' not in st.session_state:
        st.session_state.audio_key = 0

    if 'data_group' not in st.session_state:
        st.session_state.data_group = "Train"


def count_number_of_wav_files(_base_path: str, split: str) -> int:
    """
    Count the number of WAV audio files in the specified directory.

    Args:
        directory (str): Path to the directory to scan for WAV files

    Returns:
        int: Number of WAV files found in the directory

    Raises:
        FileNotFoundError: If the specified directory does not exist
    """
    # Combine the base path and split (e.g., 'train' or 'test') to create full directory path
    directory = os.path.join(_base_path, split)

    # Check if directory exists, raise error if not
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    # Create a list of .wav files in the directory:
    # - Filters for files ending in .wav (case-insensitive)
    # - Ensures they are files (not directories)
    # - Creates full path by joining directory and filename
    files = [f for f in os.listdir(directory) if f.lower().endswith('.wav')
             and os.path.isfile(os.path.join(directory, f))]

    # Return the count of .wav files found
    return len(files)


train_folder = "data/train"
train_number_of_records = 793
# train_number_of_records = 6

test_folder = "data/test"
test_number_of_records = 264
# test_number_of_records = 6

init_streamlit_session_state()

if st.session_state.data_group == "Train":
    index = 0
else:
    index = 1

data_group = st.sidebar.radio(label="Choose dataset group:",
                 options=("Train", "Test"),
                 index=index,
                 key="dataset",
                 )

# Depending on the selected group, set the data folder and number of records
if data_group == "Train":
    data_folder = train_folder
    group = "train"
    number_of_records = train_number_of_records
else:
    data_folder = test_folder
    group = "test"
    number_of_records = test_number_of_records

# Find next available file name
try:
    audio_file_number = 1  # Number of the next audio file to be recorded or the number of all
    # records (if all files are recorded)
    audio_file_name = f"audio_{group}_{audio_file_number}.wav"
    audio_file_path = os.path.join(data_folder, audio_file_name)

    while os.path.exists(audio_file_path):
        audio_file_number += 1  
        if audio_file_number == number_of_records + 1:
            break
        else:
            audio_file_name = f"audio_{group}_{audio_file_number}.wav"
            audio_file_path = os.path.join(data_folder, audio_file_name)

except Exception as e:
    st.error(f"Error getting next available audio file number: {e}")

st.title(f"Record Audio File")

if audio_file_number == number_of_records + 1:
    st.success(f"{data_group} data audio recording completed: {number_of_records} records")
else:
    message_1 = f'{audio_file_number - 1} audio files saved in the "{data_group}" data set.'
    message_2 = "Here is the text of the next audio clip: "
    st.markdown(f'**<span style="font-size: 18px;">:green[{message_1}]<br>:green[Great job!]<br><br>:green[{message_2}]</span>**', unsafe_allow_html=True)

    transcription = get_transcription_from_jsonl_file(data_folder, audio_file_name)
    st.markdown(f'***<span style="font-size: 35px;">:blue[{transcription}]</span>***', unsafe_allow_html=True)

    audio_file = st.audio_input(
        label=f"Click to record the next audio file",
        key=f"audio_input_{st.session_state.audio_key}"
    )

    if audio_file is not None:
        confirmation = st.button(
                label="CLICK HERE TO SAVE THE AUDIO FILE",
            )
        
        # Save the audio file as .wav and increment st.session_state.audio_key
        if confirmation:
            with open(audio_file_path, "wb") as f:
                f.write(audio_file.getbuffer())
            st.success(f"Audio file saved as {audio_file_name}")
            st.session_state.audio_key += 1 
            st.rerun()
            
# The code below is for uploading the dataset to Hugging Face
st.sidebar.markdown("""----------""")
upload = st.sidebar.button(label="Upload to Hugging Face")

if upload:  
    base_path="data"
    hf_repo_id="tonypeng/whisper-finetuning"

    # Process the datasets 
    number_of_wav_files_train = count_number_of_wav_files(base_path, "train")
    number_of_wav_files_test = count_number_of_wav_files(base_path, "test")

    # pdb.set_trace() # Set a break point here

    # pdb.set_trace() # Set a break point here
    if number_of_wav_files_train > 0:
        train_dataset = process_dataset(base_path, "train", number_of_wav_files_train)
    else:
        train_dataset = None

    if number_of_wav_files_test > 0:
        test_dataset = process_dataset(base_path, "test", number_of_wav_files_test)
    else:
        test_dataset = None

    # Combine into a DatasetDict
    if train_dataset is not None and test_dataset is not None:
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    elif train_dataset is not None:
        dataset_dict = DatasetDict({
            'train': train_dataset
        })
    elif test_dataset is not None:
        dataset_dict = DatasetDict({
            'test': test_dataset
        })
    else:
        st.error("No dataset found to upload.")
        st.stop()

    # Log in to Hugging Face
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    # Push Push dataset metadata first
    if dataset_dict:
        dataset_dict.push_to_hub(
            repo_id=hf_repo_id,
            private=False
            )
        
        # api = HfApi()
        # for split in ['train', 'test']:
        #     base = os.path.join(base_path, split)
        #     for audio_file in glob.glob(os.path.join(base, "*.wav")):
        #         st.write(f"Processing audio_file: {audio_file}")
        #         api.upload_file(
        #             path_or_fileobj=audio_file,
        #             path_in_repo=f"{split}/{os.path.basename(audio_file)}",
        #             repo_id=hf_repo_id,
        #             repo_type="dataset"
        #         )

        st.success(f"Dataset successfully pushed to {hf_repo_id}")
    else:
        st.error("No data found to create a dataset.")
