from dataclasses import dataclass
from functools import partial
import os
import pdb
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

from datasets import load_dataset, DatasetDict
import evaluate
import numpy as np
import sounddevice as sd
import streamlit as st
import torch
from transformers import Seq2SeqTrainer, \
                        Seq2SeqTrainingArguments, \
                        WhisperProcessor, \
                        WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


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
        
        # Load the model from Hugging Face 
        _processor = WhisperProcessor.from_pretrained(_model_type)
        _model = WhisperForConditionalGeneration.from_pretrained(_model_type).to(_device)

        # Save the model and processor locally
        _processor.save_pretrained(_model_path)
        _model.save_pretrained(_model_path)

        print(f"\n{_model_type} model and processor downloaded and saved to {_model_path}")
    else:
        print(f"\nLoading {_model_type} model and processor from {_model_path}")

    # Load from local directory
    _processor = WhisperProcessor.from_pretrained(_model_path)
    _model = WhisperForConditionalGeneration.from_pretrained(_model_path).to(_device)
        
    return _processor, _model


# def play_audio_sample(
#         _audio_sample: dict
#         ):
#     """
#     Play an audio sample.

#     Parameters:
#     - _audio_sample (dict): Dictionary containing 'array' and 'sampling_rate' of the audio.
#     """

#     try:
#         sd.play(
#             _audio_sample['array'], 
#             _audio_sample['sampling_rate']
#             )
#         sd.wait()  # Wait until sound has finished playing
#     except Exception as e:
#         print(f"Error playing audio: {e}")


# def preprocess(text: str) -> str:
#     """
#     Preprocess text by removing punctuation and converting to lowercase.
    
#     Args:
#         text (str): Input text to be preprocessed
        
#     Returns:
#         str: Processed text with punctuation removed and converted to lowercase
        
#     Example:
#         >>> preprocess("Hello, World!")
#         'hello world'
#     """
#     # Remove punctuation and special characters keeping only words and spaces
#     text = re.sub(r'[^\w\s]', '', text)
    
#     # Convert text to lowercase for consistency
#     return text.lower()


class EvalPrediction(NamedTuple):
    """
    Evaluation output (tuple of predictions and labels)
    """
    predictions: np.ndarray
    label_ids: np.ndarray


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute Word Error Rate (WER) metrics for the predicted transcriptions.

    Calculates both orthographic WER and normalized WER after preprocessing. 
    The normalized WER applies text normalization to both predictions and references
    before computing the error rate.

    Args:
        pred: Prediction object containing:
            - predictions: Predicted token IDs
            - label_ids: Ground truth label IDs 

    Returns:
        dict: Dictionary containing:
            - wer_ortho: Orthographic WER percentage
            - wer: Normalized WER percentage
    """

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]

    # The list comprehension filters out pairs where reference is empty:
    valid_samples = [(pred, ref) for pred, ref in zip(pred_str_norm, label_str_norm) 
                    if len(ref) > 0]
    
    # Unzip filtered samples
    pred_str_norm, label_str_norm = zip(*valid_samples)

    wer = 100 * wer_metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


def prepare_dataset(sample):
    """
    Convert a single dataset example: audio array to log-mel spectrogram and 
    transcription(text) to to token IDs.
    
    Args:
        example (dict): Dictionary containing audio data and transcription text.
            The audio data should have 'array' and 'sampling_rate' fields.
            The transcription should be in the 'transcription' field.
            
    Returns:
        dict: Processed example with the following fields:
            - Processed audio features from the processor (log-mel spectrogram)
            - Labels (processed text tokens)
            - Input audio length in seconds
    """

    # load raw audio data (intensity values vs time)
    audio = sample["audio"]  

    # transforms the raw audio data into log-mel spectrogram features and text into token IDs 
    # that can be directly fed into the corresponding model.
    sample = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=sample["transcription"],
        return_tensors="pt",
    )  # no need to send to GPU, it will be done later

    # compute input length of audio sample in seconds
    sample["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return sample


# def is_audio_in_length_range(length: float, max_length: float = 30.0) -> bool:
#     """
#     Check if audio length is within acceptable range.
    
#     Args:
#         length (float): Duration of audio in seconds
#         max_length (float, optional): Maximum allowed duration in seconds. Defaults to 30.
        
#     Returns:
#         bool: True if audio length is less than max_length, False otherwise
#     """
    
#     return length < max_length

# def remove_long_audio_files(dataset_: DatasetDict) -> DatasetDict:
    
#     dataset_ = dataset_.filter(
#         is_audio_in_length_range,
#         input_columns=["input_length"],
#         )
#     return dataset_

def remove_long_audio_files(dataset_: DatasetDict, max_length: float = 30.0) -> DatasetDict:

    dataset_ = dataset_.filter(
        lambda x: x < max_length,
        input_columns=["input_length"],
    )
    return dataset_


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text tasks that handles padding of audio features and text labels.
    
    This collator takes a list of features containing audio input features and text labels, 
    pads them appropriately, and prepares them for model training by:

    1. Padding audio features to max length using the feature extractor
    2. Padding text labels to max length using the tokenizer
    3. Masking padded label tokens with -100 to ignore them in loss calculation
    4. Removing BOS token from labels if present
    
    Args:
        processor: A processor containing a feature extractor and tokenizer
        
    Returns:
        Dict containing padded input features and processed labels as torch tensors
    """
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:

        # Extract just the input features of audio from each sample
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]

        # Pad input features and convert to torch tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad the labels to max length and covert to torch tensors
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so they are ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # # If BOS token is appended in previous tokenization step,
        # # cut bos token here as it's append later anyways
        # if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        # Remove BOS (beginning of sentence) token if present since it will be added later
        if torch.all(labels[:, 0] == self.processor.tokenizer.bos_token_id):
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def generate_transcription(
        _input_features,
        _attention_mask=None,
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


def add_audio_sample_length(dataset_: DatasetDict) -> DatasetDict:
    """
    Add the length of each audio sample in seconds to the dataset.

    This function iterates through each sample in the dataset, calculates the length of the
    audio array in seconds, and adds this length as a new field 'input_length' to the sample.
    The modified samples are then collected into a new dataset.

    Args:
        dataset_ (DatasetDict): The dataset containing audio samples.

    Returns:
        DatasetDict: The modified dataset with 'input_length' field added to each sample.
    """
    def compute_input_length(sample):
        audio = sample["audio"]
        sample["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return sample

    new_dataset = dataset_.map(compute_input_length)
    return new_dataset


# def calculate_wer_of_dataset(_dataset: DatasetDict) -> float:

#     ground_truth_list = []
#     transcription_list = []

#     for sample in _dataset:
#         audio_sample = sample["audio"]
#         audio_array = audio_sample["array"]
#         sampling_rate = audio_sample["sampling_rate"]
#         ground_truth = sample["transcription"]

#         # pdb.set_trace() # Set a break point here to inspect the variables

#         # Create input features (Pytorch tensor). Processor() returns a dict with only one key: 
#         # "input features", whose value is a PyTorch tensor.
#         input_features = processor(
#             audio_array, 
#             sampling_rate=sampling_rate, 
#             return_tensors="pt"
#         ).input_features.to(device)  # Move the tensor to the GPU

#         # Create the attention mask
#         attention_mask = torch.ones_like(input_features)  # Initialize with ones
#         attention_mask[input_features == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to zero

#         # Generate transcription using the model
#         transcription = generate_transcription(
#             input_features, 
#             attention_mask,
#             )

#         transcription_list.append(transcription[0])
#         ground_truth_list.append(ground_truth)

#     # Normalize each word
#     transcription_list_norm = [normalizer(pred) for pred in transcription_list]
#     ground_truth_list_norm = [normalizer(label) for label in ground_truth_list]

#     # compute normalised WER
#     _wer = wer_metric.compute(predictions=transcription_list_norm, 
#                                         references=ground_truth_list_norm)
#     return _wer


def calculate_wer_norm_of_dataset(_dataset: DatasetDict) -> float:
    """
    Calculate the Word Error Rate (WER) for a dataset by comparing model transcriptions 
    with ground truth labels.

    This function:
    1. Processes each audio sample in the dataset through the model
    2. Generates transcriptions for each sample
    3. Normalizes both predictions and ground truth text
    4. Computes the overall WER across all samples

    Args:
        _dataset (DatasetDict): Dataset containing audio samples and transcriptions.
                               Each sample should have 'audio' and 'transcription' fields.

    Returns:
        float: The overall Word Error Rate as a decimal (e.g. 0.15 for 15% WER)

    Note:
        - Uses the global model, processor, device and normalizer objects
        - Audio samples are processed in a batched manner for efficiency
        - Both predictions and ground truth undergo text normalization before WER calculation
    """
    ground_truth_list = []
    transcription_list = []

    # Process dataset in batches
    batch_size = 16  # Can be adjusted based on available memory
    for i in range(0, len(_dataset), batch_size):
        batch = _dataset[i:i + batch_size]
        
        # Prepare batch inputs
        audio_arrays = [sample["array"] for sample in batch["audio"]]
        sampling_rates = [sample["sampling_rate"] for sample in batch["audio"]]
        ground_truths = batch["transcription"]

        # Process audio batch
        inputs = processor(
            audio_arrays,
            sampling_rate=sampling_rates[0],  # Assuming same sampling rate for all
            text=ground_truths,
            return_tensors="pt",
            padding=True
        ).to(device)  # Move all attributes to the GPU

        # Generate transcriptions for batch
        with torch.no_grad():  # Disable gradient calculation for inference
            transcriptions = generate_transcription(
                inputs.input_features,
                # inputs.attention_mask
            )

        # Extend lists with batch results
        transcription_list.extend(transcriptions)
        ground_truth_list.extend(ground_truths)

    # Normalize predictions and ground truth
    transcription_list_norm = [normalizer(pred) for pred in transcription_list]
    ground_truth_list_norm = [normalizer(label) for label in ground_truth_list]

    # Compute WER
    wer_norm = wer_metric.compute(
        predictions=transcription_list_norm,
        references=ground_truth_list_norm
    )

    return wer_norm


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

# Download the "train" and "test" dataset from Hugging Face (each is a list of samples).
# Each sample is a dictionary with the following keys: 'id', 'path', 'audio', 'transcription'
# The 'audio' key is a dictionary with the following keys: 'path', 'array', 'sampling_rate'
dataset_path = "tonypeng/whisper-finetuning"

dataset = DatasetDict()
dataset["train"] = load_dataset(
    path=dataset_path, 
    split="train",
    )
dataset["test"] = load_dataset(
    path=dataset_path,
    split="test",
    )

# Add the audio sample length to the dataset
dataset["train"] = add_audio_sample_length(dataset["train"])
dataset["test"] = add_audio_sample_length(dataset["test"])

# Filter out audio files that are too long (more than 30 seconds)
max_length = 30.0  # seconds
dataset["train"] = remove_long_audio_files(dataset["train"])
dataset["test"] = remove_long_audio_files(dataset["test"])

dataset_train_length = len(dataset["train"])
dataset_test_length = len(dataset["test"])

# Load the WER metric
wer_metric = evaluate.load("wer")

# Replace any markers, symbols, and punctuations with a space, and drop any diacritics
normalizer = BasicTextNormalizer(remove_diacritics=True)

# Compute the original WER of the dataset
original_wer_norm = calculate_wer_norm_of_dataset(dataset["test"])

# Convert the dataset (audio array to log-mel spectrogram and text to to token IDs)
converted_dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names["train"], 
    num_proc=1
    )
# dataset.column_names["train"] output:['id', 'path', 'audio', 'transcription']
# So, the above line removes all existing columns and only keeps the output of the
# prepare_dataset function.

# Create a data collator that handles padding of audio features and text labels.
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False

# set language and task for generation and re-enable cache
model.generate = partial(
    model.generate, task="transcribe", use_cache=True
    )

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-finetuned",  # name on the HF Hub
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=converted_dataset["train"],
    eval_dataset=converted_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()





