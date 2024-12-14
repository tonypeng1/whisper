# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# model.config.forced_decoder_ids = None

# # load dummy dataset and read audio files
# ds = load_dataset(
#     "hf-internal-testing/librispeech_asr_dummy", 
#     "clean", 
#     split="validation"
#     )
# sample = ds[0]["audio"]

# input_features = processor(
#     sample["array"], 
#     sampling_rate=sample["sampling_rate"], 
#     return_tensors="pt"
#     ).input_features 

# # generate token ids
# predicted_ids = model.generate(input_features)

# # decode token ids to text
# transcription = processor.batch_decode(
#     predicted_ids, 
#     skip_special_tokens=False)

# # ['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

# transcription = processor.batch_decode(
#     predicted_ids, 
#     skip_special_tokens=True)

# # [' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']

import os

from datasets import load_dataset
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import sounddevice as sd


def play_audio_sample(
        _i: int,
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
        print(f"\nAudio file index {_i+1} playback finished.")
    except Exception as e:
        print(f"Error playing audio: {e}")


# Check if MPS (Metal Performance Shaders) or CUDA is available for GPU 
# acceleration on the current system
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS for GPU acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA for GPU acceleration.")
else:
    device = torch.device("cpu")
    print("Using CPU. Consider using a GPU for faster processing.")

# Define model type and model path
model_type = "openai/whisper-small.en"
model_path = model_type.split("/")[-1]  # "whisper-small.en"

if model_path not in os.listdir():  # Check if the model is already downloaded
    # Load the Whisper model in Hugging Face format:
    processor = WhisperProcessor.from_pretrained(model_type)
    model = WhisperForConditionalGeneration.from_pretrained(model_type).to(device)

    # Save the model and processor locally
    processor.save_pretrained(model_path)
    model.save_pretrained(model_path)

    print(f"\n{model_type} model and processor downloaded and saved locally.")

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

    print(f"Load {model_type} model and processor from local directory")

else:
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)

    print(f"\nLoad {model_type} model and processor from local directory")

# To support decoding audio files, please install 'librosa' and 'soundfile'.
# Select an audio file and read it:
ds = load_dataset(
    path="hf-internal-testing/librispeech_asr_dummy", 
    name="clean", 
    split="validation"
    )

# Only process the first 5 audio samples
for i in range(5):
    # Play the audio sample
    audio_sample = ds[i]["audio"]
    play_audio_sample(i, audio_sample)

    # Processor returns a dict with only one key: "input features". 
    # The value of this key is a PyTorch tensor.
    input_features = processor(
        audio_sample["array"], 
        sampling_rate=audio_sample["sampling_rate"], 
        return_tensors="pt"
    ).input_features.to(device)  # Move the tensor to the GPU

    # Create the attention mask
    attention_mask = torch.ones_like(input_features)  # Initialize with ones
    attention_mask[input_features == processor.tokenizer.pad_token_id] = 0  # Set padding tokens to zero

    # Generate token ids
    predicted_ids = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        )

    # Decode token ids to text
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
        )

    print(f"Audio file index {i+1} transcript: {transcription[0]}")