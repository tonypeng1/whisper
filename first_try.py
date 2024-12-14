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


# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")

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

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        audio_sample["array"], 
        sampling_rate=audio_sample["sampling_rate"], 
        return_tensors="pt"
    ).input_features

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