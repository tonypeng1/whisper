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
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# To support decoding audio files, please install 'librosa' and 'soundfile'.
# Select an audio file and read it:
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio_sample = ds[0]["audio"]

# Load the Whisper model in Hugging Face format:
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Use the model and processor to transcribe the audio:
input_features = processor(
    audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt"
).input_features

# Generate token ids
predicted_ids = model.generate(input_features)

# Decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

transcription[0]
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'