import os

from huggingface_hub import create_repo, HfApi

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

if not HUGGINGFACE_TOKEN:
    print("Error: HUGGINGFACE_TOKEN environment variable is not set")
    exit(1)

model_path = "whisper-small-finetuned"  # Your local directory
hf_repo_id = "tonypeng/whisper-finetuning"  # Your Hugging Face repo ID

# 3. Files to include (explicitly list them)
include_patterns = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "training_args.bin",
    "vocab.json",
]

api = HfApi()

# Create model repository
try:
    api.create_repo(hf_repo_id, repo_type="model", exist_ok=True)
    print(f"Repository {hf_repo_id} created or already exists.")
except Exception as e:
    print(f"Error creating Hugging Face repository: {e}")

# Upload model
try:
    api.upload_folder(
        folder_path=model_path,
        repo_id=hf_repo_id,
        repo_type="model",
        token=HUGGINGFACE_TOKEN,
        allow_patterns=include_patterns,
    )
    print(f"Successfully uploaded model to {hf_repo_id}")

except Exception as e:
    print(f"Error uploading the fine-tuned model to Hugging Face repository: {e}")

