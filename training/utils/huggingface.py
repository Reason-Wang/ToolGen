from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, repo_info, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import torch


def repo_exists(repo_id, repo_type: Optional[str]=None, token: Optional[str]=None):
    """
    Check if a repository exists on the Hugging Face Hub

    Args:
        repo_id (str): The repository ID to check
        repo_type (str): The type of repository to check
        token (str): The Hugging Face API token

    Returns:
        bool: Whether the repository exists
    """
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False


def upload_model(model_name_or_path, repo_id, private=False, token=""):
    """
    Upload a model to the Hugging Face Hub

    Args:
        model_name_or_path (str): The model name or path to upload
        repo_id (str): The repository ID to upload the model to
    """
    # Load the model
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    if not repo_exists(repo_id, token=token):
        print(f"Repo {repo_id} does not exist, creating repo...")
        create_repo(repo_id, private=private, token=token)

    model.push_to_hub(repo_id, token=token)