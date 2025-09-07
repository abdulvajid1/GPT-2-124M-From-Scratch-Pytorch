from huggingface_hub import upload_folder, upload_large_folder, create_repo
from huggingface_hub import login
from dotenv import load_dotenv
import os
import shutil
import pathlib
import glob
from pathlib import Path

def save_to_hf(global_step: int, keep_last_only: bool = True):
    """
    Save checkpoints to HuggingFace Hub while managing local checkpoint storage.
    
    Args:
        global_step: Current training step
        keep_last_only: If True, removes all checkpoints except the latest
    """
    checkpoints_dir = Path('checkpoints')
    
    # Find all checkpoint directories
    ckpt_pattern = os.path.join(checkpoints_dir, 'ckpt_*')
    checkpoint_dirs = sorted(glob.glob(ckpt_pattern))
    
    if not checkpoint_dirs:
        raise ValueError("No checkpoints found in directory")
    
    # Remove old checkpoints if keep_last_only is True
    if keep_last_only:
        for ckpt_dir in checkpoint_dirs[:-2]: # delete all checkpoints except last two checkpoint
            try:
                shutil.rmtree(ckpt_dir)
                print(f"Removed old checkpoint: {ckpt_dir}")
            except Exception as e:
                print(f"Error removing checkpoint {ckpt_dir}: {e}")

    create_repo(repo_id='Abdulvajid/gpt2-from-scratch',
                repo_type='model',
                exist_ok=True)

    # Upload everything in current dir
    upload_folder(
        repo_id="Abdulvajid/gpt2-from-scratch",
        repo_type='model',
        folder_path="checkpoints",           # your project dir
        commit_message="Upload checkpoints",
        ignore_patterns=[".git", "__pycache__",".venv"]
    )

    # # Upload everything in current dir
    # upload_folder(
    #     repo_id="Abdulvajid/gpt2-dataset",
    #     repo_type='checkpoints',
    #     folder_path="pretrain_data",           # your project dir
    #     # commit_message="Upload pretrain data",
    #     ignore_patterns=[".git", "__pycache__",".venv"]
    # )