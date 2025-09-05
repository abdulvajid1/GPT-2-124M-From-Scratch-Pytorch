from huggingface_hub import upload_folder, upload_large_folder, create_repo
from huggingface_hub import login
from dotenv import load_dotenv
import os
import shutil
import pathlib
import glob

def save_to_hf(global_step):
    
    ckpt_path = os.path.join('checkpoints', 'ckpt_*')
    ckpt_list = set(glob.glob(ckpt_path))
    global_save_step = (global_step // 100) * 100
    
    ckpt_save = {os.path.join('checkpoints', f"ckpt_{global_save_step}")}
    rm_ckpt_list = ckpt_list.difference(ckpt_save)
    
    for ckpt in rm_ckpt_list:
        shutil.rmtree(ckpt)

    create_repo(repo_id='Abdulvajid/gpt2-from-scratch', repo_type='model', exist_ok=True)

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

if __name__ == "__main__":
    main()