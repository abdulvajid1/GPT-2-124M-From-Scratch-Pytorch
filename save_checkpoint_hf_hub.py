from huggingface_hub import upload_folder, upload_large_folder, create_repo
from huggingface_hub import login
from dotenv import load_dotenv
import os


def main():
    load_dotenv()
    hf_token = os.getenv("HuggingFaceToken")
    if hf_token:
        login(hf_token)
    else:
        login()

    create_repo(repo_id='Abdulvajid/gpt2-dataset', repo_type='dataset', exist_ok=True)
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