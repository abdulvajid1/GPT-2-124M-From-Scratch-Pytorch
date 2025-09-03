1. pip install uv
2. uv sync (remove triton windows if's it's there)
1. load checkpoint from huggingface (--download_checkpoint)
2. uv run download_dataset.py --hf_dataset='' for different data by difualt openwebtext
3. uv run train.py --load_checkpoint --load_checkpoint_path="ckpt_67" --download_checkpoint