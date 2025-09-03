1. pip install uv
2. pip install -r requirements.txt
3. pip install torch
3. python train.py --download_checkpoint --load_checkpoint='ckpt_200'
2. uv sync (remove triton windows if's it's there)
1. load checkpoint from huggingface (--download_checkpoint)
2. uv run download_dataset.py --hf_dataset='' for different data by difualt openwebtext
3. uv run train.py --load_checkpoint --load_checkpoint_path="ckpt_67" --download_checkpoint