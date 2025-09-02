from transformers import PretrainedConfig

class GptConfig(PretrainedConfig):
    model_type = "gpt_custom"

    def __init__(
        self,
        d_model=768,
        context_len=1024,
        n_layers=12,
        vocab_size=50257,
        n_heads=12,
        device="cpu",
        intermidiate_size=None,
        n_epoch=5,
        batch_size=8,
        load_checkpoint=True,
        weight_decay=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.context_len = context_len
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.device = device
        self.intermidiate_size = intermidiate_size or d_model * 4
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.load_checkpoint = load_checkpoint
        self.weight_decay = weight_decay
