from transformers import PretrainedConfig


class DallEConfig(PretrainedConfig):
    def __init__(
        self,
        group_count: int = 4,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
        device: str = 'cpu',
        requires_grad: bool = False,
        use_mixed_precision: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        assert input_channels >= 1
        assert n_hid >= 64
        assert n_blk_per_group >= 1
        assert vocab_size >= 512
        
        self.group_count = group_count
        self.n_hid = n_hid
        self.n_blk_per_group = n_blk_per_group
        self.input_channels = input_channels
        self.vocab_size = vocab_size
        self.device = device
        self.requires_grad = requires_grad
        self.use_mixed_precision = use_mixed_precision
