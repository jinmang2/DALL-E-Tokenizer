from .configuration_dalle imoprt DallEConfig

from .modeling_dalle import (
  Conv2d, 
  EncoderBlock, 
  DallEPreTrainedModel,
  DallEEncoder,
)

DALLETokenizer = DallEEncoder
