from torch import Tensor, nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src import constants


class Wav2Vec2Wrapper(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Sampling rate.
        self.sampling_rate = 16_000

        # Construct preprocessing helper.
        self.helper = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # Create model.
        self.model = Wav2Vec2Model.from_pretrained(
            f"facebook/{constants.XLSR_NAME}")

    def forward(self, data: Tensor):
        # Model.
        inputs = self.helper(
            data,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        input = inputs["input_values"].to(data.device)
        return self.model(input)
