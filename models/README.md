# Models

The trained models will be stored in this directory.

Additionally, the submitted model is located here.
```
models/final_model_17mar_xlsr_blstm/best-epoch=012-val_loss=0.014164.ckpt
```

Also feel free to download the XLS-R model using the following commands.
```
cd conferencing-speech-2022/models
git lfs install
git clone https://huggingface.co/facebook/wav2vec2-xls-r-300m
```
This will take 523 MB of space. Downloading the model to this location will
allow it to be loaded locally. Otherwise, it will be downloaded from the
Hugging Face website before training.
