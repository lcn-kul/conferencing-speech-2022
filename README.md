# conferencing-speech-2022

## 1. Overview
Source code for LCN submission for ConferencingSpeech2022 challenge.

**RMSE Results (lower is better):**
|                                  | Baseline1 | Baseline2 |    Ours    |
|----------------------------------|:---------:|:---------:|:----------:|
| Validation RMSE (All Corpora)    |   0.8224  |   0.5475  | **0.5000** |
| Validation RMSE (PSTN & Tencent) |   0.6614  |   0.4965  | **0.4759** |
| Test RMSE (PSTN & Tencent)       |   0.768   |    TBD    |  **0.365** |


## 2. Installation
*Note: this software was developed for Linux.*

**Clone Repository**
```
git clone https://github.com/btamm12/conferencing-speech-2022.git
cd conferencing-speech-2022
```

**Google Drive Credentials**

To download the IU Bloomington from Google Drive, you need to download your
Google Drive API credentials.

1. Follow [these
   instructions](https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account)
   to create a Google Cloud project and Service Key.
   After following these instructions, you will have downloaded a JSON file
   containing your Google credentials. Place this JSON file in the following
   location:
   ```
   conferencing-speech-2022/gdrive_creds.json
   ```
2. Go to [this
   link](https://console.developers.google.com/apis/library/drive.googleapis.com)
   to enable Google Drive API for this project.
3. Wait 5 minutes for changes to propagate through Google systems.

## 3. Reproducing Results

Run the following commands to reproduce the results.

**1. Create Virtual Environment**
```
make create_environment
```

**2. Download Datasets**
```
make data
```

**3. Extract MFCC/XLS-R Features**
```
make features
```

**4. Create Data Shards for Better I/O Speed**
```
make shards
```

**5. Train Models (GPU Recommended)**
```
make train
```

**6. Predict Models on Validation Set(s)**
```
make predict
```

**7. Predict Final Model on Test Set**
```
make predict_submission
```

**8. Follow the [README file](src/eval/README.md) in the `src/eval/` folder to copy
the ground-truth files and prediction files to the correct locations.**

**9. Evaluate Models on Validation Set(s)**
```
make eval
```
