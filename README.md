# conferencing-speech-2022

## 1. Overview
Source code for LCN submission for ConferencingSpeech2022 challenge.

**RMSE Results (lower is better):**
|                                  | Baseline1 | Baseline2 |    Ours    |
|----------------------------------|:---------:|:---------:|:----------:|
| Validation RMSE (All Corpora)    |   0.8224  |   0.5475  | **0.5000** |
| Validation RMSE (PSTN & Tencent) |   0.6614  |   0.4965  | **0.4759** |
| Test RMSE (PSTN & Tencent)       |   0.745   |   0.543   |  **0.344** |

**UPDATE:** Also have a look at our other repository for better models and easy inference. \
https://github.com/lcn-kul/xls-r-analysis-sqa

## 2. Installation
*Note: this software was developed for Linux.*

**Clone Repository**
```
git clone https://github.com/lcn-kul/conferencing-speech-2022.git
cd conferencing-speech-2022
```

**Google Drive Credentials**

To download the IU Bloomington from Google Drive, you need to download your
Google Drive API credentials.

1. Follow [these
   instructions](https://cloud.google.com/iam/docs/creating-managing-service-accounts#creating)
   (under "Creating a service account") to create a Google Cloud project and a
   Service Account for this project.
2. Next, create a Service Key by following [these
   instructions](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating)
   (under "Create a service account key"). You will download a JSON file containing
   your Google credentials. Place this JSON file in the following location:
   ```
   conferencing-speech-2022/gdrive_creds.json
   ```
3. Go to [this
   link](https://console.developers.google.com/apis/library/drive.googleapis.com)
   to enable Google Drive API for this project.
4. Wait up to 5 minutes for changes to propagate through Google systems.

## 3. Reproducing Results

Run the following commands to reproduce the results.

**Edit 07/04/2022:** The TUB test data is not available at the moment, so I
have commented out the code related to the test set (download, extract, shards,
predict).

**1. Create Virtual Environment**
```
make create_environment
source venv/bin/activate
make requirements
```

**2. Download Datasets**
```
make download
```

**3. Extract MFCC/XLS-R Features (GPU Recommended)**
```
make features
```

**4. Create Data Shards for Better I/O Speed**
```
make shards
```

**5. Calculate Norm/Variance**
```
make norm
```

**6. Train Models (GPU Recommended)**
```
make train
```

**7. Predict Models on Validation Set(s)**
```
make predict
```

**8. Predict Final Model on Test Set**

**Edit:** commented out implementation (see above)

```
make predict_submission
```

**9. Follow the [README file](src/eval/README.md) in the `src/eval/` folder to copy
the ground-truth files and prediction files to the correct locations.**

**10. Evaluate Models on Validation Set(s)**
```
make eval
```

## 4. Citation

Tamm, B., Balabin, H., Vandenberghe, R., Van hamme, H. (2022) Pre-trained Speech Representations as Feature Extractors for Speech Quality Assessment in Online Conferencing Applications. Proc. Interspeech 2022, 4083-4087, doi: 10.21437/Interspeech.2022-10147

```
@inproceedings{tamm22_interspeech,
  author={Bastiaan Tamm and Helena Balabin and Rik Vandenberghe and Hugo {Van hamme}},
  title={{Pre-trained Speech Representations as Feature Extractors for Speech Quality Assessment in Online Conferencing Applications}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={4083--4087},
  doi={10.21437/Interspeech.2022-10147}
}
```
