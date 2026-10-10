# Brain Tumour Classification

[![Tests](https://github.com/bhuvannv13/Brain_Tumour_classification_api/actions/workflows/tests.yml/badge.svg)](https://github.com/bhuvannv13/Brain_Tumour_classification_api/actions/workflows/tests.yml)

Deep learning experiments for detecting and classifying brain tumours from MRI scans, with a Streamlit app for trying a trained model on your own images.

> Educational project. Not a medical device and not suitable for clinical use.

## What's in this repo

| File | Purpose |
|---|---|
| `Copy_of_F&N__CNNBrainTumorCHI.ipynb` | Custom CNN that classifies MRI scans into four classes: glioma, meningioma, pituitary tumour and no tumour. Images are resized to 150x150. |
| `Brain Tumour Detection.ipynb` | Binary tumour / no-tumour detection using transfer learning with ResNet50 (224x224 input). |
| `BrainTumor_DataAugumentation.ipynb`, `BrainTumor_DataAugumentation_2.ipynb` | Image preprocessing: Gaussian blur, median filter and negative filter applied class by class to the training and testing folders. |
| `app.py` | Streamlit app with background information on brain tumours and an upload page for predictions. |
| `inference.py` | Model loading and image preprocessing, separate from the UI so it can be tested. |
| `tests/` | Pytest suite for preprocessing and the saved model; runs on every push via GitHub Actions. |
| `best_cnnmodel_1.h5` | Saved Keras model. |
| `archive.zip` | Small binary MRI dataset (`yes` / `no` folders, 253 images) used by the detection notebook. |

## Data

- **Four-class model:** [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) from Kaggle, with `Training` and `Testing` folders containing `glioma_tumor`, `meningioma_tumor`, `no_tumor` and `pituitary_tumor`.
- **Binary model:** the `yes` / `no` MRI images in `archive.zip`.

## Results

Figures below are taken from the saved notebook outputs.

| Model | Task | Result |
|---|---|---|
| Custom CNN | 4-class classification | 0.80 accuracy, 0.77 macro F1 on 2,258 test images |
| ResNet50 transfer learning | Tumour / no tumour | About 0.83 to 0.88 validation accuracy on a very small validation set (24 images) |

The binary dataset is small, so its numbers should be read as indicative only.

## Getting started

```bash
git clone https://github.com/bhuvannv13/Brain_Tumour_classification_api.git
cd Brain_Tumour_classification_api
pip install -r requirements.txt
```

The notebooks were written in Google Colab and read data from Google Drive. Before running them locally, change the dataset paths at the top of each notebook to point at your own copy of the data.

```bash
jupyter notebook
```

To launch the app, which loads `best_cnnmodel_1.h5` from this folder:

```bash
streamlit run app.py
```

## Tests

```bash
pip install pytest
python -m pytest -v
```

The tests check that uploads are resized to 150x150 and converted to the BGR channel order the model was trained with, that greyscale and transparent images are handled, and that the saved model returns one probability per class.

## Preprocessing filters

- **Gaussian blur** (5x5 kernel) reduces noise and fine detail.
- **Median filter** (kernel size 5) removes noise while preserving edges.
- **Negative filter** inverts intensities to change contrast.

## Possible improvements

- Transfer learning (for example EfficientNet) for the four-class task
- A patient-level train/test split and cross-validation
- Grad-CAM visualisations to show which regions drive each prediction

## Author

Bhuvann Vinodh Ram ([@bhuvannv13](https://github.com/bhuvannv13))

## License

MIT. See [LICENSE](LICENSE).
