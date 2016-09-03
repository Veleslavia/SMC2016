#Automatic musical instrument recognition in audiovisual recordings by combining image and audio classification strategies : SMC 2016

This is a code sources for our paper at SMC 2016 conference dedicated to the comparison of audio-based and image-based 
strategies for musical instrument recognition.

##Requirements

Python requirement can be found in `requirements.txt` file.

##Dataset requirements

In order to reproduce results presented in the paper, please, download the following datasets:

- [ImageNet](http://image-net.org) dataset (synsets n02672831, n02787622, n02992211, n03110669, n03249569, n03372029, n03467517, n03838899, n03928116, n04141076, n04487394, n04536866)
- [IRMAS](http://www.mtg.upf.edu/download/datasets/irmas) dataset
- [RWC Music Database: Musical Instrument Sound](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html)

##Step-by-step Guide

### Audio-based approach

#### Feature extraction

This step is optional. You can use features provided in `./audio/irmas/irmas_essentia_features.csv` and `./audio/rwc/rwc_essentia_features.csv` files
In order to extract features with [Essentia](http://essentia.upf.edu/) library, run for each dataset

```bash
python ./audio/feature_extraction.py data_directory_path output_file_path 
```

#### Training classifiers

We perform 10-fold cross-validation for audio. The parameter `dataset_name` can be only RWC or IRMAS

SVM classification
```bash
python ./audio/svm_classification.py path_to_features_file.csv dataset_name
```

XGBoost classification
```bash
python ./audio/xgb_classification.py path_to_features_file.csv dataset_name
```

The trained classifier stores at the same directory as a .plk file for the following cross-evaluation on other datasets. 
To reproduce the test results, please, save a label encoder additionally or use the encoder provided `./audio/irmas_le.pkl` and `./audio/rwc_le.pkl`.

### Image-based approach

In order to reproduce fine-tuning, be sure, that you have ImageNet subset stored at `./../dataset/images` or change IMAGES_DIR variable in `./utils/settings/py`
You also need to download pretrained weights for [VGG-16 model](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl) and store it at `./image/cnnnet` folder.

Then run 

```bash
python ./image/train_classify.py 
```

The fine-tuning will perform 5 epoch, display intermediate results and store the new weights for each epoch in separated .pkl file.

##Reference

- Olga Slizovskaia, Emilia Gomez & Gloria Haro (2016, September). "Automatic musical instrument recognition in audiovisual recordings by combining image and audio classification strategies" in 13th Sound and Music Computing Conference (SMC), Hamburg, Germany.
