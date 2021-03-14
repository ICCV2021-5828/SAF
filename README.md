# Shuffle Augmentation of Features (SAF)

The source code for ICCV 2021 paper submission "Shuffle Augmentation of Features from Unlabled Data for UDA".

The code is implemented with PyTorch.

## Requirements
The required python libraries are listed in the file `requirements.txt`.


## Train
To run the training, first download the [VisDA-2017](https://github.com/VisionLearningGroup/taskcv-2017-public) dataset and prepare the file list using the script `image_list/image_list_utils.py`.
The training script will find `image_list/visda-2017-train.txt` and `image_list/visda-2017-validation.txt` for data loading.

Then execute the following command:
```bash
./run.sh
```
and the training will begin.
