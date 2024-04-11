# U-Net(Convolutional Networks for Biomedical Image Segmentation)

## Project Introduction:
* [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

## Environment Configuration:
* Python3.6/3.7/3.8
* Pytorch1.10
* For detailed environment configuration, see ```requirements.txt```

## File Structure:
```
  ├── src: The backbone of the model and the construction of U-NET
  ├── train_utils: Modules related to training, validation, and GPU training
  ├── my_dataset.py: Customized dataset for reading VOC dataset
  ├── train.py: Example training using deeplabv3_resnet50
  └── predict.py: A simple prediction script that uses trained weights for testing
```


## Training Method:
* Make sure to prepare the dataset in advance.
* Make sure to download the corresponding pretrained model weights in advance.
* If using a single GPU or CPU for training, simply use the train.py training script.

## Notes:
* When using the training script, make sure to set --data-path (VOC_root) to the root directory 
  where your VOCdevkit folder is located.
* When using the prediction script, make sure to set weights_path to the path of the generated weights that you want to use.
