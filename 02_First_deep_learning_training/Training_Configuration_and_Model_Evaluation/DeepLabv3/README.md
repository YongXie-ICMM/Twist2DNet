# DeepLabV3(Rethinking Atrous Convolution for Semantic Image Segmentation)

# Project Introduction:
# This project is mainly from the source code of the official torchvision module of PyTorch.
* https://github.com/pytorch/vision/tree/main/torchvision/models/segmentation

## Environment Configuration:
* Python3.6/3.7/3.8
* Pytorch1.10
* For detailed environment configuration, see ```requirements.txt```

## File Structure:
```
  ├── src: The backbone of the model and the construction of DeepLabv3
  ├── train_utils: Modules related to training, validation, and GPU training
  ├── my_dataset.py: Customized dataset for reading VOC dataset
  ├── train.py: Example training using deeplabv3_resnet50
  ├── predict.py: A simple prediction script that uses trained weights for testing
  └── pascal_voc_classes.json: pascal_voc label file
```

## Pretrained Weight Download Links:
* Note: The official pretrained weights were trained on COCO dataset and then fine-tuned on 
  PASCAL VOC dataset, which has 21 classes (including background).
* deeplabv3_resnet50: https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
* Note that the downloaded pretrained weights should be renamed, e.g., in train.py, the script 
  reads `deeplabv3_resnet50_coco.pth`, not `deeplabv3_resnet50_coco-cd0a2569.pth`.

## Training Method:
* Make sure to prepare the dataset in advance.
* Make sure to download the corresponding pretrained model weights in advance.
* If using a single GPU or CPU for training, simply use the train.py training script.

## Notes:
* When using the training script, make sure to set --data-path (VOC_root) to the root directory 
  where your VOCdevkit folder is located.
* When using the prediction script, make sure to set weights_path to the path of the generated weights that you want to use.


