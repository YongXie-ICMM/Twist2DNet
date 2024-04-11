# Identification and Structural Characterization of Twisted Bilayer Transition Metal Dichalcogenides by Deep Learning


## Project Introduction:
    This is a public repository for deep learning-based Identification and Structural Characterization of Twisted 
    Bilayer Transition Metal Dichalcogenides. The codes 
    were mainly developed by Mr. Haitao Yang (Email: haitaoyang2022@gmail.com),. 


## Environment Configuration:
* Python3.6/3.7/3.8
* Pytorch1.10
* For detailed environment configuration, see ```requirements.txt```


## File Structure:
```  
  ├── 01_Datasets_preparation:
  │   ├── 01_CVD_growth:
  │   ├── 02_Crop_image:
  │       ├── crop_image: Directory to save cropped 512x512 optical microscope images
  │       ├── Oringin_image: Directory containing original 2592x1944 optical microscope images
  │       └── crop_image.py: Script for batch cropping 2592x1944 optical microscope images
  │   ├── 03_04_Labelme_to_dataset:
  │       ├── labels: Directory containing JSON files annotated with labelme
  │           └── json_to_dataset.bat: Batch conversion of JSON files to dataset file format
  │       └── Extract_Image.py: Script for batch extraction of images in VOC format dataset
  │   └── 05_Datasets_augmentation_and_storage: Format of VOC-type file paths
  │       ├── VOCdevkit
  │           ├── VOC2007
  │                ├── ImageSets:
  │                    └── Segmentation: Directory containing txt files with names of training, validation, and test datasets
  │                ├── JPEGImages: Directory containing original 512x512 image data
  │                └── SegmentationClass: Directory containing transformed VOC dataset type category images
  │       └── Assign_Datasets.py: Script for automatic allocation of training, validation, and test datasets, saving txt files to ImageSets/Segmentation
  ├── 02_First_deep_learning_training: Code for semantic segmentation network models for layer number recognition. Refer to individual README.md files within each network for specific usage instructions.
  │         └──Training_Configuration_and_Model_Evaluation
  │                ├── DeepLabv3: Rethinking Atrous Convolution for Semantic Image Segmentation
  │                ├── FCN: Fully Convolutional Networks for Semantic Segmentation
  │                ├── LRASPP: Searching for MobileNetV3
  │                └── Unet: Convolutional Networks for Biomedical Image Segmentation
  ├── 03_OpenCV_calculation_for_twisted_bilayer_materials:
  │   └── get_twist_angle.py: Program for calculating image rotation based on layer recognition using OpenCV on MoS2 optical microscope images
  ├── 04_ArtificialSynthetic_datasets_generation_and_the_second_deep_learning_training: Regression analysis network model based on ResNet18 for predicting image rotation on MoS2 optical microscope images after layer recognition using semantic segmentation. Refer to its internal README.md file for specific usage instructions.
  ├── README.md: Documentation
  └── requirements.txt: Main environment requirements for project execution
```

