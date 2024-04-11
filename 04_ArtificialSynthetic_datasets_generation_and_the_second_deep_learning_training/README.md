## Project Introduction:
* [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [https://github.com/pytorch/vision](https://github.com/pytorch/vision)

## Environment Configuration:
* Python3.6/3.7/3.8
* Pytorch1.10
* For detailed environment configuration, see ```requirements.txt```

## File Structure:
```
  ├── datasets_path: Path to store generated dataset files
  ├── save_weights: Path to save trained model weights
  ├── Triangle_Generation.py: Script file for generating artificial dataset
  ├── train.py: Script file for training
  └── predict.py: A simple prediction script that utilizes trained weights for testing
```


## Training Method:
* Ensure the dataset is prepared beforehand.
* Make sure to perform predictions only after obtaining the trained weight files.
* If using a single GPU or CPU for training, utilize the train.py training script.

