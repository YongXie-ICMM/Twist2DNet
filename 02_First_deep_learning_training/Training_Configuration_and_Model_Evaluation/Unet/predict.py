import os
import time
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from src import UNet
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet predict")
    # exclude background
    parser.add_argument("--num-classes", default=3, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--image-path", default="test.png", help="predicting image path")
    parser.add_argument("--weights-path", default="save_weights/unet_model.pth", help="segmentation weights path")
    parser.add_argument("--palette_path", default="./palette.json", help="Plot the predicted image RGB values")
    args = parser.parse_args()
    return args


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def get_segmentation_result(args):
    assert os.path.exists(args.weights_path), f"weights {args.weights_path} not found."
    assert os.path.exists(args.image_path), f"image {args.image_path} not found."
    assert os.path.exists(args.palette_path), f"palette {args.palette_path} not found."
    with open(args.palette_path, "rb") as f:
        palette_dict = json.load(f)
        palette = []
        for v in palette_dict.values():
            palette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(num_classes=args.num_classes + 1)

    # delete weights about aux_classifier
    weights_dict = torch.load(args.weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # load image
    original_img = Image.open(args.image_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(512),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension

    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        # This comment refers to drawing semantic segmentation images according to the palette.json file.
        mask.putpalette(palette)
        mask.save("./test_result.png")


def main():
    args = parse_args()
    get_segmentation_result(args)


if __name__ == '__main__':
    main()
