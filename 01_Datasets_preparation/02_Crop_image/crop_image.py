import os
import numpy as np
import cv2

"""
Crop MoS2 images collected by optical microscope and resize them to 512*512 pixels
"""


def crop_image(image_dir, size, save_path):

    file_path_list = []
    for filename in os.listdir(image_dir):
        file_path = image_dir + '/' + filename
        file_path_list.append(file_path)
    for counter, image_path in enumerate(file_path_list):

        om_img = cv2.imread(image_path)
        if(om_img is None):
            print("Null!")
        else:
            # print("count:{}".format(counter))
            h = np.shape(om_img)[0]
            w = np.shape(om_img)[1]
            h_no = h // size
            w_no = w // size
            i = 0
            for row in range(0, h_no):
                for col in range(0, w_no):
                    i = i+1
                    cropped_img = om_img[size * row: size * (row + 1), size * col: size * (col + 1), :]
                    # print("name:{}".format(save_path + '/' + "MoS2_" + str(15*(counter-1)+i) + ".png"))
                    cv2.imwrite(save_path + '/' + "MoS2_" + str(15*(counter-1)+i) + ".png",
                               cropped_img)

if __name__ == '__main__':
    image_dir = "./Origin_image"  # dir path of origin image 
    save_path = "./crop_image"  # dir path of crop image
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    size = 512
    crop_image(image_dir, size, save_path)
