# 使用该脚本批量处理经Batch_Conversion处理后文件夹中所需的数据集
# Batch processing of the dataset required in the folder after Batch_Conversion is used
import os
import shutil

def main(from_dir_path, to_dir_path):

   for dir_name in os.listdir('./labelme_json'):
      pic_name = dir_name[:-5]+'.png'
      from_dir = from_dir_path + '/' + dir_name + '/label.png'
      to_dir = from_dir_path + '/' + pic_name
      shutil.copyfile(from_dir, to_dir)


if __name__ == '__main__':
    from_dir_path = './labelme_json/'
    to_dir_path = './cv2_mask'
    if os.path.isdir(to_dir_path):
        os.makedirs(to_dir_path)
    main(from_dir_path, to_dir_path)

