import os
import shutil


def copy_and_rename_png_images_alphabetically(source_dir, target_dir):
    # 创建目标文件夹如果它不存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取所有png文件，并按文件名字母顺序排序
    png_files = [file for file in os.listdir(source_dir) if file.endswith('.bmp')]
    png_files.sort()  # 按字母顺序排序

    # 遍历排序后的文件列表并复制每个文件，然后重命名为数字序列
    for index, file_name in enumerate(png_files):
        source_file_path = os.path.join(source_dir, file_name)
        target_file_path = os.path.join(target_dir, f"{index}.bmp")

        # 复制并重命名文件
        shutil.copy(source_file_path, target_file_path)
        print(f"Copied {source_file_path} to {target_file_path}")


# 使用示例
# source_directory = '/media/hjk/HardDisk/DataSet/Sonar/image'
# target_directory = '/media/hjk/HardDisk/DataSet/Sonar/image1'
# copy_and_rename_png_images_by_order(source_directory, target_directory)

if __name__ == '__main__':
        # 指定输入图像文件夹和输出图像文件夹
    source_directory = '/media/wby/shuju/Seg_Water/Under/SUIM/test/masks'
    target_directory = '/media/wby/shuju/Seg_Water/Under/SUIM/test/mask'

    # 调用图像大小调整函数
    copy_and_rename_png_images_alphabetically(source_directory, target_directory)