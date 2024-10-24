import os

def clear_output_file(output_file):
    """清空输出文件的内容"""
    with open(output_file, 'w') as file:
        file.write('')  # 清空文件

def rename_images_to_numbers(folder_path, output_file):
    # 清空输出文件
    clear_output_file(output_file)
    
    # 获取文件夹中的所有文件名
    files = os.listdir(folder_path)
    
    # 只保留图片文件，假设图片是常见格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    images = [f for f in files if f.lower().endswith(image_extensions)]
    
    # 获取图片总数并计算所需的位数
    total_images = len(images)
    num_digits = len(str(total_images))  # 计算命名所需的位数

    # 按次序重命名为数字，使用格式化后的序号
    with open(output_file, 'a') as file:
        for i, image in enumerate(sorted(images), 1):
            # 获取文件的扩展名
            ext = os.path.splitext(image)[1]
            # 构造新的文件名，使用零填充的格式
            new_name = f"image{str(i).zfill(num_digits)}{ext}"
            # 获取完整的旧文件路径和新文件路径
            old_file = os.path.join(folder_path, image)
            new_file = os.path.join(folder_path, new_name)
            # 重命名文件
            os.rename(old_file, new_file)
            # 写入输出文件日志
            file.write(f"Renamed: {old_file} -> {new_file}\n")
            print(f"Renamed: {old_file} -> {new_file}")

# 示例用法
folder_path = './images/images_originales'  # 替换为你的文件夹路径
output_file = './images/text_infos/RenameImages_output_log.txt'  # 替换为你的输出日志文件路径
rename_images_to_numbers(folder_path, output_file)
