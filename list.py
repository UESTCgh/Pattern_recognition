import os

def save_directory_structure_to_md():
    # 获取当前工作目录
    current_dir = os.getcwd()
    md_filename = "directory_structure.md"

    with open(md_filename, "w", encoding="utf-8") as md_file:
        # 写入当前目录的标题
        md_file.write(f"# 📁 当前文件夹内容\n\n")
        md_file.write(f"**路径**: `{current_dir}`\n\n")

        # 获取目录结构并写入文件
        write_directory_structure(md_file, current_dir, prefix="")

    print(f"文件夹结构已成功保存到 {md_filename}")

def write_directory_structure(md_file, directory, prefix):
    """
    将目录结构写入到 md 文件中，以树形结构表示。

    Args:
    - md_file: 打开的文件对象
    - directory: 当前目录路径
    - prefix: 树形结构前缀
    """
    entries = [e for e in os.listdir(directory) if not e.startswith(".") and os.path.isdir(os.path.join(directory, e))]
    entries.sort()  # 排序，确保更美观

    entries_count = len(entries)

    for i, entry in enumerate(entries):
        # 判断是否是最后一个元素，用于绘制不同的树形结构符号
        connector = "└── " if i == entries_count - 1 else "├── "
        
        # 写入当前文件夹名称
        md_file.write(f"{prefix}{connector}{entry}/\n")

        # 确定下一级的前缀
        if i == entries_count - 1:
            new_prefix = prefix + "    "
        else:
            new_prefix = prefix + "│   "

        # 递归列出子目录，只递归到第三级
        if prefix.count("│") < 2:
            write_directory_structure(md_file, os.path.join(directory, entry), new_prefix)

if __name__ == "__main__":
    save_directory_structure_to_md()
