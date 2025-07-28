import os
import subprocess
from tqdm import tqdm

# 设置输入和输出目录
xyz_directory = 'xyz'
mol_directory = 'mol'
os.makedirs(mol_directory, exist_ok=True)  # 确保输出目录存在

# 遍历所有XYZ文件并转换它们
for xyz_file in tqdm(os.listdir(xyz_directory)):
    if xyz_file.endswith('.xyz'):
        mol_file = xyz_file.replace('.xyz', '.mol')  # 创建MOL文件名
        xyz_path = os.path.join(xyz_directory, xyz_file)
        mol_path = os.path.join(mol_directory, mol_file)
        # 构建并运行Open Babel命令
        command = ['obabel', xyz_path, '-O', mol_path]
        subprocess.run(command)


