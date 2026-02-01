import os
from datetime import datetime

def create_exp_folder():
    # Step 1: 创建run文件夹（如果不存在）
    if not os.path.exists("leo_run_transformer"):
        os.mkdir("leo_run_transformer")

    # Step 2: 创建train文件夹（如果不存在）
    train_folder = os.path.join("leo_run_transformer", "train")
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)

    # Step 3: 基于时间戳创建exp文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder_name = f"exp_{timestamp}"
    exp_folder = os.path.join(train_folder, exp_folder_name)
    
    os.mkdir(exp_folder)  # 创建exp文件夹
    weights_folder = os.path.join(exp_folder, "weights")
    os.mkdir(weights_folder)  # 创建weights文件夹
    
    return exp_folder, weights_folder  # 返回exp和weights文件夹路径


def create_val_exp_folder():
    # Step 1: 创建run文件夹（如果不存在）
    if not os.path.exists("run"):
        os.mkdir("run")

    # Step 2: 创建predict文件夹（如果不存在）
    predict_folder = os.path.join("run", "predict")
    if not os.path.exists(predict_folder):
        os.mkdir(predict_folder)

    # Step 3: 基于时间戳创建exp文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder_name = f"exp_{timestamp}"
    exp_folder = os.path.join(predict_folder, exp_folder_name)
    
    os.mkdir(exp_folder)  # 创建exp文件夹
    
    return exp_folder  # 返回新创建的文件夹路径