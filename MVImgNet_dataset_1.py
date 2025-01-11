import json
import cv2
import numpy as np
from torch.utils.data import Dataset

class MVImgNetDataset1(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/ubuntu/LabData/hanx/MVImgNet/MVImgNet_by_categories/7_prompt_1.json', 'rt') as f:
            self.data = json.load(f)  # 读取整个 JSON 文件为一个数组

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filenames = item['source']
        target_filenames = item['target']
        prompts = item['prompt']

        sources = []
        targets = []

        for source_filename, target_filename in zip(source_filenames, target_filenames):
            # 读取控制图像
            source = cv2.imread(source_filename)
            # 读取目标图像
            target = cv2.imread(target_filename)

            # 调整图像尺寸为512x512
            source = cv2.resize(source, (512, 512))
            target = cv2.resize(target, (512, 512))

            # OpenCV读取的图像是BGR格式，需要转换为RGB格式
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)



            # 将图像归一化到[0, 1]范围
            source = source.astype(np.float32) / 255.0

            # 将目标图像归一化到[-1, 1]范围
            target = (target.astype(np.float32) / 127.5) - 1.0

            sources.append(source)
            targets.append(target)

        return dict(jpg=targets, txt=prompts, hint=sources)

# # 示例用法
# if __name__ == "__main__":
#     dataset = MVImgNetDataset1()
#     print(f"Dataset size: {len(dataset)}")
#     sample = dataset[0]
#     print(f"Sample data: {sample}")
# 示例用法
if __name__ == "__main__":
    dataset = MVImgNetDataset1()
    print(f"Dataset size: {len(dataset)}")
    # 打印dataset的shape，第一个元素的shape，以及类型type
    print(f"Dataset shape: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    print(f"First item shape: {len(dataset[0]['jpg'])}, {len(dataset[0]['txt'])}, {len(dataset[0]['hint'])}")
    print(f"First item shape: {dataset[0]['jpg'][0].shape}, {dataset[0]['txt'][0]}, {dataset[0]['hint'][0].shape}")
    print(f"First item type: {type(dataset[0])}")

# Dataset size: 130
# Dataset shape: 130
# Dataset type: <class '__main__.MVImgNetDataset1'>
# First item shape: 2, 2, 2
# First item shape: (1080, 1080, 3), a black and gold electric stove on a counter, (1080, 1080, 3)
# First item type: <class 'dict'>