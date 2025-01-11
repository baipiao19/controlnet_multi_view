import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/ubuntu/LabData/xiaoyw/ControlNet/training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/home/ubuntu/LabData/xiaoyw/ControlNet/training/fill50k/' + source_filename)
        target = cv2.imread('/home/ubuntu/LabData/xiaoyw/ControlNet/training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


# 示例用法
if __name__ == "__main__":
    dataset = MyDataset()
    print(f"Dataset size: {len(dataset)}")
    # 打印dataset的shape，第一个元素的shape，以及类型type
    print(f"Dataset shape: {len(dataset)}")
    print(f"Dataset type: {type(dataset)}")
    print(f"First item shape: {dataset[0]['jpg'].shape}, {dataset[0]['txt']}, {dataset[0]['hint'].shape}")
    print(f"First item type: {type(dataset[0])}")

# Dataset size: 50000
# Dataset shape: 50000
# Dataset type: <class '__main__.MyDataset'>
# First item shape: (512, 512, 3), pale golden rod circle with old lace background, (512, 512, 3)
# First item type: <class 'dict'>