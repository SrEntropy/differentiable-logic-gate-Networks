import torch
import random
import cv2
import os
import numpy as np


"""
    -dataset: OurMNIST-DVS
    -Iterates through OurMNIST-DVS 
    -128x128 
    -Split data and stores into a single training/testing tensor file.
    -Output: DVS-MNIST-split/train.pt,test.pt
"""




# ---- CONFIG ----
SRC_ROOT = "OurMNIST-DVS"           # raw npy folder
DST_ROOT = "DVS-MNIST-split"     # split dataset
MAX_TIME = 100_000
TRAIN_RATIO = 0.8
SEED = 42

# ---- SETUP ----
os.makedirs(DST_ROOT, exist_ok=True)
random.seed(SEED)

train_samples, train_labels = [], []
test_samples,  test_labels  = [], []

for label_str in sorted(os.listdir(SRC_ROOT)):
    label_path = os.path.join(SRC_ROOT, label_str)
    if not os.path.isdir(label_path):
        continue

    label = int(label_str)

    files = [f for f in os.listdir(label_path) if f.endswith('.npy')]
    random.shuffle(files)

    split_idx = int(len(files) * TRAIN_RATIO)
    train_files = files[:split_idx]
    test_files  = files[split_idx:]

    for split_files, is_train in [
        (train_files, True),
        (test_files, False)
    ]:
        for file in split_files:
            file_path = os.path.join(label_path, file)
            events = np.load(file_path)

            all_x = events[0, :].astype(int)
            all_y = events[1, :].astype(int)
            all_ts = events[2, :]
            all_p = events[3, :].astype(int)

            all_ts -= all_ts[0]  # normalize

            image = np.zeros((64,64,2),dtype=np.float32)

            mask = all_ts < MAX_TIME

            for x, y, p in zip(all_x[mask], all_y[mask], all_p[mask]):
                image[int(y/2),int(x/2),p] = 1.0

            # pooled = image[:,:.0].reshape(64, 2, 64, 2).sum(axis=(1, 3))
            # output[0] = (pooled >= 2).astype(int)

            # pooled = image[:,:,1].reshape(64, 2, 64, 2).sum(axis=(1, 3))
            # output[1] = (pooled >= 2).astype(int)

            # cv2.imshow("neg", image[:,:,0])
            # cv2.imshow("pos", image[:,:,1])
            # cv2.waitKey()
            # break

            vector = image.flatten()
            tensor_image = torch.tensor(vector, dtype=torch.float32)
            tensor_label = torch.tensor(label, dtype=torch.long)

            if is_train:
                train_samples.append(tensor_image)
                train_labels.append(tensor_label)
            else:
                test_samples.append(tensor_image)
                test_labels.append(tensor_label)

# Save all in one file per split
torch.save({'data': torch.stack(train_samples),
            'labels': torch.stack(train_labels)},
           os.path.join(DST_ROOT, 'train.pt'))

torch.save({'data': torch.stack(test_samples),
            'labels': torch.stack(test_labels)},
           os.path.join(DST_ROOT, 'test.pt'))

print(f"Saved: train_custom.pt with {len(train_samples)} samples")
print(f"Saved: test_custom.pt with {len(test_samples)} samples")