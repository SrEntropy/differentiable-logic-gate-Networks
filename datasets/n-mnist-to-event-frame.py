import numpy as np
import glob
import os
import torch
import cv2

# 1. Download dataset from https://www.garrickorchard.com/datasets/n-mnist
# 2. The actual resolution is 34x34 pixels
# 3. There is an average of 1500 events / 100 ms, with the maxium of 3279 events.
# 4. Parse the binary files with this code:
# TODO: Save data to Tensors, create DataLoader


directory = "/home/power-station/Repo/npc25-difflogic-4-control/datasets/N-MNIST/Train"

all_samples = []
labels = []
for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        print(path)
        for filename in os.listdir(path):

                # Read data from file
                file = os.path.join(path, filename)
                f = open(file, 'rb')
                raw_data = np.fromfile(f, dtype=np.uint8)
                f.close()

                # Parse data for binary files
                raw_data = np.uint32(raw_data)
                all_y = raw_data[1::5]
                all_x = raw_data[0::5]
                all_p = (raw_data[2::5] & 128) >> 7
                all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

                # Unpack data intro dictionary
                events = {}
                events['x'] = all_x
                events['y'] = all_y
                events['t'] = all_ts
                events['p'] = all_p

                # Normalize timestamps
                events['t'] = events['t'] - events['t'][0]

                # Filter events / create binary representation
                image = np.zeros((34,34,2))
                for i in range(len(events["t"])):
                        if(events["t"][i] < 100000):
                                image[events["y"][i],events["x"][i],events["p"][i]] = 1;
                        else:
                                break
                vector = image.flatten()
                tensor = torch.tensor(vector, dtype=torch.float32)
                all_samples.append(tensor)
                labels.append(torch.tensor(int(subdir), dtype=torch.float32))

samples_tensor = torch.stack(all_samples)
labels_tensor = torch.stack(labels)
print(samples_tensor.shape)
print(labels_tensor.shape)
torch.save({'data': samples_tensor, 'labels': labels_tensor}, 'nmnist_train_event_frames.pt')
