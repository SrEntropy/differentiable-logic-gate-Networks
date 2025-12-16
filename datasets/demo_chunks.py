from metavision_core.event_io import EventsIterator
import cv2
import numpy as np
import math
import torch
import os
import random

# Path to your .hdf5 file
directory = "/home/pwz/Repo/MNIST-demo/"
SIZE = 64
TRESHOLD = 200
MIN_X = 576
MIN_Y = 296
TRAIN = 0.8

all_samples_train = []
labels_train = []
all_samples_test = []
labels_test = []

for filename in os.listdir(directory):
  if filename.endswith(".raw"):
    file = os.path.join(directory, filename)
    label = int(filename[0])
    print(file)
    print(label)
    # # if (len(file)) > 38:
    # #   MIN_X = 576
    # #   MIN_Y = 296
    # # else:
    # #   print("DUPA")
    # #   exit()
    # #   MIN_X = 0
    # #   MIN_Y = 0
    # print(MIN_X)
    # print(MIN_Y)
    events_iterator = EventsIterator(file)

    all_x = []
    all_y = []
    all_ts = []
    all_p = []
    for evs in events_iterator:
        for i in range (len(evs)):
          all_x.append(int(evs[i][0])-MIN_X)
          all_y.append(int(evs[i][1])-MIN_Y)
          all_ts.append(int(evs[i][3]))
          all_p.append(int(evs[i][2]))

    events = {}
    events['x'] = all_x
    events['y'] = all_y
    events['t'] = all_ts
    events['p'] = all_p

    # Filter events / create binary representation
    vector = np.zeros(SIZE*SIZE)
    #image = np.zeros((SIZE,SIZE))
    full_image = []
    for i in range(len(events["t"])):
            if math.floor(events["y"][i]/2) >= 64 or math.floor(events["y"][i]/2) < 0:
              #print("Y JEST ŹLE - %d", math.floor(events["y"][i]/2))
              continue
            if math.floor(events["x"][i]/2) >= 64 or math.floor(events["x"][i]/2) < 0:
              #print("X JEST ŹLE - %d", math.floor(events["x"][i]/2))
              continue
            vector[math.floor(events["y"][i]/2)*SIZE+(math.floor(events["x"][i]/2))] = 1
            #image[math.floor(events["y"][i]/2), (math.floor(events["x"][i]/2))] = 1
            if np.sum(vector) > TRESHOLD:
                    #cv2.imshow("label", image)
                    #cv2.waitKey()
                    
                    tensor = torch.tensor(vector, dtype=torch.float32)
                    full_image.append(tensor)
                    vector = np.zeros(SIZE*SIZE)
                    #image = np.zeros((SIZE,SIZE))
                    #continue
                    #continue
                    if (len(full_image) == 8):
                            if (random.random() <= TRAIN):
                                    all_samples_train.append(torch.stack(full_image))
                                    labels_train.append(torch.tensor(label))
                            else:
                                    all_samples_test.append(torch.stack(full_image))
                                    labels_test.append(torch.tensor(label))
                            full_image = []

train_tensor = torch.stack(all_samples_train)
test_tensor = torch.stack(all_samples_test)
labels_train_tensor = torch.stack(labels_train)
labels_test_tensor = torch.stack(labels_test)

print(train_tensor.shape)
print(test_tensor.shape)
print(labels_train_tensor.shape)
print(labels_test_tensor.shape)
torch.save({'data': train_tensor, 'labels': labels_train_tensor}, 'demo_5proc_train.pt')
torch.save({'data': test_tensor, 'labels': labels_test_tensor}, 'demo_5proc_test.pt')
