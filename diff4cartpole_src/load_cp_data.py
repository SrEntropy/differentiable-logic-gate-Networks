import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyCPSDataset(Dataset):
    def __init__(self, 
                 start, 
                 end, 
                 binarize=0,
                 norm=False,
                 norm_vec_path="D:\\zihao\\telluride_25\\cp_sim\\dataset\\Trial_14__17_08_2024_for_telluride\\Models\\Dense-7IN-32H1-32H2-1OUT-0\\normalization_vec_a.csv",
                 data_dir="D:\\zihao\\telluride_25\\cp_sim\\dataset\\Trial_14__17_08_2024_for_telluride\\Recordings\\Train\\"):
        """
        Args:
            root_dir (string): Directory with all the Experiment-XXX.csv files.
        """
        self.data = []
        self.norm_vec = torch.empty((), dtype=torch.float32)
        if os.path.exists(norm_vec_path):
            self.norm_vec = pd.read_csv(norm_vec_path, header=None).values.flatten()
            self.norm_vec = torch.tensor(self.norm_vec, dtype=torch.float32)
        else:
            print(f"File {norm_vec_path} does not exist.")
        # Loop through Experiment-091 to -120.csv
        for i in range(start, end):
            data_path = os.path.join(data_dir, f"Experiment-{i:03d}.csv")
            if os.path.exists(data_path):
                # Skip first 38 rows, use row 39 as header
                df = pd.read_csv(data_path, skiprows=38)
                # Select relevant input and output columns
                inputs = df[['angleD', 'angle_cos', 'angle_sin', 'position', 'positionD', 'target_equilibrium', 'target_position']].values
                outputs = df[['Q_calculated_offline']].values

                for x, y in zip(inputs, outputs):
                    x = torch.tensor(x, dtype=torch.float32)
                    self.data.append((x, torch.tensor(y, dtype=torch.float32)))
            else:
                print(f"File {data_path} does not exist.")

    def normalize_data(self):
        # Normalize the data using the normalization vector
        for i, (x, y) in enumerate(self.data):
            # print(x.shape, self.norm_vec.shape)
            x = x * self.norm_vec
            self.data[i] = (x, y)

    def binarize_data(self, binarize_input=False, binarize_output=False):
            # Iterate through the data and binarize if needed
            for i, (x, y) in enumerate(self.data):
                if binarize_input:
                    x_np = x.numpy().view(np.uint32)
                    # Convert each uint32 to bits
                    bits_list = []
                    for val in x_np:
                        bits = np.unpackbits(
                            np.array([val], dtype=np.uint32).view(np.uint8)
                        )
                        bits_list.append(bits)
                    # Concatenate to a single array of shape [224]
                    bits_concat = np.concatenate(bits_list, axis=0)
                    # Convert back to torch tensor if needed
                    bits_tensor = torch.from_numpy(bits_concat)
                    x = bits_tensor.type(torch.float32)
                if binarize_output:
                    y_np = y.numpy().view(np.uint32)
                    # Convert each uint32 to bits
                    bits_y_list = []
                    for val in y_np:
                        bits_y = np.unpackbits(
                            np.array([val], dtype=np.uint32).view(np.uint8)
                        )
                        bits_y_list.append(bits_y)
                    # Concatenate to a single array of shape [32]
                    bits_y_concat = np.concatenate(bits_y_list, axis=0)
                    bits_y_tensor = torch.from_numpy(bits_y_concat)
                    y = bits_y_tensor.type(torch.float32)

                self.data[i] = (x, y)
    
    def thermo_encode_data(self, num_bits_x=100, num_bits_y=1000):
        encoded_data = []
        for i, (x, y) in enumerate(self.data):
            x_scaled = (x + 1) / 2  
            thresholds = torch.linspace(0, 1, steps=num_bits_x, device=x.device)
            x_encoded = (x_scaled.unsqueeze(1) >= thresholds).float()
            x_encoded_flat = x_encoded.flatten()
            # For y, we can use a similar approach but with more bits
            y_scaled = (y + 1) / 2
            y_thresholds = torch.linspace(0, 1, steps=num_bits_y, device=y.device)
            y_encoded = (y_scaled.unsqueeze(1) >= y_thresholds).float()
            y_encoded_flat = y_encoded.flatten()
            encoded_data.append((x_encoded_flat, y_encoded_flat))
        self.data = encoded_data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# # Load normalization vector
# norm_vec_path = "D:\\zihao\\telluride_25\\cp_sim\\dataset\\Trial_14__17_08_2024_for_telluride\\Models\\Dense-7IN-32H1-32H2-1OUT-0\\normalization_vec_a.csv"  # update with your full path if needed
# # Example usage
# train_dir = "D:\\zihao\\telluride_25\\cp_sim\\dataset\\Trial_14__17_08_2024_for_telluride\\Recordings\\Train\\"  # replace with your actual directory path
# train_dataset = MyCPSDataset(start=91, end=121, norm_vec_path=norm_vec_path, data_dir=train_dir)
# # test_dir = "D:\\zihao\\telluride_25\\cp_sim\\dataset\\Trial_14__17_08_2024_for_telluride\\Recordings\\Test\\" 
# # test_dataset = MyCPSDataset(start=81, end=86, norm_vec_path=norm_vec_path, data_dir=test_dir)
# train_dataset.normalize_data()

# # Plot the distribution of all input features
# import matplotlib.pyplot as plt
# def plot_feature_distribution(dataset):
#     inputs = [x for x, _ in dataset]
#     inputs = torch.stack(inputs)
    
#     plt.figure(figsize=(12, 8))
#     for i in range(inputs.shape[1]):
#         plt.subplot(3, 3, i + 1)
#         plt.hist(inputs[:, i].numpy(), bins=50, alpha=0.7)
#         plt.title(f'Feature {i+1}')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.show()
# plot_feature_distribution(train_dataset)

# # Probe how many samples are in the dataset
# print(f"Number of samples in training dataset: {len(train_dataset)}")

# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# # Test loading
# for batch_inputs, batch_outputs in train_loader:
#     print(batch_inputs.shape, batch_outputs.shape)
#     break