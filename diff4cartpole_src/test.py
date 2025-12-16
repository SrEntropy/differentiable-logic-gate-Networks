import torch

class ThermoEncoder:
    def __init__(self, data):
        """
        data: list of (x, y) tuples
        x: Tensor of shape (input_dim,), values in [-1, 1]
        y: any label (e.g., int or tensor)
        """
        self.data = data

    def thermo_encode_data(self, num_bits=100):
        encoded_data = []

        for i, (x, y) in enumerate(self.data):
            x_scaled = (x + 1) / 2  # [-1,1] → [0,1]
            thresholds = torch.linspace(0, 1, steps=num_bits, device=x.device)
            
            x_unsqueezed = x_scaled.unsqueeze(1)
            print(x_unsqueezed)
            print(thresholds)
            x_encoded = (x_scaled.unsqueeze(1) >= thresholds).float()
            print(x_encoded)
            x_encoded_flat = x_encoded.flatten()

            encoded_data.append((x_encoded_flat, y))

        self.data = encoded_data

    def print_encoded(self):
        for i, (x, y) in enumerate(self.data):
            print(f"Sample {i}:")
            print(f"  Encoded x (len={len(x)}): {x}")
            print(f"  Label y: {y}\n")

# ==== TEST ====

if __name__ == "__main__":
    # Create toy dataset: 3 samples, 2 features each, values in [-1, 1]
    dummy_data = [
        (torch.tensor([-1.0, 0.0]), 0),
        (torch.tensor([0.5, -0.5]), 1),
        (torch.tensor([1.0, 1.0]), 2),
    ]

    num_bits = 5
    encoder = ThermoEncoder(dummy_data)
    encoder.thermo_encode_data(num_bits=num_bits)
    encoder.print_encoded()

    # Optional: Check shape
    x0, _ = encoder.data[0]
    expected_len = len(dummy_data[0][0]) * num_bits
    assert x0.shape[0] == expected_len, f"Expected length {expected_len}, got {x0.shape[0]}"
    print("✅ Encoding test passed.")
