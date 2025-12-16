# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "torch",
#   "torchvision",
#   "IPython",
#   "python-telegram-bot",
#   "asyncio",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///
# pip install python-dotenv python-telegram-bot asyncio

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime

import time
import os
import socket
from zoneinfo import ZoneInfo
import ast

import torch.profiler
############################ CONFIG ########################

#TODO cleanup

from dotenv import dotenv_values
config = { **dotenv_values(".env"), **os.environ }

# PASS_INPUT_TO_ALL_LAYERS=1 C_INIT="XAVIER_U" C_SPARSITY=10 G_SPARSITY=1 OPT_GATE16_CODEPATH=3 KINETO_LOG_LEVEL=99 GATE_ARCHITECTURE="[2000,2000]" INTERCONNECT_ARCHITECTURE="[]" PRINTOUT_EVERY=211 EPOCHS=300 uv run mnist.py
# CONNECTIVITY_GAIN=1 LEARNING_RATE=0.001 PASS_RESIDUAL=0 PASS_INPUT_TO_ALL_LAYERS=0 C_INIT="NORMAL" C_SPARSITY=100 C_INIT_PARAM=0 GATE_ARCHITECTURE="[250,250,250,250,250,250]"PRINTOUT_EVERY=55 VALIDATE_EVERY=211 EPOCHS=200 uv run mnist.py

LOG_TAG = config.get("LOG_TAG", "MNIST")
TIMEZONE = config.get("TIMEZONE", "UTC")

BINARIZE_IMAGE_TRESHOLD = float(config.get("BINARIZE_IMAGE_TRESHOLD", 0.75))
IMG_WIDTH = int(config.get("IMG_WIDTH", 34)) # 16 is suitable for Tiny Tapeout
IMG_CROP = int(config.get("IMG_CROP", 34))
INPUT_SIZE = IMG_WIDTH * IMG_WIDTH * 2

DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 1)) # USE VAL AS TEST FOR NOW (TODO)
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 10))
ONLY_USE_DATA_SUBSET = config.get("ONLY_USE_DATA_SUBSET", "0").lower() in ("true", "1", "yes")

SEED = config.get("SEED", random.randint(0, 1024*1024))
LOG_NAME = f"{LOG_TAG}_{SEED}"
GATE_ARCHITECTURE = ast.literal_eval(config.get("GATE_ARCHITECTURE", "[8000,8000]")) # previous: [1300,1300,1300] resnet95: [2000, 2000] conn_gain96: [2250, 2250, 2250] power_law_fixed97: [8000,8000,8000, 8000,8000,8000] auto_scale_logits_97_5 [8000]
INTERCONNECT_ARCHITECTURE = ast.literal_eval(config.get("INTERCONNECT_ARCHITECTURE", "[[],[1]]")) # previous: [[32, 325], [26, 52], [26, 52]])) resnet95, conn_gain96, auto_scale_logits_97_5: [] power_law_fixed97: [[],[-1],[-1], [-1],[-1],[-1]] p
if not INTERCONNECT_ARCHITECTURE or INTERCONNECT_ARCHITECTURE == []:
    INTERCONNECT_ARCHITECTURE = [[] for g in GATE_ARCHITECTURE]
assert len(GATE_ARCHITECTURE) == len(INTERCONNECT_ARCHITECTURE)
BATCH_SIZE = int(config.get("BATCH_SIZE", 256))

EPOCHS = int(config.get("EPOCHS", 30))
EPOCH_STEPS = math.floor((60_000 * TRAIN_FRACTION) / BATCH_SIZE) # MNIST consists of 60K images
TRAINING_STEPS = EPOCHS*EPOCH_STEPS
PRINTOUT_EVERY = int(config.get("PRINTOUT_EVERY", EPOCH_STEPS))
VALIDATE_EVERY = int(config.get("VALIDATE_EVERY", EPOCH_STEPS * 5))

LEARNING_RATE = float(config.get("LEARNING_RATE", 0.075))

SUPPRESS_PASSTHROUGH = config.get("SUPPRESS_PASSTHROUGH", "0").lower() in ("true", "1", "yes")
SUPPRESS_CONST = config.get("SUPPRESS_CONST", "0").lower() in ("true", "1", "yes")

PROFILE = config.get("PROFILE", "0").lower() in ("true", "1", "yes")
if PROFILE: prof = torch.profiler.profile(schedule=torch.profiler.schedule(skip_first=10, wait=3, warmup=1, active=1, repeat=1000), record_shapes=True, with_flops=True) #, with_stack=True, with_modules=True)
PROFILER_ROWS = int(config.get("PROFILER_ROWS", 20))

FORCE_CPU = config.get("FORCE_CPU", "0").lower() in ("true", "1", "yes")
COMPILE_MODEL = config.get("COMPILE_MODEL", "0").lower() in ("true", "1", "yes")

C_INIT = config.get("C_INIT", "NORMAL") # NORMAL, DIRAC
G_INIT = config.get("G_INIT", "NORMAL") # NORMAL, UNIFORM, PASSTHROUGH, XOR
C_SPARSITY = float(config.get("C_SPARSITY", 1.0)) # NOTE: 1.0 works well only for SHALLOW nets, 3.0 for deeper is necessary to binarize well
G_SPARSITY = float(config.get("G_SPARSITY", 1.0))

PASS_INPUT_TO_ALL_LAYERS = config.get("PASS_INPUT_TO_ALL_LAYERS", "0").lower() in ("true", "1", "yes") # previous: 1
PASS_RESIDUAL = config.get("PASS_RESIDUAL", "0").lower() in ("true", "1", "yes")

TAU = float(config.get("TAU", -1))

config_printout_keys = ["LOG_NAME", "TIMEZONE",
               "BINARIZE_IMAGE_TRESHOLD", "IMG_WIDTH", "INPUT_SIZE", "IMG_CROP", "DATA_SPLIT_SEED", "TRAIN_FRACTION", "NUMBER_OF_CATEGORIES", "ONLY_USE_DATA_SUBSET",
               "SEED", "GATE_ARCHITECTURE", "INTERCONNECT_ARCHITECTURE", "BATCH_SIZE",
               "EPOCHS", "EPOCH_STEPS", "TRAINING_STEPS", "PRINTOUT_EVERY", "VALIDATE_EVERY",
               "LEARNING_RATE",
               "C_INIT", "G_INIT", "C_SPARSITY", "G_SPARSITY",
               "PASS_INPUT_TO_ALL_LAYERS", "PASS_RESIDUAL",
               "TAU",
               "SUPPRESS_PASSTHROUGH", "SUPPRESS_CONST",
               "PROFILE", "FORCE_CPU", "COMPILE_MODEL"]
config_printout_dict = {key: globals()[key] for key in config_printout_keys}

############################ LOG ########################
log = print

############################ DEVICE ########################

try:
    device = torch.device(
                    "cuda" if torch.cuda.is_available()         and not FORCE_CPU else 
                    "mps"  if torch.backends.mps.is_available() and not FORCE_CPU else 
                    "cpu")
except:
    device = torch.device("cpu")

#################### TENSOR BINARIZATION ##################

def binarize_inplace(x, dim=-1, bin_value=1):
    ones_at = torch.argmax(x, dim=dim)
    x.data.zero_()
    x.data.scatter_(dim=dim, index=ones_at.unsqueeze(dim), value=bin_value)

############################ MODEL ########################

class FixedPowerLawInterconnect(nn.Module):
    def __init__(self, inputs, outputs, alpha, x_min=1.0, name=''):
        super(FixedPowerLawInterconnect, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.alpha = alpha

        max_length = inputs
        size = outputs
        r = torch.rand(size)
        if alpha > 1:
            magnitudes = x_min * (1 - r) ** (-1 / (alpha - 1))          # Power law distribution
            signs = torch.randint(low=0, high=2, size=(size,)) * 2 - 1  # -1 or +1
            offsets = magnitudes * signs * max_length
            indices = torch.arange(start=0, end=size) + offsets.long()
            indices = indices % max_length
        else:
            c       = torch.randperm(outputs) % inputs
            indices = torch.randperm(inputs)[c]
        self.register_buffer("indices", indices)
        self.binarized = False

    @torch.profiler.record_function("mnist::Fixed::FWD")
    def forward(self, x):
        return x[:, self.indices] if not self.binarized else torch.matmul(x, self.c)

        # Performance comparison
        # 1) x[:, self.indices]
        # MPS: 4.29 ms per iteration [300,300], tiny bit faster
        # MPS: 9.93 ms per iteration [3000,3000]
        # Takes significantly less memory though!

        # 2)
        # batch_size = x.shape[0]
        # if self.batch_indices.shape[0] != batch_size:
        #     self.batch_indices = self.indices.repeat(batch_size, 1)
        # return torch.gather(x, dim=1, index=self.batch_indices) 
        # MPS: 4.57 ms per iteration [300,300]
        # MPS: 9.32 ms per iteration [3000,3000], tiny bit faster

    def binarize(self, bin_value=1):
        with torch.no_grad():
            c = torch.zeros((self.inputs, self.outputs), dtype=torch.float32, device=device)
            c.scatter_(dim=0, index=self.indices.unsqueeze(0), value=bin_value)
            self.register_buffer("c", c)
            self.binarized = True

    def __repr__(self):
        with torch.no_grad():
            i = self.indices.view(self.outputs // 2, 2) # [batch_size, number_of_gates, 2]
            A = i[:,0]
            B = i[:,1]

            d = torch.abs(A-B)
            d = torch.minimum(d, self.inputs - d)
            return f"FixedPowerLawInterconnect({self.inputs} -> {self.outputs // 2}x2, α={self.alpha}, mean={d.float().mean().long()} median={d.float().median().long()})"

class LearnableInterconnect(nn.Module):
    def __init__(self, inputs, outputs, name=''):
        super(LearnableInterconnect, self).__init__()
        self.c = nn.Parameter(torch.zeros((inputs, outputs), dtype=torch.float32))
        if C_INIT == "DIRAC" or C_INIT == "UNIQUE":
            with torch.no_grad():
                nn.init.normal_(self.c, mean=0.0, std=.01) # add a very small amount of noise as a tie-breaker
                idx = torch.randperm(outputs) % inputs
                idx = torch.randperm(inputs)[idx]
                idx = idx.unsqueeze(0)
                self.c.data.scatter_add_(dim=0, index=idx, src=torch.ones_like(idx, dtype=torch.float32) * 10.0) # 10 here is inspired by the min/max range of a gaussian
        else:
            nn.init.normal_(self.c, mean=0.0, std=1)
        self.name = name
        self.binarized = False
    
    @torch.profiler.record_function("mnist::Sparse::FWD")
    def forward(self, x):
        connections = F.softmax(self.c * C_SPARSITY, dim=0) if not self.binarized else self.c
        return torch.matmul(x, connections)

    def binarize(self, bin_value=1):
        binarize_inplace(self.c, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableInterconnect({self.c.shape[0]} -> {self.c.shape[1] // 2}x2)"

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, name=''):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_gates * 2
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, self.number_of_gates), dtype=torch.float32)) # [16, W]
        self.zeros = torch.empty(0)
        self.ones = torch.empty(0)
        self.binarized = False
        nn.init.normal_(self.w, mean=0, std=1)
        if G_INIT == "UNIFORM":
            nn.init.uniform_(self.w, a=0.0, b=1.0)
        elif G_INIT == "PASS" or G_INIT == "PASSTHROUGH":
            with torch.no_grad(): self.w[3, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        elif G_INIT == "XOR":
            with torch.no_grad(): self.w[6, :] = 5 # 5 will roughly result in a 95% percent once softmax() applied
        else:
            nn.init.normal_(self.w, mean=0.0, std=1)

        # g0  = 
        # g1  =         AB
        # g2  =   A    -AB
        # g3  =   A
        # g4  =      B -AB
        # g5  =      B
        # g6  =   A  B-2AB
        # g7  =   A  B -AB
        # g8  = 1-A -B  AB
        # g9  = 1-A -B 2AB
        # g10 = 1   -B
        # g11 = 1   -B  AB
        # g12 = 1-A
        # g13 = 1-A     AB
        # g14 = 1      -AB
        # g15 = 1

        W = torch.zeros(4, 16, device=device)
        # 1 weights
        W[0, 8:16] =            1
        # A weights
        W[1, [2, 3,  6,  7]] =  1
        W[1, [8, 9, 12, 13]] = -1
        # B weights
        W[2, [4, 5,  6,  7]] =  1
        W[2, [8, 9, 10, 11]] = -1
        # A*B weights
        W[3, 1] =               1
        W[3, 6] =              -2
        W[3, [2, 4, 7,14]] =   -1
        W[3, [1, 8,11,13]] =    1
        W[3, 9] =               2 

        self.W16_to_4 = W

    @torch.profiler.record_function("mnist::LearnableGate16::FWD")
    def forward(self, x):
        # batch_size = x.shape[0]
        # x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        # A = x[:,:,0] # [batch_size, number_of_gates]
        # B = x[:,:,1] # [batch_size, number_of_gates]
        
        # weights_t = F.softmax(self.w * G_SPARSITY, dim=0).transpose(0,1) if not self.binarized else self.w.transpose(0,1)
        # weights = torch.matmul(weights_t, self.W16_to_4.transpose(0,1))  # [number_of_gates, 4]
        # result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=2)
        # return result.sum(dim=2)

        batch_size = x.shape[0]
        x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        A = x[:,:,0]          # [batch_size, number_of_gates]
        B = x[:,:,1]          # [batch_size, number_of_gates]
        
        weights = F.softmax(self.w * G_SPARSITY, dim=0) if not self.binarized else self.w
        weights = torch.matmul(self.W16_to_4, weights) # [4, number_of_gates]
        result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=1)
        return result.sum(dim=1)

    def binarize(self, bin_value=1):
        binarize_inplace(self.w, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableGate16Array({self.number_of_gates})"

class Model(nn.Module):
    def __init__(self, seed, gate_architecture, interconnect_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.gate_architecture = gate_architecture
        self.interconnect_architecture = interconnect_architecture
        self.first_layer_gates = self.gate_architecture[0]
        self.last_layer_gates = self.gate_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        layer_inputs = input_size
        R = [input_size]
        for layer_idx, (layer_gates, interconnect_params) in enumerate(zip(gate_architecture, interconnect_architecture)):
            if len(interconnect_params) == 1 and interconnect_params[0] > 0:
                interconnect = FixedPowerLawInterconnect(layer_inputs, layer_gates*2, alpha= interconnect_params[0],  name=f"i_{layer_idx}")
            else:
                interconnect = LearnableInterconnect    (layer_inputs, layer_gates*2,                                 name=f"i_{layer_idx}")
            layers_.append(interconnect)
            layers_.append(LearnableGate16Array(layer_gates, f"g_{layer_idx}"))
            layer_inputs = layer_gates
            R.append(layer_gates)
            if PASS_INPUT_TO_ALL_LAYERS:
                layer_inputs += input_size
            if PASS_RESIDUAL and (layer_idx > 0 or not PASS_INPUT_TO_ALL_LAYERS):
                layer_inputs += R[-2]
        self.layers = nn.ModuleList(layers_)

    @torch.profiler.record_function("mnist::Model::FWD")
    def forward(self, X):
        with torch.no_grad():
            S = torch.sum(X, dim=-1)
            self.log_input_mean = torch.mean(S).item()
            self.log_input_std = torch.std(S).item()
            self.log_input_norm = torch.norm(S).item()

        Xo = torch.zeros(X.shape[0],8000, dtype=torch.float32)
        Xo = Xo.to(device)
        for i in range (10):    #TODO - make intro variable
            Xc = X[:,i,:]
            for layer_idx in range(0, len(self.layers)):
                Xc = self.layers[layer_idx](Xc)
            Xo = Xo + Xc

        X = Xo
        # print ("TRAINING: ", self.training)
        #if not self.training:   # INFERENCE ends here! Everything past this line will only concern training
        #    return X            # Finishing inference here is both:
        #                        # 1) an OPTIMISATION and
        #                        # 2) it ensures no discrepancy between VALIDATION step during training vs STANDALONE inference

        X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)

        if TAU < 0:
            gain = np.sqrt(self.outputs_per_category / 6.0)
        else:
            gain = TAU

        with torch.no_grad():
            self.log_applied_gain = gain
            self.log_pregain_mean = torch.mean(X).item()
            self.log_pregain_std = torch.std(X).item()
            self.log_pregain_min = torch.min(X).item()
            self.log_pregain_max = torch.max(X).item()
            self.log_pregain_norm = torch.norm(X).item()
        X = X / gain

        with torch.no_grad():
            self.log_logits_mean = torch.mean(X).item()
            self.log_logits_std = torch.std(X).item()
            self.log_logits_norm = torch.norm(X).item()

        return X

    def clone_and_binarize(self, device, bin_value=1):
        model_binarized = Model(self.seed, self.gate_architecture, self.interconnect_architecture, self.number_of_categories, self.input_size).to(device)
        model_binarized.load_state_dict(self.state_dict())
        for layer in model_binarized.layers:
            if hasattr(layer, 'binarize') and callable(layer.binarize):
                layer.binarize(bin_value)
        return model_binarized

    def get_passthrough_fraction(self):
        pass_fraction_array = []
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                pass_fraction_array.append(pass_weight / total_weight)
        return pass_fraction_array
    
    def get_unique_fraction(self):
        unique_fraction_array = []
        with torch.no_grad():
            for model_layer in self.layers:
                # TODO: better to measure only unique indices that point to the previous layer and ignore skip connections:
                # ... = sum(unique_indices < previous_layer.c.shape[1])
                if hasattr(model_layer, 'c'):
                    unique_indices = torch.unique(torch.argmax(model_layer.c, dim=0)).numel()
                    unique_fraction_array.append(unique_indices / model_layer.c.shape[0])
                elif hasattr(model_layer, 'top_c') and hasattr(model_layer, 'top_indices'):
                    outputs = model_layer.top_c.shape[1]
                    top1 = torch.argmax(model_layer.top_c, dim=0)
                    indices = model_layer.top_indices[top1, torch.arange(outputs)]
                    max_inputs = indices.max().item() # approximation
                    unique_indices = torch.unique(indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
                elif hasattr(model_layer, 'indices'):
                    max_inputs = model_layer.indices.max().item() # approximation
                    unique_indices = torch.unique(model_layer.indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
        return unique_fraction_array

    def compute_selected_gates_fraction(self, selected_gates):
        gate_fraction_array = []
        indices = torch.tensor(selected_gates, dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                gate_fraction_array.append(pass_weight / total_weight)
        return torch.mean(torch.tensor(gate_fraction_array)).item()


### INSTANTIATE THE MODEL AND MOVE TO GPU ###

log(f"PREPARE MODEL on device={device}")
random.seed(SEED)
torch.manual_seed(SEED)
model = Model(SEED, GATE_ARCHITECTURE, INTERCONNECT_ARCHITECTURE, NUMBER_OF_CATEGORIES, INPUT_SIZE).to(device)
if COMPILE_MODEL:
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
log(f"model={model}")

############################ DATA ########################

# ### GENERATORS
# def transform():    
#     def binarize_image_with_histogram(image):
#         if BINARIZE_IMAGE_TRESHOLD > 0:
#             threshold = torch.quantile(image, BINARIZE_IMAGE_TRESHOLD)
#         else:
#             threshold = -BINARIZE_IMAGE_TRESHOLD
#         return (image > threshold).float()

#     return transforms.Compose([
#         transforms.CenterCrop((IMG_CROP, IMG_CROP)),
#         transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(-1)),
#         transforms.Lambda(lambda x: binarize_image_with_histogram(x))
#     ])

import inspect
import hashlib
import re

log(f"READ DATA")

### MOVE TRAIN DATASET TO GPU ###

### PWZ NEW CODE
train = torch.load("datasets/nmnist_train_tem10.pt")
train_images = train["data"]
train_dataset_samples = train["data"].shape[0]
train_labels = torch.empty((train_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

train_labels_ = torch.empty((train_dataset_samples), dtype=torch.long)
for i, label in enumerate(train["labels"]):
    train_labels_[i] = label
train_labels = torch.nn.functional.one_hot(train_labels_, num_classes=NUMBER_OF_CATEGORIES)
train_labels = train_labels.type(torch.float32)

train_images = train_images.to(device)
train_labels = train_labels.to(device)

### MOVE TEST DATASET TO GPU ###
test = torch.load("datasets/nmnist_test_tem10.pt")
test_images = test["data"]
test_dataset_samples = test["data"].shape[0]
test_labels = torch.empty((test_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

test_labels_ = torch.empty((test_dataset_samples), dtype=torch.long)
for i, label in enumerate(test["labels"]):
    test_labels_[i] = label
test_labels = torch.nn.functional.one_hot(test_labels_, num_classes=NUMBER_OF_CATEGORIES)
test_labels = test_labels.type(torch.float32)

test_images = test_images.to(device)
test_labels = test_labels.to(device)

### MOVE VAL DATASET TO GPU ###

val = torch.load("datasets/nmnist_test_tem10.pt")
val_images = val["data"]
val_dataset_samples = val["data"].shape[0]
val_labels = torch.empty((val_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

val_labels_ = torch.empty((val_dataset_samples), dtype=torch.long)
for i, label in enumerate(val["labels"]):
    val_labels_[i] = label
val_labels = torch.nn.functional.one_hot(val_labels_, num_classes=NUMBER_OF_CATEGORIES)
val_labels = val_labels.type(torch.float32)

val_images = val_images.to(device)
val_labels = val_labels.to(device)

### VALIDATE ###

def get_validate(default_model):
    def validate(dataset="val", model=default_model):
        if dataset == "val":
            number_of_samples = val_dataset_samples
            sample_images = val_images
            sample_labels = val_labels
        elif dataset == "test":
            number_of_samples = test_dataset_samples
            sample_images = test_images
            sample_labels = test_labels
        elif dataset == "train":
            number_of_samples = train_dataset_samples
            sample_images = train_images
            sample_labels = train_labels
        else:
            raise IOError(f"Unknown dataset {dataset}")
        val_loss = 0.0
        val_steps = 0
        correct = 0
        for start_idx in range(0, number_of_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, number_of_samples)
            val_indices = torch.arange(start_idx, end_idx, device=device)    
            x_val = sample_images[val_indices]
            y_val = sample_labels[val_indices]
            with torch.no_grad():
                val_output = model(x_val)
                val_loss += F.cross_entropy(val_output, y_val, reduction="sum").item()
                correct += (val_output.argmax(dim=1) == y_val.argmax(dim=1)).sum().item()
            val_steps += len(x_val)
        assert val_steps == number_of_samples
        val_loss /= val_steps
        val_accuracy = correct / val_steps
        return val_loss, val_accuracy
    return validate

def get_binarized_model(model=None, bin_value=1):
    return model.clone_and_binarize(device, bin_value)

def l1_topk(weights_after_softmax, k=4, special_dim=0): # but goes to 1 when binarized; 0 when uniform
    # test
    # t1 = torch.zeros([8,32,75]); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get zero
    # t1 = torch.rand(8, 32, 75); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get almost zero
    # ones_at = torch.argmax(t1, dim=1); t1.zero_(); t1.scatter_(dim=1, index=ones_at.unsqueeze(1), value=100); l1_topk(F.softmax(t1,dim=1),special_dim=1) # should get 1
    other_dims_prod = torch.prod(torch.tensor([x for i, x in enumerate(weights_after_softmax.shape) if i != special_dim]))
    normalization_factor = (weights_after_softmax.shape[special_dim]-k)/weights_after_softmax.shape[special_dim] * other_dims_prod # in case of a uniform tensor
    top_k_values, _ = torch.topk(weights_after_softmax, k, dim=special_dim)
    top_k_sum = top_k_values.sum(dim=special_dim, keepdim=True)
    non_top_k_sum = (1 - top_k_sum).sum()
    return 1. - non_top_k_sum / normalization_factor

### TRAIN ###

validate = get_validate(model)
val_loss, val_accuracy = validate(dataset="val")
passthrough_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_passthrough_fraction()])
unique_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_unique_fraction()])
log(f"INIT VAL loss={val_loss:6.3f} acc={val_accuracy*100:6.2f}%                 - Pass {passthrough_log} | Connectivity {unique_log}")

log(f"EPOCH_STEPS={EPOCH_STEPS}, will train for {EPOCHS} EPOCHS")
optimizer = torch.optim.Adam (model.parameters(), lr=LEARNING_RATE)
time_start = time.time()

if PROFILE: prof.start()
for i in range(TRAINING_STEPS):
    start_idx = (i * BATCH_SIZE) % train_dataset_samples
    end_idx = min(start_idx + BATCH_SIZE, train_dataset_samples)
    if start_idx < BATCH_SIZE:
        all_indices = torch.randperm(train_dataset_samples, device=device)
    indices = all_indices[start_idx: end_idx]    

    x = train_images[indices]
    y = train_labels[indices]
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        for l in model.layers:
            if hasattr(l,'w'):
                if SUPPRESS_CONST:
                    for const_gate_ix in [0,15]:
                        l.w.data[const_gate_ix, :] = 0
                if SUPPRESS_PASSTHROUGH:
                    for pass_gate_ix in [3, 5, 10, 12]:
                        l.w.data[pass_gate_ix, :] = 0

        model_output = model(x)
        loss = F.cross_entropy(model_output, y)
        loss.backward()
        optimizer.step()

    # TODO: model.eval here perhaps speeds everything up?
    if (i + 1) % PRINTOUT_EVERY == 0:
        passthrough_log = " ".join([f"{value * 100:2.0f}" for value in model.get_passthrough_fraction()])
        unique_log = " ".join([f"{value * 100:2.0f}" for value in model.get_unique_fraction()])
        # grad_log = ", ".join([f"{value.grad.std():.0e}" for value in model.parameters()])
        # grad_log = ", ".join([f"{value.grad.std():2.3e}" for value in model.parameters()])
        grad_log = " ".join([f"{torch.log10(value.grad.std())*10:2.0f}" for value in model.parameters()])
        # inputs_log = f"μ{model.log_input_mean:.2f}±{model.log_input_std:.2f}"
        # self.log_input_norm = torch.norm(X).item()
        # logits_log = f"μ{model.log_pregain_mean:.0f}±{model.log_pregain_std:.1f}*{1.0/model.log_applied_gain:0.3f} v{model.log_pregain_min:.0f}^{model.log_pregain_max:.0f}= μ{model.log_logits_mean:.0f}±{model.log_logits_std:.1f} ‖{model.log_logits_norm:.0f}‖"
        logits_log = f"{model.log_pregain_min:.0f}..{model.log_pregain_max:.0f} μ{model.log_pregain_mean:.0f}±{model.log_pregain_std:.1f}/{model.log_applied_gain:.1f} = ±{model.log_logits_std:.1f} ‖{model.log_logits_norm:.0f}‖"
        log(f"Iteration {i + 1:10} - Loss {loss:6.3f} - Pass (%) {passthrough_log} | Conn (%) {unique_log} | e-∇x10 {grad_log} | Logits {logits_log}")
    if (i + 1) % VALIDATE_EVERY == 0 or (i + 1) == TRAINING_STEPS:
        current_epoch = (i+1) // EPOCH_STEPS

        train_loss, train_acc = validate('train')
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TRN loss={train_loss:.3f} acc={train_acc*100:.2f}%")
        model_binarized = get_binarized_model(model)
        _, bin_train_acc = validate(dataset="train", model=model_binarized)
        train_acc_diff = train_acc-bin_train_acc
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS} BIN TRN            acc={bin_train_acc*100:.2f}%, train_acc_diff={train_acc_diff*100:.2f}%")

        test_loss, test_acc = validate('test')
        test_acc_diff = train_acc-test_acc
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TST            acc={test_acc*100:.2f}%, test_acc_diff= {test_acc_diff*100:.2f}%, loss={test_loss:.3f}")
        
    if PROFILE:
        torch.cpu.synchronize()
        if   device.type == "cuda": torch.cuda.synchronize()
        elif device.type == "mps":  torch.mps.synchronize()
    if PROFILE: prof.step()
if PROFILE: prof.stop()


time_end = time.time()
training_total_time = time_end - time_start 
log(f"Training took {training_total_time:.2f} seconds, per iteration: {(training_total_time) / TRAINING_STEPS * 1000:.2f} milliseconds")

test_loss, test_acc = validate('test')
log(f"    TEST loss={test_loss:.3f} acc={test_acc*100:.2f}%")

model_binarized = get_binarized_model(model)
bin_test_loss, bin_test_acc = validate(dataset="test", model=model_binarized)
log(f"BIN TEST loss={bin_test_loss:.3f} acc={bin_test_acc*100:.2f}%")

model_filename = (
    f"{datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y%m%d-%H%M%S')}"
    f"_binTestAcc{round(bin_test_acc * 10000)}"
    f"_seed{SEED}_epochs{EPOCHS}_{len(GATE_ARCHITECTURE)}x{GATE_ARCHITECTURE[0]}"
    f"_b{BATCH_SIZE}_lr{LEARNING_RATE * 1000:.0f}"
    f"_interconnect.pth"
)

with torch.no_grad():
    model_binarized.eval()
    include_dataset = {'dataset_input': val_images, 'dataset_output': model_binarized(val_images)}
    print(model_binarized(val_images).shape)
# torch.save(model.state_dict(), model_filename) #!!!
torch.save(model_binarized.state_dict() | include_dataset, model_filename) #!!!
log(f"Saved to {model_filename}")

if PROFILE:
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=PROFILER_ROWS))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=PROFILER_ROWS))
    if device.type == "cuda":
        print("-"*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=PROFILER_ROWS))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=PROFILER_ROWS))
    prof.export_chrome_trace(f"{LOG_NAME}_profile.json")
