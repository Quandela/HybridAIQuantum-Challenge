import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import time
import pandas as pd
from boson_sampler import BosonSampler
from utils import accuracy
import perceval as pcvl
import perceval.providers.scaleway as scw  # Uncomment to allow running on scaleway


class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self,bs=None,bs_size=None,device='cpu',dropout=False):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.dropout = dropout
        self.pre_net = nn.Linear(512, bs_size)
        self.drop = nn.Dropout(p=0.2)
        self.linear = nn.Linear(bs.embedding_size,128)
        self.bs = bs
        self.device = device
        self.post_net = nn.Linear(128, 10)

    def forward(self, input_features):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """

        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        # print(input_features.size())
        pre_out = self.pre_net(input_features)
        # print(pre_out.shape)
        q_out = []
        for elem in pre_out:
            # print(elem.shape)
            embs = self.bs.embed(elem,1000).unsqueeze(0)
            if self.dropout:
                embs = self.drop(embs)
            # print(embs.shape)
            x = self.linear(embs) 
            x = self.post_net(x)
            q_out.append(x)
        # print(len(q_out))
        q_out = torch.cat([q_out[i] for i in range(10)],dim=0)

        return q_out

# evaluation of the model
# evaluation of the model
def evaluate(model, val_loader, bs: BosonSampler = None):
    # if model.embedding_size:
    #     outputs = []
    #     for step, batch in enumerate(tqdm(val_loader)):
    #         # embedding in the BS
    #         images, labs = batch
    #         images = images.squeeze(0).squeeze(0)
    #         t_s = time.time()
    #         embs = bs.embed(images,1000)
    #         outputs.append(model.validation_step(batch, emb=embs.unsqueeze(0)))
    # else:
    outputs = [model.validation_step(batch) for batch in val_loader]
    return(model.validation_epoch_end(outputs))