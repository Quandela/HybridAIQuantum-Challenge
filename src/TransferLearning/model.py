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

    def __init__(self,bs=None,bs_size=None,device='cpu',dropout=False,pos=False):
        """
        Definition of the *dressed* layout.
        """

        super().__init__()
        self.dropout = dropout
        self.pos = pos
        self.pre_net = nn.Linear(512, bs_size)
        # self.embedding1 = nn.Linear(bs.embedding_size, 10)
        # self.embedding2 = nn.Linear(bs.embedding_size, 10)  
        # self.embedding3 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding4 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding5 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding6 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding7 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding8 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding9 = nn.Linear(bs.embedding_size, 10) 
        # self.embedding10 = nn.Linear(bs.embedding_size, 10) 
        # self.drop1 = nn.Dropout(p=0.2)
        # self.drop2 = nn.Dropout(p=0.2)
        # self.drop3 = nn.Dropout(p=0.2)
        # self.drop4 = nn.Dropout(p=0.2)
        # self.drop5 = nn.Dropout(p=0.2)
        # self.drop6 = nn.Dropout(p=0.2)
        # self.drop7 = nn.Dropout(p=0.2)
        # self.drop8 = nn.Dropout(p=0.2)
        # self.drop9 = nn.Dropout(p=0.2)
        # self.drop10 = nn.Dropout(p=0.2)
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
            # if self.dropout:
            #     embs = self.drop1(embs)
            # print(embs.shape)
            x = nn.Linear(embs.shape[-1],128)(embs) 
            x = self.post_net(x)
            q_out.append(x)
        # print(len(q_out))
        q_out = torch.cat([q_out[i] for i in range(10)],dim=0)
        # print(q_out.shape)

        # for elem in pre_out:
        #     embs = self.bs.embed(elem,1000).unsqueeze(0)
        #     x = nn.Linear(embs.shape[-1],10)(embs) 
        #     q_out.append(x)
        # q_out = torch.cat([q_out[i] for i in range(10)],dim=0)
        # embs = self.bs.embed(pre_out[0],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop1(embs)
        # x = self.embedding1(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[1],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop2(embs)
        # x = self.embedding2(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[2],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop3(embs)
        # x = self.embedding3(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[3],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop4(embs)
        # x = self.embedding4(embs) 
        # q_out.append(x)
        
        # embs = self.bs.embed(pre_out[4],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop5(embs)
        # x = self.embedding5(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[5],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop6(embs)
        # x = self.embedding6(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[6],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop7(embs)
        # x = self.embedding7(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[7],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop8(embs)
        # x = self.embedding8(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[8],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop9(embs)
        # x = self.embedding9(embs) 
        # q_out.append(x)

        # embs = self.bs.embed(pre_out[9],1000).unsqueeze(0)
        # if self.dropout:
        #     embs = self.drop10(embs)
        # x = self.embedding10(embs) 
        # q_out.append(x)

        # q_out = torch.cat([q_out[i] for i in range(10)],dim=0)

        # if self.pos:
        #     return self.post_net(q_out)
        # else:
        #     return q_out
        return q_out
    
class MnistModel(nn.Module):
    # def __init__(self, device = 'cpu'):
    #     super().__init__()
    #     input_size = 28 * 28
    #     self.device = device
    #     self.linear1 = nn.Linear(input_size, 500)
    #     self.linear2 = nn.Linear(500, 300)
    #     self.linear3 = nn.Linear(300, 512)
    #     self.fc = nn.Linear(512,10)
    
    # def forward(self, xb):
    #     xb = xb.reshape(-1, 784)
    #     xb = self.linear1(xb)
    #     xb = self.linear2(xb)
    #     xb = self.linear3(xb)
    #     out = self.fc(xb)
    #     return(out)
    def __init__(self, device = 'cpu',minst=False):
        super().__init__()
        self.minst = minst
        self.device = device
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, emb = None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels) ## Calculate the loss
        acc = accuracy(out, labels)
        return loss, acc
    
    def validation_step(self, batch, emb =None):
        images, labels = batch
        images, labels = images.to(self.device), labels.to(self.device)
        if self.embedding_size:
            out = self(images, emb.to(self.device)) ## Generate predictions
        else:
            out = self(images) ## Generate predictions
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return({'val_loss':loss, 'val_acc': acc})
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return result['val_loss'], result['val_acc']

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