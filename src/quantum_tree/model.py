import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Linear(128 * 2 * 2, output_size)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class ResBlock(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.GroupNorm(4, out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if (stride != 1) or (in_planes != out_planes):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)

        out = F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)


        self.layer1 = nn.Sequential(
            ResBlock(16, 16, stride=1),
            ResBlock(16, 16, stride=1),
            ResBlock(16, 16, stride=1),
        )

        self.layer2 = nn.Sequential(
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32, stride=1),
            ResBlock(32, 32, stride=1),
        )

        self.layer3 = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64, stride=2),
            # ResBlock(64, 64, stride=1),
        )

        self.layer4 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            # ResBlock(128, 256, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.post_conv = nn.Conv2d(128, 128, kernel_size=1, bias=True)
        self.post_bn = nn.GroupNorm(16, 128)

        self.proj = nn.Linear(128*2*2, output_size)

        self.init_weights()

    def init_weights(self, w0=30.0):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in = nn.init._calculate_correct_fan(m.weight, mode='fan_in')
                bound = math.sqrt(6.0 / fan_in) / w0
                nn.init.uniform_(m.weight, -bound, bound)
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.post_conv(x)
        x = self.post_bn(x)
        x = F.relu(x, inplace=True)

        x = x.view(x.size(0), -1)
        return self.proj(x)

class CNNBaseline(nn.Module):
    def __init__(self, output_size = 10):
        super().__init__()
        self.net = nn.Sequential(
            ResNet(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size),
        )
    
    def forward(self, x):
        return self.net(x)

    def evaluation(self, x):
        return self.net(x)

class QNN(nn.Module):
    def __init__(self, bs, device):
        super().__init__()
        self.hidden_dim = bs.m
        self.n_params = bs.nb_parameters
        self.bs = bs
        self.device = device

        self.param_proj = nn.Sequential(
            CNN(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.n_params)
        )
        self.param_proj = torch.compile(self.param_proj)
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 10)
        )
        self.dropout = nn.Dropout(0.1)
        
        self.surrogate = nn.Sequential(
            nn.Linear(self.n_params, 256),
            nn.ReLU(), 
            nn.Linear(256, 256),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(256, self.hidden_dim),
        )

    def quantum(self, features):
        with torch.no_grad():
            # features = features - features.min()
            # features = features / features.max()
            features = features.cpu()
            results = []
            for feature in features:
                r = self.bs.embed(feature, 1000)
                results.append(r)
            return torch.stack(results, dim=0).to(self.device)
        
    def forward(self, x):
        params = self.param_proj(x)
        # params = params - params.min()
        # params = params / params.max()
        measured = self.quantum(params)
        # print(measured.detach().cpu().tolist())
        # print(measured.shape)
        approximated = self.surrogate(params)

        return self.out_proj(approximated), F.mse_loss(approximated, measured), self.out_proj(measured)
    
    def evaluation(self, x):
        params = self.param_proj(x)
        measured = self.quantum(params)
        return self.out_proj(measured)