import perceval as pcvl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import merlin as ML
from tqdm import tqdm 

def create_circuit(m, d, depth):
    # m: number of modes
    # d: input dimension, d <= m
    # depth: number of layers
    
    c = pcvl.Circuit(m)

    # first layer
    w = pcvl.GenericInterferometer(m,
                                            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{0}i{i}")) // \
                                                      pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{0}o{i}")),
                                            shape=pcvl.InterferometerShape.TRIANGLE)
        
    # data encoding
    c_var = pcvl.Circuit(m)
    c_var.add(0, pcvl.Barrier(m), merge=True)
    for i in range(d):
        px = pcvl.P(f"px{i + 1}")
        c_var.add(i + (m - d) // 2, pcvl.PS(px))
    c_var.add(0, pcvl.Barrier(m), merge=True)

    # inputs
    params = c_var.get_parameters()
    
    for j in range(1, depth):
        # circuit creation
        w = pcvl.GenericInterferometer(m,
                                            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{j}i{i}")) // \
                                                      pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{j}o{i}")),
                                            shape=pcvl.InterferometerShape.TRIANGLE)
        
        # data encoding
        c_var = pcvl.Circuit(m)
        c_var.add(0, pcvl.Barrier(m), merge=True)
        for i in range(d):
            px = params[i]
            c_var.add(i + (m - d) // 2, pcvl.PS(px))
        c_var.add(0, pcvl.Barrier(m), merge=True)
       
        c.add(0, w, merge=True)
        c.add(0, c_var, merge=True)

    # last layer
    w = pcvl.GenericInterferometer(m,
                                            lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{depth}i{i}")) // \
                                                      pcvl.BS() // pcvl.PS(pcvl.P(f"theta_{depth}o{i}")),
                                            shape=pcvl.InterferometerShape.TRIANGLE)
    c.add(0, w, merge=True)
    return c

class hybridModel(nn.Module):
    def __init__(self, m, d):
        super().__init__()
        self.n_classes = 10
        
        self.depth = 2
        self.circuit = create_circuit(m, d, self.depth)
        self.thetas = [p.name for p in self.circuit.get_parameters() if not p.name.startswith("px")]
        input_state = [1, 0] * (m // 2) + [0] * (m % 2)
        self.input_state = pcvl.BasicState(input_state)
        self.q_output_dim = 10
        self.qlayer = ML.QuantumLayer(
                        input_size=d,
                        output_size=self.q_output_dim,
                        circuit=self.circuit,
                        trainable_parameters=["theta"],                            # Which parameters to train
                        input_parameters=["px"],
                        no_bunching=True,
                        input_state = self.input_state,
                        output_mapping_strategy=ML.OutputMappingStrategy.GROUPING)
        self.hidden_dim = 100
        self.clayer = nn.Linear(d, self.hidden_dim)
        self.post_proc = nn.Linear(self.q_output_dim + self.hidden_dim, self.n_classes)

    def forward(self, x):
        x_c = F.relu(self.clayer(x)) # shape [batch_size, dim=10]
        x_q = self.qlayer(x) # shape [batch_size, dim=10]
        x_cat = torch.cat((x_c, x_q), dim=1)
        x = self.post_proc(x_cat)
        return x
    
def training(model, trainloader, testloader, n_epochs, batch_size, lr, X_train, y_train,verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(range(n_epochs), disable= not verbose)
    train_accuracies = []
    test_accuracies = []

    # before training
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    train_epoch_acc, test_epoch_acc = [], []
    with torch.no_grad():
        for batch_X, batch_y in testloader:
            train_outputs = model(X_train)
            train_preds = torch.argmax(train_outputs, dim=1).numpy()
            train_acc = accuracy_score(y_train.numpy(), train_preds)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_accuracies.append(train_acc)
        test_accuracies.append(correct/total)
    
    for epoch in pbar:
        model.train()
        train_loss = 0
        for batch_X, batch_y in trainloader:
            # Zero the gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
    
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
    
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        train_epoch_acc, test_epoch_acc = [], []
        with torch.no_grad():
            for batch_X, batch_y in testloader:
                train_outputs = model(X_train)
                train_preds = torch.argmax(train_outputs, dim=1).numpy()
                train_acc = accuracy_score(y_train.numpy(), train_preds)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
    
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                

        train_accuracies.append(train_acc)
        test_accuracies.append(correct/total)
        if verbose:
            print(f'Train Loss: {train_loss / len(trainloader):.4f} | '
                  f'Train Acc: {train_acc*100:.2f}% | '
                  f'Test Loss: {test_loss / len(testloader):.4f} | '
                  f'Test Acc: {100 * correct / total:.2f}%')
        
    return np.array(train_accuracies), np.array(test_accuracies)