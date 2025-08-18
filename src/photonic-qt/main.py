import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
from datetime import datetime
import time
import torch.optim as optim
import perceval as pcvl
from boson_sampler import BosonSampler
import numpy as np
import matplotlib.pyplot as plt
from utils import MNIST_partial
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'TorchMPS'))
from torchmps import MPS
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def setup_session():
    session = None
    if session is not None:
        session.start()
    return session

def create_boson_samplers(session):
    bs_1 = BosonSampler(m=9, n=4, postselect=0, session=session)
    print(f"Boson sampler defined with number of parameters = {bs_1.nb_parameters}, and embedding size = {bs_1.embedding_size}")
    pcvl.pdisplay(bs_1.create_circuit())
    
    bs_2 = BosonSampler(m=8, n=4, postselect=0, session=session)
    print(f"Boson sampler defined with number of parameters = {bs_2.nb_parameters}, and embedding size = {bs_2.embedding_size}")
    pcvl.pdisplay(bs_2.create_circuit())
    
    print(126 * 70)
    return bs_1, bs_2

class SharedWeightFC(nn.Module):
    def __init__(self, in_features, out_features, shared_rows):
        super(SharedWeightFC, self).__init__()
        self.shared_weights = nn.Parameter(torch.randn(shared_rows, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.out_features = out_features
        self.shared_rows = shared_rows

    def forward(self, x):
        weight_matrix = self.shared_weights.repeat(self.out_features // self.shared_rows, 1)
        return torch.matmul(x, weight_matrix.t()) + self.bias

class CNNModel(nn.Module):
    def __init__(self, use_weight_sharing=False, shared_rows=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 12, kernel_size=5)
        
        if use_weight_sharing:
            self.fc1 = SharedWeightFC(in_features=12*4*4, out_features=20, shared_rows=shared_rows)
            self.fc2 = nn.Linear(20, 10)
        else:
            self.fc1 = nn.Linear(12*4*4, 20)
            self.fc2 = nn.Linear(20, 10)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def create_datasets():
    train_dataset = MNIST_partial(split='train')
    val_dataset = MNIST_partial(split='val')
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader, batch_size

def apply_pruning(model, amount=0.3):
    """Apply structured pruning to convolutional and fully connected layers."""
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
            print(f"Applied structured pruning to Conv2d layer {name} with {amount * 100:.1f}% filters pruned.")
        elif isinstance(layer, nn.Linear):
            prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=1)
            print(f"Applied structured pruning to Linear layer {name} with {amount * 100:.1f}% neurons pruned.")

def remove_pruning(model):
    """Remove pruning masks to finalize the reduced model."""
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(layer, 'weight')
            except ValueError:
                pass  # Layer was not pruned

class MaskedAdam(torch.optim.Adam):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad_mask = param.data != 0
                param.grad.data.mul_(grad_mask)
        super().step(closure)

def train_classical_cnn(train_loader, val_loader, num_epochs, use_pruning=False, pruning_amount=0.5, use_weight_sharing=False, shared_rows=10):
    learning_rate = 1e-3
    
    model = CNNModel(use_weight_sharing=use_weight_sharing, shared_rows=shared_rows)
    
    if use_pruning:
        apply_pruning(model, amount=pruning_amount)
        remove_pruning(model)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    if use_pruning:
        optimizer = MaskedAdam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_classical_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of parameters in classical CNN model: ", num_classical_parameter)
    
    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    model.eval()
    correct = 0
    total = 0
    loss_test_list = [] 
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_test = criterion(outputs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
    return model

def calculate_qubits(model):
    # Always use standard CNN architecture for quantum circuit parameter calculation
    # regardless of classical training method (weight sharing, pruning, etc.)
    standard_model = CNNModel(use_weight_sharing=False, shared_rows=10)
    
    numpy_weights = {}
    nw_list = [] 
    nw_list_normal = []
    for name, param in standard_model.state_dict().items():
        numpy_weights[name] = param.cpu().numpy()
    for i in numpy_weights:
        nw_list.append(list(numpy_weights[i].flatten()))
    for i in nw_list:
        for j in i:
            nw_list_normal.append(j)
    print("# of NN parameters for quantum circuit: ", len(nw_list_normal))
    n_qubits = int(np.ceil(np.log2(len(nw_list_normal))))
    print("Required qubit number: ", n_qubits)
    n_qubit = n_qubits
    return n_qubit, nw_list_normal

def probs_to_weights(probs_, model_template):
    new_state_dict = {}
    data_iterator = probs_.view(-1)
    
    for name, param in model_template.state_dict().items():
        shape = param.shape
        num_elements = param.numel()
        chunk = data_iterator[:num_elements].reshape(shape)
        new_state_dict[name] = chunk
        data_iterator = data_iterator[num_elements:]
        
    return new_state_dict

def generate_qubit_states_torch(n_qubit):
    all_states = torch.cartesian_prod(*[torch.tensor([-1, 1]) for _ in range(n_qubit)])
    return all_states

class PhotonicQuantumTrain(nn.Module):
    def __init__(self, n_qubit, bond_dim = 7):
        super().__init__()
        self.MappingNetwork = MPS(input_dim=n_qubit+1, output_dim=1, bond_dim=bond_dim)
    
    def forward(self, x, qnn_parameters, bs_1, bs_2, n_qubit, nw_list_normal):
        self.q_params_1 = qnn_parameters[:108]
        self.q_params_2 = qnn_parameters[108:]
        device = x.device
        
        res_1 = bs_1.run(parameters=self.q_params_1, samples=100000)
        trans_res_1 = bs_1.translate_results(res=res_1)
        trans_res_1 = trans_res_1/torch.mean(trans_res_1)
        probs_1 = trans_res_1.to(device)  
        
        res_2 = bs_2.run(parameters=self.q_params_2, samples=100000)
        trans_res_2 = bs_2.translate_results(res=res_2)
        trans_res_2 = trans_res_2/torch.mean(trans_res_2)
        probs_2 = trans_res_2.to(device)  

        probs_ = torch.ger(probs_1, probs_2).flatten().reshape(126 * 70,1)
        probs_ = probs_[:len(nw_list_normal)]
        probs_ = probs_.reshape(len(nw_list_normal),1)
        
        qubit_states_torch = generate_qubit_states_torch(n_qubit)[:len(nw_list_normal)]
        qubit_states_torch = qubit_states_torch.to(device)

        combined_data_torch = torch.cat((qubit_states_torch, probs_), dim=1)
        combined_data_torch = combined_data_torch.reshape(len(nw_list_normal), n_qubit+1)
        
        prob_val_post_processed = self.MappingNetwork(combined_data_torch)
        prob_val_post_processed = prob_val_post_processed - prob_val_post_processed.mean()
        
        # Always use standard CNN architecture for quantum training regardless of classical training method
        model_template = CNNModel(use_weight_sharing=False, shared_rows=10)
        state_dict = probs_to_weights(prob_val_post_processed, model_template)
            
        dtype = torch.float32
        
        conv1_weight = state_dict['conv1.weight'].to(device).type(dtype)
        conv1_bias = state_dict['conv1.bias'].to(device).type(dtype)
        conv2_weight = state_dict['conv2.weight'].to(device).type(dtype)
        conv2_bias = state_dict['conv2.bias'].to(device).type(dtype)
        fc1_weight = state_dict['fc1.weight'].to(device).type(dtype)
        fc1_bias = state_dict['fc1.bias'].to(device).type(dtype)
        fc2_weight = state_dict['fc2.weight'].to(device).type(dtype)
        fc2_bias = state_dict['fc2.bias'].to(device).type(dtype)
        
        x = F.conv2d(x, conv1_weight, conv1_bias, stride=1)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.conv2d(x, conv2_weight, conv2_bias, stride=1)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.linear(x, fc1_weight, fc1_bias)
        x = F.linear(x, fc2_weight, fc2_bias)
        return x

def train_quantum_model(qt_model, train_loader, train_loader_qnn, bs_1, bs_2, n_qubit, nw_list_normal, num_training_rounds, num_epochs):
    step = 1e-3
    gamma_lr_scheduler = 0.1
    q_delta = 2 * np.pi
    
    init_qnn_parameters = q_delta * np.random.rand(108+84)
    print(f"\n ---- QNN parameters of shape {init_qnn_parameters.shape} \n ----")

    qnn_parameters = init_qnn_parameters
    
    criterion = nn.CrossEntropyLoss()
    optimizer_mapping = optim.Adam(qt_model.parameters(), lr=step)
    
    num_trainable_params = sum(p.numel() for p in qt_model.parameters() if p.requires_grad)
    print("# of trainable parameter in Mapping model: ", num_trainable_params)
    print("# of trainable parameter in QNN model: ", bs_1.nb_parameters + bs_2.nb_parameters)
    print("# of trainable parameter in full model: ", num_trainable_params + bs_1.nb_parameters + bs_2.nb_parameters)
    
    loss_list = [] 
    loss_list_epoch = [] 
    acc_list_epoch  = [] 
    
    for round_ in range(num_training_rounds): 
        print("-----------------------")
        
        acc_list = [] 
        acc_best = 0
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # Training on regular batches
        for epoch in range(num_epochs):
            qt_model.train()
            train_loss = 0
            for i, (images, labels) in enumerate(train_loader):
                correct = 0
                total = 0
                since_batch = time.time()
                
                images, labels = images.to(device), labels.to(device)
                optimizer_mapping.zero_grad()
                outputs = qt_model(images, qnn_parameters=qnn_parameters, bs_1=bs_1, bs_2=bs_2, n_qubit=n_qubit, nw_list_normal=nw_list_normal)
                labels_one_hot = F.one_hot(labels, num_classes=10).float()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                
                loss_list.append(loss.cpu().detach().numpy())
                acc = 100 * correct / total
                acc_list.append(acc)
                train_loss += loss.cpu().detach().numpy()
                
                if acc > acc_best:
                    acc_best = acc
                
                loss.backward()
                optimizer_mapping.step()
                
                if (i+1) % 20 == 0:
                    print(f"Training round [{round_+1}/{num_training_rounds}], Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, batch time: {time.time() - since_batch:.2f}, accuracy:  {(acc):.2f}%")
            
            train_loss /= len(train_loader)

        # QNN parameter optimization using scipy minimize (like in ref.ipynb)
        num_batch_qnn = 1
        for batch_ in range(num_batch_qnn):
            train_iter = iter(train_loader_qnn)
            images, labels = next(train_iter)
            
            global qnn_train_step
            qnn_train_step = 0
            
            def qnn_minimize_loss(qnn_parameters_=None):
                global qnn_train_step
                
                correct = 0
                total = 0
                
                images_gpu, labels_gpu = images.to(device), labels.to(device)
                outputs = qt_model(images_gpu, qnn_parameters=qnn_parameters_, bs_1=bs_1, bs_2=bs_2, n_qubit=n_qubit, nw_list_normal=nw_list_normal)
                _, predicted = torch.max(outputs.data, 1)
                total += labels_gpu.size(0)
                correct += (predicted == labels_gpu).sum().item()
                loss = criterion(outputs, labels_gpu)
                loss_val = loss.cpu().detach().numpy()
                acc = 100 * correct / total
                
                qnn_train_step += 1
                if qnn_train_step % 100 == 0:
                    print(f"Training round [{round_+1}/{num_training_rounds}], qnn_train_step: [{qnn_train_step}/{1000}], loss: {loss_val}, accuracy: {acc} %")
                
                return loss_val
            
            # Use scipy minimize like in ref.ipynb
            init_param = qnn_parameters
            result = minimize(qnn_minimize_loss, init_param, method='COBYLA', options={'maxiter': 1000, 'adaptive': True})
            qnn_parameters = result.x
            
            # Update epoch metrics with final values
            epoch_loss = result.fun
            # Calculate final accuracy for this batch
            images_gpu, labels_gpu = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = qt_model(images_gpu, qnn_parameters=qnn_parameters, bs_1=bs_1, bs_2=bs_2, n_qubit=n_qubit, nw_list_normal=nw_list_normal)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels_gpu).sum().item()
                total = labels_gpu.size(0)
                epoch_acc = 100 * correct / total

        loss_list_epoch.append(epoch_loss)
        acc_list_epoch.append(epoch_acc)
    
    return qt_model, qnn_parameters, loss_list_epoch, acc_list_epoch

def plot_results(loss_list_epoch, acc_list_epoch):
    plt.plot([loss_i if isinstance(loss_i, (int, float)) else loss_i.cpu().detach() for loss_i in loss_list_epoch])
    plt.show()
    
    plt.plot(acc_list_epoch)
    plt.show()
    
    print([float(loss_i) if isinstance(loss_i, (int, float)) else float(loss_i.cpu().detach()) for loss_i in loss_list_epoch])
    print(acc_list_epoch)

def evaluate_model(qt_model, train_loader, val_loader, qnn_parameters, bs_1, bs_2, n_qubit, nw_list_normal):
    criterion = nn.CrossEntropyLoss()
    
    qt_model.eval()
    correct = 0
    total = 0
    loss_train_list = []
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = qt_model(images, qnn_parameters, bs_1, bs_2, n_qubit, nw_list_normal)
            loss_train = criterion(outputs, labels).cpu().detach().numpy()
            loss_train_list.append(loss_train)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the train set: {(100 * correct / total):.2f}%")
    print(f"Loss on the train set: {np.mean(loss_train_list):.2f}")

    qt_model.eval()
    correct = 0
    total = 0
    loss_test_list = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = qt_model(images, qnn_parameters, bs_1, bs_2, n_qubit, nw_list_normal)
            loss_test = criterion(outputs, labels).cpu().detach().numpy()
            loss_test_list.append(loss_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on the test set: {(100 * correct / total):.2f}%")
    print(f"Loss on the test set: {np.mean(loss_test_list):.2f}")
    print("Generalization error:", np.mean(loss_test_list) - np.mean(loss_train_list))

def parse_args():
    parser = argparse.ArgumentParser(description='Photonic Quantum Training')
    parser.add_argument('--bond_dim', type=int, default=7, help='Bond dimension for MPS (default: 7)')
    parser.add_argument('--num_training_rounds', type=int, default=2, help='Number of training rounds for quantum model (default: 2)')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training (default: 5)')
    parser.add_argument('--classical_epochs', type=int, default=5, help='Number of epochs for classical CNN training (default: 1)')
    parser.add_argument('--pruning', action='store_true', help='Enable pruning experiment')
    parser.add_argument('--pruning_amount', type=float, default=0.5, help='Pruning amount (default: 0.5)')
    parser.add_argument('--weight_sharing', action='store_true', help='Enable weight sharing experiment')
    parser.add_argument('--shared_rows', type=int, default=10, help='Number of shared rows for weight sharing (default: 10)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Running experiment with:")
    print(f"  Pruning: {args.pruning} (amount: {args.pruning_amount if args.pruning else 'N/A'})")
    print(f"  Weight sharing: {args.weight_sharing} (shared rows: {args.shared_rows if args.weight_sharing else 'N/A'})")
    
    session = setup_session()
    bs_1, bs_2 = create_boson_samplers(session)
    
    train_dataset, val_dataset, train_loader, val_loader, batch_size = create_datasets()
    
    model = train_classical_cnn(
        train_loader, val_loader, args.classical_epochs,
        use_pruning=args.pruning, pruning_amount=args.pruning_amount,
        use_weight_sharing=args.weight_sharing, shared_rows=args.shared_rows
    )
    
    n_qubit, nw_list_normal = calculate_qubits(model)
    
    qt_model = PhotonicQuantumTrain(n_qubit, bond_dim=args.bond_dim).to(device)
    
    batch_size_qnn = 1000
    train_loader_qnn = DataLoader(train_dataset, batch_size_qnn, shuffle=True)
    
    qt_model, qnn_parameters, loss_list_epoch, acc_list_epoch = train_quantum_model(
        qt_model, train_loader, train_loader_qnn, bs_1, bs_2, n_qubit, nw_list_normal, 
        args.num_training_rounds, args.num_epochs
    )
    
    plot_results(loss_list_epoch, acc_list_epoch)
    
    evaluate_model(qt_model, train_loader, val_loader, qnn_parameters, bs_1, bs_2, n_qubit, nw_list_normal)

if __name__ == "__main__":
    main()