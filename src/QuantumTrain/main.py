import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TorchMPS'))
import warnings
warnings.filterwarnings("ignore")

from photonic_qt_utils import (
    setup_session,
    create_boson_samplers,
    calculate_qubits,
    PhotonicQuantumTrain,
    train_quantum_model,
    evaluate_model
)
from classical_utils import (
    create_datasets,
    train_classical_cnn
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def plot_results(loss_list_epoch, acc_list_epoch):
    plt.plot([loss_i if isinstance(loss_i, (int, float)) else loss_i.cpu().detach() for loss_i in loss_list_epoch])
    plt.show()
    
    plt.plot(acc_list_epoch)
    plt.show()
    
    print([float(loss_i) if isinstance(loss_i, (int, float)) else float(loss_i.cpu().detach()) for loss_i in loss_list_epoch])
    print(acc_list_epoch)

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