# this code presents the main SSL training loop and linear evaluation of a qSSL approach on MNIST dataset
import argparse
from tqdm import tqdm
import math
from merlin import QuantumLayer, OutputMappingStrategy
from utils import *

# parser
parser = argparse.ArgumentParser(description='SSL with Quantum Loss')
#data
parser.add_argument('-cl', '--classes', type=int, default=10, help='Number of classes')
parser.add_argument('-s', '--size', type=int, default=20, help='size x size of the image')
# training
parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument( '--ft-epochs', type=int, default=10, help='Number of epochs for fine tuning')

parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size')
# the SSL model
parser.add_argument('-bn', '--batch_norm', action='store_true', default=False,
                    help='Set if we use BatchNorm after compression of the encoder')
parser.add_argument('-bck-d', '--backbone-dim', type=int, default=64, help='Dimension of the backbone output')
parser.add_argument('--cnn', action='store_true', default=False, help='backbone is CNN if True, MLP otherwise')
parser.add_argument('--enc-dim', type=int, default=[8, 8], nargs='+',
                        help='Dimensions of the encoder')
# Contrastive Loss
parser.add_argument('-tau', '--temperature', type=float, default=0.07, help='Temperature of the InfoNCELoss')
# quantum SSL
parser.add_argument('-w', '--width', type=int, default=8, help='Dimension of the features encoded in the QNN')
parser.add_argument('-quant', '--quantum', action='store_true', default=False, help='Set if we use Quantum SSL')
parser.add_argument('-m', '--modes', type=int, default=10, help='Number of modes')
parser.add_argument('--no_bunching', action='store_true',
                        help='NoBunching mode for the quantumlayer')
parser.add_argument('--trained', action='store_true',
                        help='If Boson Layer is trained')
parser.add_argument('-d', '--datadir', type=str, default='./data', help='Data directory')
parser.add_argument('--no-plots', action='store_true', default=False, help='Disable plot display')




#####################
### the SSL model ###
#####################

#### Interferometer used for the loss
# - Backbone = classical
# - Projector = classical
# - Loss = QuantumLayer + similarity metrics
## pipeline: data -> backbone -> projector -> interferometer -> similarity loss



class SSL_q_loss(nn.Module):
    def __init__(self,
                 args,
                 backbone,
                 similarity = InfoNCELoss()
                 ):
        super().__init__()

        # backbone
        self.backbone = backbone

        self.backbone_features = args.backbone_dim

        # photonic circuit
        self.modes = args.modes
        self.no_bunching = args.no_bunching
        self.circuit = create_quantum_circuit(modes = self.modes, feature_size= self.backbone_features)
        self.quantum = args.quantum
        if self.quantum:
            input_state = [(i + 1) % 2 for i in range(self.modes)]
            photon_count = sum(input_state)
            print(f"\n ------------ Quantum Loss ------------ \n - input state: {input_state} \n - no_bunching: {self.no_bunching} "
                  f"\n - # parameters: {len([p.name for p in self.circuit.get_parameters() if not p.name.startswith("feature")])}"
                  f"\n --------------------------------------")
            self.projector = QuantumLayer(
                input_size=self.backbone_features,
                output_size=None, # but we do not use it
                circuit = self.circuit,
                trainable_parameters= [p.name for p in self.circuit.get_parameters() if not p.name.startswith("feature")],
                input_parameters=["feature"],
                input_state = input_state,
                no_bunching = self.no_bunching,
                output_mapping_strategy=OutputMappingStrategy.NONE)
            self.sig = nn.Sigmoid()
            self.criterion = similarity
        else:
            pseudo_output_size = math.comb(self.modes+self.modes//2 -1,self.modes//2) if self.no_bunching else math.comb(self.modes,self.modes//2)
            self.projector = nn.Linear(self.backbone_features, pseudo_output_size)
            self.criterion = InfoNCELoss(temperature = args.temperature)

        if not args.trained:
            self.projector.requires_grad_(False)

        print(
            f"\n -------------------- projector used -------------------- \n {self.projector} \n -------------------------------------------------")
    def forward(self, y1, y2):
        # first data augmentation
        z1 = self.backbone(y1)
        if self.quantum:
            # sigmoid activation so input falls in [0,1]
            z1 = self.sig(z1)
        x1 = self.projector(z1)

        # second data augmentation
        z2 = self.backbone(y2)
        if self.quantum:
            # sigmoid activation so input falls in [0,1]
            z2 = self.sig(z2)
        x2 = self.projector(z2)

        #comparing in the Hilbert space
        loss = self.criterion(x1, x2)

        return loss


##########################
### training functions ###
##########################

def training_step(model, train_loader, optimizer):
    pbar = tqdm(train_loader)
    total_loss = 0.0

    for x1, x2 in pbar:
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        #print(f"\n max X1 = {torch.max(x1)}")
        # Add channel dimension for CNN if needed
        if len(x1.shape) == 3 and x1.shape[-1] != 1:  # (batch, height, width)
            x1 = x1.unsqueeze(1)  # (batch, 1, height, width)
            x2 = x2.unsqueeze(1)  # (batch, 1, height, width)
        
        loss = model(x1, x2)

        # Check for NaN/inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)


def train(model, train_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    training_losses = []
    for epoch in range(args.epochs):
        loss = training_step(model, train_loader, optimizer)
        print(f"epoch: {epoch + 1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)
    return model, training_losses


def linear_evaluation(model, train_loader, val_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.ft_epochs):
        # training
        model.train()
        pbar = tqdm(train_loader)
        train_acc = 0
        train_loss_total = 0
        for img, target in pbar:
            #print(f"\n- Img MAX = {torch.max(img)}")
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item()
            train_acc += accuracy
            train_loss_total += loss.item()
            pbar.set_postfix({'Training Loss': f'{loss.item():.4f} - Training Accuracy: {accuracy:.4f}'})

        # validation
        model.eval()
        pbar = tqdm(val_loader)
        val_acc = 0
        val_loss_total = 0
        with torch.no_grad():
            for img, target in pbar:
                output = model(img)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item()
                val_acc += accuracy
                val_loss_total += loss.item()
                pbar.set_postfix({'Validation Loss': f'{loss.item():.4f} - Validation Accuracy: {accuracy:.4f}'})

        avg_train_acc = train_acc / len(train_loader.dataset)
        avg_val_acc = val_acc / len(val_loader.dataset)
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        print(f"Epoch {epoch + 1}/{args.ft_epochs}: Train Acc = {avg_train_acc:.4f}, Val Acc = {avg_val_acc:.4f}")

    return model, train_losses, val_losses, train_accs, val_accs


def create_results_directory(args):
    encoder = "cnn" if args.cnn else "mlp"
    mode = "quantum" if args.quantum else "classical"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(encoder, mode, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir




if __name__ == '__main__':
    args = parser.parse_args()
    
    # Create results directory
    results_dir = create_results_directory(args)
    print(f"Results will be saved to: {results_dir}")
    
    ### data ###
    train_dataset, val_dataset = load_training_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    ####################
    ### SSL training ###
    ####################

    ### model to train ###
    # backbone
    if args.cnn:
        backbone = CustomCNN(args)
    else:
        backbone = CustomMLP(args)
    print(f"\n -------------------- backbone used -------------------- \n {backbone} \n -------------------------------------------------")
    # end to end model
    model = SSL_q_loss(args, backbone)

    ### SSL training ###
    model, ssl_training_losses = train(model, train_loader, args)
    # plot SSL training loss
    plot_training_loss(ssl_training_losses, args, results_dir)

    ### linear-evaluation ###
    frozen_backbone = model.backbone
    feat_size = model.backbone_features
    frozen_model = nn.Sequential(frozen_backbone, nn.Linear(feat_size, args.classes))
    # freeze backbone and train FC layer
    frozen_model.requires_grad_(False)
    frozen_model[-1].requires_grad_(True)
    # evaluate the model
    train_dataset, eval_dataset = load_finetuning_data(args)
    print(f"\n Loaded a train dataset of shape {len(train_dataset)}")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    model, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs = linear_evaluation(frozen_model, train_loader,
                                                                                  val_loader, args)

    ### display ###
    # compute and save t-SNE and confusion matrix for fine-tuned model
    print("\n### Computing t-SNE and Confusion Matrix for Fine-tuned Model ###")
    tsne_features, tsne_labels = compute_tsne(model, val_loader, device='cpu')
    cm, accuracy = compute_confusion_matrix(model, val_loader, device='cpu')
    print(f"Fine-tuned model accuracy: {accuracy:.4f}")
    # plot evaluation metrics
    plot_evaluation_metrics(ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs, args, results_dir)
    # plot and save t-SNE and confusion matrix if plots are enabled
    plot_tsne(tsne_features, tsne_labels, 'fine_tuned', results_dir, args)
    plot_confusion_matrix(cm, accuracy, 'fine_tuned', results_dir, args)


    #########################################
    ### Comparison with a random backbone ###
    #########################################
    print("\n### Random Baseline Evaluation ###")
    if args.cnn:
        frozen_backbone = CustomCNN(args)
    else:
        frozen_backbone = CustomMLP(args)

    feat_size = args.backbone_dim
    frozen_random_model = nn.Sequential(frozen_backbone, nn.Linear(feat_size, args.classes))
    # freeze backbone and unfreeze last FC layer
    frozen_random_model.requires_grad_(False)
    frozen_random_model[-1].requires_grad_(True)
    
    ### data ###
    train_dataset, eval_dataset = load_finetuning_data(args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    ### linear-evaluation ###
    random_model, r_ft_train_losses, r_ft_val_losses, r_ft_train_accs, r_ft_val_accs = linear_evaluation(frozen_random_model, train_loader,
                                                                                                  val_loader, args)

    ### display ###
    # compute and save t-SNE and confusion matrix for random model
    print("\n### Computing t-SNE and Confusion Matrix for Random Model ###")
    random_tsne_features, random_tsne_labels = compute_tsne(random_model, val_loader, device='cpu')
    random_cm, random_accuracy = compute_confusion_matrix(random_model, val_loader, device='cpu')
    print(f"Random model accuracy: {random_accuracy:.4f}")

    # plot and save t-SNE and confusion matrix if plots are enabled
    plot_tsne(random_tsne_features, random_tsne_labels, 'random', results_dir, args)
    plot_confusion_matrix(random_cm, random_accuracy, 'random', results_dir, args)
    
    # prepare random results for saving
    random_results = {
        'train_losses': r_ft_train_losses,
        'val_losses': r_ft_val_losses,
        'train_accs': r_ft_train_accs,
        'val_accs': r_ft_val_accs
    }
    
    # save results to JSON with random baseline
    save_results_to_json(args, ssl_training_losses, ft_train_losses, ft_val_losses, ft_train_accs, ft_val_accs, results_dir, random_results)


