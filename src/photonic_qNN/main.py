import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from merlin import QuantumLayer, OutputMappingStrategy
import math
from sklearn import svm
import argparse
from sklearn.decomposition import PCA
from utils_photonic_qNN import (create_quantum_circuit, train_model, set_seed, count_parameters,
                                ScaleLayer, visualize_scale_parameters,
                                display_confusion_matrices, display_tsne, save_experiment_results,
                                load_dataset)


parser = argparse.ArgumentParser(description='Proposing a quantum kernel method to classify diverse datasets')
parser.add_argument('--bs', type=int, default = 64, help='Batch size')
parser.add_argument("--lr", type = float, default = 0.05, help = "Learning rate")
parser.add_argument("--modes", type = int, default = 10, help = "Number of modes in the interferometer")
parser.add_argument("--epochs", type = int, default = 10, help = "Number of epochs to train the model")
parser.add_argument("--size", type = int, default = 28, help = "Size of the images that is used")
parser.add_argument("--pca", type = bool, default = False, help = "Define if PCA is applied on the data")
parser.add_argument("--pca_comp", type = int, default = 8, help = "Number of PCA components used")
parser.add_argument("--display", type = bool, default = False, help = "Display layers, ConfMat and tSNE")

def main():
    #set_seed(42)
    args = parser.parse_args()
    # dataloader
    batch_size = args.bs
    # load data
    print("\n Loading dataset...")
    X_train,X_val,y_train,y_val, train_loader,val_loader, INPUT_SIZE, OUTPUT_FEATURES = load_dataset(args)
    print(f"... data loader with input size = {INPUT_SIZE}")
    print(f" - training statistics: \n - X_train: {X_train.shape} \n - X_val: {X_val.shape}")

    # Reduce to desired number of components (e.g., 40)
    if args.pca:
        print("\n - Extracting PCA components")
        n_components = args.pca_comp
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], args.size*args.size))
        X_val_pca = pca.transform(X_val.reshape(X_val.shape[0], args.size*args.size))

        X_pca = torch.sigmoid(torch.FloatTensor(X_pca))
        X_val_pca = torch.sigmoid(torch.FloatTensor(X_val_pca))
        print(X_pca.shape)
        print(X_pca[2])

        train_dataset = TensorDataset(X_pca, torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val_pca, torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        INPUT_SIZE = n_components

    ##################
    ## photonic qNN ##
    ##################
    print("\n - Building the photonic quantum neural network...")
    MODES = args.modes
    FREQUENCY = 1
    input_state = [(i + 1) % 2 for i in range(MODES)]
    photons_count = sum(input_state)
    print(f"input state: {input_state}")
    # build circuit with the trainable generic interferometers , FREQUENCY x {encoding layer, trainable generic interferometers}
    circuit = create_quantum_circuit(MODES, size = INPUT_SIZE, frequency = FREQUENCY)
    # build QuantumLayer (nn.Module) from circuit
    boson_layer = QuantumLayer(
        input_size=INPUT_SIZE,
        output_size=math.comb(MODES + photons_count - 1, photons_count),  # but we do not use it
        circuit=circuit,
        trainable_parameters=["phase", "bs"],
        input_parameters = ["px"],
        input_state=input_state,
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=False,
    )

    # learnable layer to map to the correct number of classes
    classification_layer = nn.Linear(in_features=math.comb(MODES + photons_count-1,photons_count), out_features=OUTPUT_FEATURES,
                                     bias=True)
    # input layer that multiplies the input by a learned encoding
    input_layer = ScaleLayer(INPUT_SIZE*FREQUENCY, scale_type="learned")
    if args.display:
        visualize_scale_parameters(input_layer)
    #nn.init.xavier_uniform_(classification_layer.weight)
    #nn.init.constant_(classification_layer.bias, 0.0)
    # create q_model as nn.Module
    q_model = nn.Sequential(input_layer, boson_layer, classification_layer)
    print("... Model built")

    ########################
    ## classical NN (MLP) ##
    ########################
    # 2 layers with ReLU activation
    layer_1 = nn.Linear(in_features=INPUT_SIZE, out_features=32, bias=True)
    layer_2 = nn.Linear(in_features=32, out_features=OUTPUT_FEATURES, bias=True)
    model = nn.Sequential(layer_1, nn.ReLU(), layer_2)

    #########################
    ## Training the models ##
    #########################

    EPOCHS = args.epochs
    LR = args.lr

    print(f" --- Training the quantum kernel")

    q_train_losses, q_val_losses, best_q_acc, q_train_accs, q_val_accs = train_model(q_model, train_loader, val_loader, num_epochs=EPOCHS, lr = LR, frequency = FREQUENCY, quantum = True)
    if args.display:
        visualize_scale_parameters(q_model[0])
        # save qLayer if needed
        #torch.save(q_model[0].state_dict(), 'scale_layer_trained_5.pt')

    # train classical baseline
    print(f" --- Training the classical kernel (linear)")
    cl_train_losses, cl_val_losses, best_cl_acc, cl_train_accs, cl_val_accs = train_model(model, train_loader, val_loader, num_epochs=EPOCHS, lr = 0.01)

    ### APPLY SVM ###
    print("--- APPLYING a SVM ---")
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    svm_acc = clf.score(X_val, y_val)
    print(f" -> Validation Accuracy after SVM: {svm_acc}")
    print("--- SVM applied ---")

    print(f" - TRAINING IS DONE - \n - Best validation accuracy for the quantum kernel = {best_q_acc:.4f} (for {count_parameters(q_model)} parameters),"
          f" \n - Best validation accuracy for the linear kernel = {best_cl_acc:.4f} (for {count_parameters(model)} parameters)")

    # display tSNE and confusion matrices if asked in arguments
    if args.display:
        print("\n - Computing the confusion matrices...")
        display_confusion_matrices(q_model, model, val_loader, device = 'cpu')
        print(" - Computing the tSNE plots")
        display_tsne(nn.Sequential(q_model[0], q_model[1]), model[0], val_loader, MODES, device="cpu")

    # save results
    print("\n - Saving results...")
    dict = {"dataset": "mnist",
            "learnable scale": True ,
            "best q ACC": best_q_acc, "best cl ACC": best_cl_acc,
            "q parameters":count_parameters(q_model),"cl parameters":count_parameters(model)}
    save_experiment_results(dict)
    print("Results saved !")
    print("\nEXPERIMENT COMPLETE !")

if __name__ == "__main__":
    main()