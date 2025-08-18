import torch
from tqdm import tqdm
import time
from utils import plot_training_metrics
from utils import compute_downsample_shape, pca_downsample
import torch.nn.functional as F
import os
import pickle
from pathlib import Path

# Get the directory of this file (training.py)
base_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level (assuming your project structure has 'src' and 'data' as siblings)
script_dir = Path(__file__).parent
PCA_PATH= (script_dir/ "data").resolve()
data_dir = os.path.join(base_dir, "..", "data")
pca_model_path = os.path.join(PCA_PATH, "pca_model.pkl")


def fit(epochs, lr, model, train_loader, val_loader, bs, optimizer, cfg):
    """
    Train the model for a specified number of epochs.
    If the model uses quantum embedding (embedding_size > 0), embed images using the BosonSampler.
    """
    history = []
    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    
    
    # Compute the downsample shape (e.g. (9, 14) if nb_parameters == 126)
    downsample_shape = compute_downsample_shape(bs.nb_parameters)
    print(f"Downsample shape for images set to: {downsample_shape[0]}x{downsample_shape[1]}")
    
    for epoch in range(epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            if model.embedding_size:
                # Assume batch contains a list of images if batch_size > 1.
                images, labs = batch  # images shape: [B, 1, 28, 28]
                # Process each image sequentially (QaaS does not handle batching well)
                emb_list = []
                for img in images:

                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        # Load your pre-trained PCA model (ensure you have it saved in pca_model.pkl)
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        # Use the PCA-based downsampling and assign to img_resized
                        img_resized  = pca_downsample(img, pca_model)
                        #print(f" \n ---------- \n Image resized with PCA to {img_resized.shape} \n ----------")
                    else:
                        # Use bilinear interpolation as fallback
                        # Using sequential embedding (remove parallel_embed call)
                        # Add batch and channel dimensions: [1, 1, 28, 28]
                        img_expanded = img.unsqueeze(0)  # now shape: [1, 1, 28, 28]
                        # Resize image to computed downsample shape.  (e.g. (9,14))
                        img_resized = F.interpolate(img_expanded, size=downsample_shape, mode='bilinear', align_corners=False)
                        # Remove batch and channel dims: shape becomes [height, width]  resulting shape becomes [1, 9, 14]
                        img_resized = img_resized.squeeze(0)
                        # Now the flattened tensor has exactly downsample_shape[0]*downsample_shape[1] pixels.
                        #print(f" \n ---------- \n Image resized with bilinear to {img_resized.shape} \n ----------")
                    #print(f"\n ---- \n Img resized of shape {img_resized.shape} \n ---- ")
                    emb = bs.embed(img_resized, bs.n_samples)
                    emb_list.append(emb)

                # Stack embeddings and add the batch dimension if needed.
                #embeddings = torch.stack(embeddings)  # shape: [B, embedding_size]
                emb = torch.stack(emb_list)

                # TODO concatenate Fourier features
                #  # Compute Fourier features for each image and stack them:
                # fourier_feats = [fourier_encode(img, n_features=cfg["model"].get("fourier_features", 50)) for img in image_list]
                # fourier_feats = torch.stack(fourier_feats)  # shape: [B, fourier_features]

                # # Concatenate along feature dimension:
                # emb = torch.cat((emb, fourier_feats), dim=1)  # new embedding size = bs.embedding_size + fourier_features


                # Now call the training step (ensure your model.training_step accepts batched embeddings)
                loss, acc = model.training_step((images, labs), emb=emb)
                #loss, acc = model.training_step(batch, emb=emb)
            else:
                loss, acc = model.training_step(batch)
            
            #if step % 50 == 0:
            tqdm.write(f"Step {step}: loss={loss.item():.4f}, acc={acc.item():.4f}")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach())
            running_acc += float(acc.detach())
            if model.embedding_size and step % 100 == 0 and step > 0:
                print(f"Step {step}, Training acc: {running_acc/(step+1):.4f}, Loss: {running_loss/(step+1):.4f}")
        result = evaluate(model, val_loader, bs, cfg)
        model.epoch_end(epoch, result)
        history.append(result)
        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(running_acc / len(train_loader))
        val_loss_list.append(result["val_loss"])
        val_acc_list.append(result["val_acc"])
        plot_training_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list)
    return history



#TODO Integrate Fourier Features into the Data Pipeline  **
#  Concatenate with Quantum Embedding. OR  Preprocess Images Before Feeding into the CNN.
#fourier_feats = fourier_encode(images.squeeze(0).squeeze(0), n_features=50)  # choose 50 features
## After obtaining the quantum embedding (emb), concatenate them:
#emb = torch.cat((emb, fourier_feats.unsqueeze(0)), dim=1)
#######
#fourier_size = cfg["model"].get("fourier_features", 0)
#total_embedding_size = bs.embedding_size + fourier_size if cfg["model"]["type"] == "hybrid" else 0
## When initializing HybridMnistModel:
#net = HybridMnistModel(device=device, embedding_size=total_embedding_size)




def evaluate(model, val_loader, bs, cfg):
    """
    Evaluate the model on the validation set.
    If quantum embedding is used, compute it for each sample.
    """
    model.eval()
    outputs = []

    # Compute the target downsample shape (e.g. 9x14 if bs.nb_parameters equals 126)
    downsample_shape = compute_downsample_shape(bs.nb_parameters)

    with torch.no_grad():
        for batch in val_loader:
            if model.embedding_size:
                images, labs = batch # images shape: [B, 1, 28, 28]
                batch_embeddings = []
                # Process each image in the batch individually:
                for i in range(images.size(0)):
                    img = images[i]  # shape: [1, 28, 28]
                    # Apply downsampling based on the chosen method.
                    if cfg["quantum"].get("downsample_method", "bilinear") == "pca":
                        # Load the PCA model (ensure the path is correct)
                        with open(pca_model_path, "rb") as f:
                            pca_model = pickle.load(f)
                        img_resized = pca_downsample(img, pca_model)
                    else:
                        # Use bilinear interpolation
                        # Expand dimensions to [1, 1, 28, 28] if needed.
                        img_expanded = img.unsqueeze(0)
                        img_resized = F.interpolate(img_expanded, size=downsample_shape, mode='bilinear', align_corners=False)
                        img_resized = img_resized.squeeze(0)


                    # Now img_resized should have exactly downsample_shape[0]*downsample_shape[1] pixels.
                    # Compute the quantum embedding.
                    # Depending on configuration, use adaptive or standard embedding
                    if cfg["quantum"].get("adaptive_sampling", False):
                        emb = bs.adaptive_embed(img_resized, min_samples=100, max_samples=bs.n_samples, tol=1e-3)
                    else:
                        emb = bs.embed(img_resized, bs.n_samples)
                    batch_embeddings.append(emb)
                # Stack embeddings to form a tensor of shape [B, embedding_size]
                emb_tensor = torch.stack(batch_embeddings)
                # Pass the batch and the corresponding embeddings to the model's validation step
                out = model.validation_step((images, labs), emb=emb_tensor)
            else:
                out = model.validation_step(batch)
            outputs.append(out)
    batch_losses = [x["val_loss"] for x in outputs]
    batch_accs = [x["val_acc"] for x in outputs]
    val_loss = torch.stack(batch_losses).mean().item()
    val_acc = torch.stack(batch_accs).mean().item()
    return {"val_loss": val_loss, "val_acc": val_acc}
