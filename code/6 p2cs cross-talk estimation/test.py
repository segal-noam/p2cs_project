import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import os
import wandb
import numpy as np
import random
from collections import Counter
from typing import Tuple, Optional, List, Dict, Any
from tqdm import tqdm

# Global constants
DATA_PATH = '/zdata/user-data/noam/data/p2cs'  # Update this path as needed

# -----------------------------------
# Model Classes


class DualHeadAdapter(nn.Module):
    def __init__(self, input_dim=2560, embed_dim=256, temperature=0.05, gamma=0.0, dropout=0.2):
        super().__init__()
        self.hk_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim)
        )
        self.rr_head = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim)
        )

        # Learnable logit scale (temperature)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / temperature))

        self.temperature = temperature
        self.gamma = gamma
        # Learnable gamma parameter
        # self.gamma = nn.Parameter(torch.ones([]) * gamma)

    def forward(self, hk_input, rr_input):
        hk_emb = F.normalize(self.hk_head(hk_input), dim=-1)
        rr_emb = F.normalize(self.rr_head(rr_input), dim=-1)
        return hk_emb, rr_emb, self.temperature, self.gamma


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer from "Unsupervised Domain Adaptation by Backpropagation"
    (Ganin & Lempitsky, 2015)

    This layer reverses the gradient during backpropagation by multiplying by -lambda.
    During forward pass, it acts as an identity function.
    """

    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_param
        return output, None


def gradient_reversal(x, lambda_param=1.0):
    """
    Convenience function to apply gradient reversal.

    Args:
        x: Input tensor
        lambda_param: Scaling factor for gradient reversal (default: 1.0)

    Returns:
        Tensor with same values as input but reversed gradients
    """
    return GradientReversalLayer.apply(x, lambda_param)


class PhylogeneticAdversary(nn.Module):
    def __init__(self, half_input_dim=256, dropout=0.2, lambda_param=1.0, reverse_gradient=True):
        super().__init__()

        self.lambda_param = lambda_param
        self.reverse_gradient = reverse_gradient

        self.ff = nn.Sequential(
            nn.Linear(half_input_dim * 2, half_input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(half_input_dim * 2, half_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half_input_dim, 1),
            nn.Sigmoid()  # output is between 0 and 1
        )

    def forward(self, hk_head_embedding, rr_head_embedding):
        # Apply gradient reversal to the concatenated embeddings
        concatenated_embeddings = torch.cat(
            [hk_head_embedding, rr_head_embedding], dim=-1)
        if self.reverse_gradient:
            reversed_gradient_embeddings = gradient_reversal(
                concatenated_embeddings, self.lambda_param)
        else:
            reversed_gradient_embeddings = concatenated_embeddings
        return self.ff(reversed_gradient_embeddings)

# -------------------------------------------------------------
# Loss and evaluation functions


def soft_clip_loss(similarity_matrix, target_matrix, temperature=0.07, eps=1e-8):
    """
    Computes symmetric CLIP-style contrastive loss with soft target similarity.

    Args:
        similarity_matrix (Tensor): [N, N] cosine similarities (unnormalized).
        target_matrix (Tensor): [N, N] soft similarity targets (e.g., 1, 0.5, 0).
        temperature (float): Scaling factor applied to similarities.
        eps (float): Small value to prevent division by zero or log(0).

    Returns:
        loss (Tensor): Scalar loss value.
    """
    # Apply temperature scaling
    logits = similarity_matrix / temperature
    tgt_logits = target_matrix / temperature

    # Log-probs along rows and columns (image-to-text and text-to-image)
    log_probs_i2t = F.log_softmax(logits, dim=1)  # row-wise
    log_probs_t2i = F.log_softmax(logits, dim=0)  # column-wise

    # Normalize target matrix row-wise and column-wise to form soft distributions
    log_tgt_probs_i2t = F.log_softmax(tgt_logits, dim=1)
    log_tgt_probs_t2i = F.log_softmax(tgt_logits, dim=0)

    # Compute KL divergence in both directions
    loss_i2t = F.kl_div(log_probs_i2t, log_tgt_probs_i2t,
                        reduction='batchmean', log_target=True)
    loss_t2i = F.kl_div(log_probs_t2i, log_tgt_probs_t2i,
                        reduction='batchmean', log_target=True)

    return 0.5 * (loss_i2t + loss_t2i)


def organism_aware_clip_loss(hk_emb, rr_emb, organism, hk_genes, rr_genes, temperature, gamma, mask_inter_organism_loss=False):
    """Compute organism-aware CLIP loss with optional inter-organism loss masking.

    Args:
        hk_emb: HK embeddings
        rr_emb: RR embeddings  
        organism: List of organism names
        hk_genes: List of HK gene names
        rr_genes: List of RR gene names
        temperature: Temperature scaling factor
        gamma: Gamma parameter for inter-organism similarity
        mask_inter_organism_loss: If True, only intra-organism pairs contribute to loss
    """
    # Normalize embeddings
    hk_emb = F.normalize(hk_emb, dim=-1)
    rr_emb = F.normalize(rr_emb, dim=-1)

    batch_size = hk_emb.shape[0]
    device = hk_emb.device

    # Calculate similarity matrix
    similarity_matrix = hk_emb @ rr_emb.T  # shape (B, B), unnormalized

    # Create organism mask - True where organisms match
    organism_tensor = torch.tensor([hash(org)
                                   for org in organism], device=device)
    same_organism_mask = (
        organism_tensor[:, None] == organism_tensor[None, :])  # (B, B)

    # Create cognate mask (diagonal)
    cognate_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

    # Identify duplicated HK genes and their cognates within the batch
    hk_gene_counts = Counter(hk_genes)
    rr_gene_counts = Counter(rr_genes)
    duplicated_hk_indices = [i for i, gene in enumerate(
        hk_genes) if hk_gene_counts[gene] > 1]
    duplicated_rr_indices = [i for i, gene in enumerate(
        rr_genes) if rr_gene_counts[gene] > 1]

    # Create a mask for duplicated HKs and their cognates
    duplicated_hk_mask = torch.zeros(
        (batch_size, batch_size), device=device, dtype=torch.bool)
    for i in duplicated_hk_indices:
        all_indices_of_this_hk = [j for j, gene in enumerate(
            hk_genes) if gene == hk_genes[i]]
        duplicated_hk_mask[i, all_indices_of_this_hk] = True

    # Create a mask for duplicated RR genes and their cognates
    duplicated_rr_mask = torch.zeros(
        (batch_size, batch_size), device=device, dtype=torch.bool)
    for i in duplicated_rr_indices:
        all_indices_of_this_rr = [j for j, gene in enumerate(
            rr_genes) if gene == rr_genes[i]]
        duplicated_rr_mask[i, all_indices_of_this_rr] = True

    # Construct the target similarity matrix ensuring gamma remains a tensor for gradient flow
    # Create a matrix of ones with the same shape as similarity_matrix
    target_matrix = torch.ones_like(similarity_matrix) * gamma

    # For same organism pairs, set target to 0 (unless they are cognates)
    target_matrix[same_organism_mask] = 0.0

    # For cognate pairs (including duplicated HK cognates), set target to 1
    # | duplicated_hk_mask | duplicated_rr_mask  # TODO: reinstate
    cognate_pairs = cognate_mask
    target_matrix[cognate_pairs] = 1.0

    # Ensure targets are non-negative (should be handled by the logic above, but for safety)
    target_matrix = torch.relu(target_matrix)

    # Apply inter-organism loss masking if requested
    if mask_inter_organism_loss:
        # Create mask for intra-organism pairs only (same organism or cognates)
        intra_organism_mask = same_organism_mask | cognate_mask
        # Set inter-organism similarities to a neutral value to mask their contribution
        similarity_matrix = torch.where(
            intra_organism_mask, similarity_matrix, torch.zeros_like(similarity_matrix))
        target_matrix = torch.where(
            intra_organism_mask, target_matrix, torch.zeros_like(target_matrix))

    # Compute the soft contrastive loss using the helper function
    total_loss = soft_clip_loss(
        similarity_matrix, target_matrix, temperature=temperature)

    return total_loss


def evaluate_model(model, test_dataloader, device='cuda', gamma=0.5, hk_vae=None):
    model.eval()
    if hk_vae is not None:
        hk_vae.eval()
    total_test_loss = 0
    with torch.no_grad():
        for (hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes), indices in tqdm(test_dataloader, desc="Evaluating on Test Set"):  # Unpack the tuple
            hk_batch_data, rr_batch_data = hk_batch_data.to(
                device), rr_batch_data.to(device)  # Move tensors to device
            # organism, hk_genes, and rr_genes are lists of strings/objects, don't move them to device like tensors

            # Pass HK data through VAE encoder if VAE is provided
            if hk_vae is not None:
                mu, logvar = hk_vae.encode(hk_batch_data)
                hk_input_for_adapter = mu  # Use the latent mean as input to the adapter
            else:
                hk_input_for_adapter = hk_batch_data  # Use original HK data

            hk_emb, rr_emb, temperature, gamma = model(
                hk_input_for_adapter, rr_batch_data)
            loss = organism_aware_clip_loss(
                hk_emb, rr_emb, organism, hk_genes, rr_genes, temperature, gamma)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    return avg_test_loss


def evaluate_cognate_prediction(model, dataloader, device='cuda', restrict_organism=True, hk_vae=None):
    model.eval()
    if hk_vae is not None:
        hk_vae.eval()
    hk_embeddings = []
    rr_embeddings = []
    organism_list = []

    with torch.no_grad():
        for (hk_batch, rr_batch, organism, hk_genes, rr_genes), _ in tqdm(dataloader, desc="Generating Embeddings"):
            hk_batch, rr_batch = hk_batch.to(device), rr_batch.to(device)
            # organism, hk_genes, and rr_genes are lists of strings/objects, don't move them to device like tensors

            # Pass HK data through VAE encoder if VAE is provided
            if hk_vae is not None:
                mu, logvar = hk_vae.encode(hk_batch)
                hk_input_for_adapter = mu
            else:
                hk_input_for_adapter = hk_batch

            hk_emb, rr_emb, _, _ = model(hk_input_for_adapter, rr_batch)
            hk_embeddings.append(hk_emb)
            rr_embeddings.append(rr_emb)
            organism_list.extend(organism)

    # Stack embeddings on GPU
    hk_embeddings = torch.cat(hk_embeddings, dim=0)  # [N, D]
    rr_embeddings = torch.cat(rr_embeddings, dim=0)  # [N, D]

    # Normalize for cosine similarity
    hk_norm = F.normalize(hk_embeddings, p=2, dim=1)
    rr_norm = F.normalize(rr_embeddings, p=2, dim=1)

    # Compute cosine similarity matrix [N, N]
    similarity_matrix = torch.matmul(hk_norm, rr_norm.T)

    if restrict_organism:
        # Create a mask to restrict similarity comparison to same-organism RR embeddings
        organism_tensor = torch.tensor(
            [hash(o) for o in organism_list], device=device
        )  # hash to numeric for comparison

        # Compare outer product to get a mask [N, N] where True = same organism
        org_equal_mask = (organism_tensor[:, None] == organism_tensor[None, :])

        # Set similarity to -inf where organism does not match
        similarity_matrix[~org_equal_mask] = float('-inf')

    # Find nearest RR for each HK
    nearest_indices = torch.argmax(similarity_matrix, dim=1)

    # Ground truth: cognate should be at the same index
    ground_truth = torch.arange(len(hk_embeddings), device=device)

    accuracy = (nearest_indices == ground_truth).float().mean().item()
    return accuracy


def evaluate_cognate_prediction_topk(model=None, dataloader=None, device='cuda', k=None, threshold=None, restrict_organism=True, hk_embeddings=None, rr_embeddings=None, organism_list=None, hk_vae=None):
    """
    Fast TopK evaluation for the case where all HK-RR pairs are unique.
    Uses fully vectorized operations with no Python loops for maximum efficiency.

    This function assumes each HK protein appears exactly once in the dataset.

    Args:
        model: Trained model. Required if hk_embeddings and rr_embeddings are None.
        dataloader: DataLoader for evaluation. Used to get organism info if embeddings are not provided.
        device: Device to run inference on.
        k: Number of nearest neighbors to consider as cognates.
           If None, only thresholding is used.
        threshold: Similarity threshold. If None, only k-NN is used.
        restrict_organism: Whether to restrict comparisons to same organism.
        hk_embeddings (Tensor, optional): Pre-calculated HK embeddings.
        rr_embeddings (Tensor, optional): Pre-calculated RR embeddings.
        organism_list (list, optional): List of organisms corresponding to the embeddings.
                                       Required if embeddings are provided and dataloader is None.
        hk_vae (nn.Module, optional): The VAE model to encode HK embeddings before passing to the adapter.


    Returns:
        tuple: (TP, TN, FP, FN) confusion matrix values
    """
    if hk_vae is not None:
        hk_vae.eval()

    if hk_embeddings is None or rr_embeddings is None:
        if model is None or dataloader is None:
            raise ValueError(
                "Either provide pre-calculated embeddings and organism_list, or a model and dataloader.")

        model.eval()
        hk_embeddings = []
        rr_embeddings = []
        organism_list = []

        with torch.no_grad():
            for (hk_batch, rr_batch, organism, hk_gene, rr_gene), _ in tqdm(dataloader, desc="Generating Embeddings for Evaluation"):
                hk_batch, rr_batch = hk_batch.to(device), rr_batch.to(device)

                # Pass HK data through VAE encoder if VAE is provided
                if hk_vae is not None:
                    mu, logvar = hk_vae.encode(hk_batch)
                    hk_input_for_adapter = mu
                else:
                    hk_input_for_adapter = hk_batch

                hk_emb, rr_emb, _, _ = model(hk_input_for_adapter, rr_batch)
                hk_embeddings.append(hk_emb)
                rr_embeddings.append(rr_emb)
                organism_list.extend(organism)

        # Stack embeddings on GPU
        hk_embeddings = torch.cat(hk_embeddings, dim=0)  # [N, D]
        rr_embeddings = torch.cat(rr_embeddings, dim=0)  # [N, D]
    else:
        # Ensure provided embeddings are on the correct device
        hk_embeddings = hk_embeddings.to(device)
        rr_embeddings = rr_embeddings.to(device)
        # If embeddings are provided, we need the organism list.
        # If dataloader is provided, we get it from there.
        # If dataloader is NOT provided, organism_list must be provided explicitly.
        if dataloader is not None:
            organism_list = []
            for (hk_batch, rr_batch, organism, hk_gene, rr_gene), _ in dataloader:
                organism_list.extend(organism)
        elif organism_list is None:
            raise ValueError(
                "organism_list must be provided when using pre-calculated embeddings and dataloader is None.")

    N = len(hk_embeddings)

    # Normalize for cosine similarity
    hk_norm = F.normalize(hk_embeddings, p=2, dim=1)
    rr_norm = F.normalize(rr_embeddings, p=2, dim=1)

    # Compute similarity matrix [N, N]
    similarity_matrix = torch.matmul(hk_norm, rr_norm.T)

    if restrict_organism:
        # Create organism mask - True where organisms match
        organism_tensor = torch.tensor(
            [hash(o) for o in organism_list], device=device, dtype=torch.long)
        same_organism_mask = (
            organism_tensor[:, None] == organism_tensor[None, :])  # [N, N]

        # Set similarity to -inf where organism does not match
        similarity_matrix[~same_organism_mask] = float('-inf')
    else:
        # If not restricting by organism, all pairs are considered
        same_organism_mask = torch.ones(
            N, N, dtype=torch.bool, device=device)  # Used for evaluation mask

    # Create prediction matrix: [N, N] where entry [i,j] is True if HK i predicts RR j as cognate
    prediction_matrix = torch.zeros(N, N, dtype=torch.bool, device=device)

    # Determine prediction indices based on threshold and k
    if threshold is not None:
        # Apply threshold first
        threshold_mask = (similarity_matrix >= threshold)

        if k is not None:
            # Combine thresholding with k-NN
            # Get top k indices for each HK
            _, top_k_indices = torch.topk(
                similarity_matrix, k=min(k, N), dim=1)  # [N, k]

            # Create a mask for the top k indices
            top_k_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
            hk_indices_expand = torch.arange(N, device=device)[
                :, None].expand(-1, k)
            top_k_mask[hk_indices_expand.flatten(
            ), top_k_indices.flatten()] = True

            # Final prediction mask: all above threshold OR among top k
            prediction_matrix = threshold_mask | top_k_mask
        else:
            # Only thresholding is used
            prediction_matrix = threshold_mask
    elif k is not None:
        # Only k-NN is used
        _, top_k_indices = torch.topk(
            similarity_matrix, k=min(k, N), dim=1)  # [N, k]
        hk_indices_expand = torch.arange(N, device=device)[
            :, None].expand(-1, k)
        prediction_matrix[hk_indices_expand.flatten(),
                          top_k_indices.flatten()] = True
    else:
        # No k or threshold specified, default to k=1 (nearest neighbor)
        print("Warning: Neither k nor threshold specified. Defaulting to k=1.")
        _, top_k_indices = torch.topk(similarity_matrix, k=1, dim=1)  # [N, 1]
        hk_indices_expand = torch.arange(N, device=device)[
            :, None].expand(-1, 1)
        prediction_matrix[hk_indices_expand.flatten(),
                          top_k_indices.flatten()] = True

    # Ground truth matrix: [N, N] where entry [i,j] is True if HK i and RR j are true cognates
    # In this case, true cognates are on the diagonal (same index)
    ground_truth_matrix = torch.eye(N, dtype=torch.bool, device=device)

    if restrict_organism:
        # Only consider same-organism pairs for evaluation
        evaluation_mask = same_organism_mask
    else:
        # Consider all pairs
        evaluation_mask = torch.ones(N, N, dtype=torch.bool, device=device)

    # Apply evaluation mask
    pred_masked = prediction_matrix[evaluation_mask]
    truth_masked = ground_truth_matrix[evaluation_mask]

    # Calculate confusion matrix - FULLY VECTORIZED
    TP = torch.sum(truth_masked & pred_masked).item()
    FN = torch.sum(truth_masked & ~pred_masked).item()
    FP = torch.sum(~truth_masked & pred_masked).item()
    TN = torch.sum(~truth_masked & ~pred_masked).item()

    return TP, TN, FP, FN


def adversary_loss_excluding_diagonal(predicted_distances, ground_truth_distances):
    """Calculate MSE loss excluding diagonal elements."""
    # Create mask to exclude diagonal elements
    n = predicted_distances.shape[0]
    mask = torch.ones_like(predicted_distances, dtype=torch.bool)
    mask.fill_diagonal_(False)  # Set diagonal to False

    # Flatten and apply mask
    pred_flat = predicted_distances[mask]
    gt_flat = ground_truth_distances[mask]

    # Calculate MSE loss on non-diagonal elements
    return F.mse_loss(pred_flat, gt_flat)


def evaluate_adversary_on_same_organism_prediction(predicted_organism_distances, ground_truth_organism_distances):
    """
    Evaluates the accuracy of predicting 'same organism' vs 'different organism'
    by thresholding a provided organism distance matrix.

    Args:
        predicted_organism_distances (torch.Tensor): [N, N] matrix of predicted distances between organisms.
        ground_truth_organism_distances (torch.Tensor): [N, N] binary matrix where True (or 1) indicates same organism, False (or 0) different organism.

    Returns:
        tuple: (accuracy (float), negative_pair_ratio (float)) where:
            - accuracy: Classification accuracy calculated only on positive pairs (same organism pairs).
            - negative_pair_ratio: Ratio of negative example pairs (different organisms) to total pairs in the sub-batch.
    """
    # Use classification threshold of 0.5: predict 'same organism' if distance <= 0.5
    preds = predicted_organism_distances <= 0.5

    # If ground_truth_organism_distances is float, treat >0.5 as different, <=0.5 as same
    if ground_truth_organism_distances.dtype != torch.bool:
        y_true = ground_truth_organism_distances <= 0.5
    else:
        y_true = ground_truth_organism_distances

    y_pred = preds

    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Calculate accuracy only on positive pairs (same organism pairs)
    positive_mask = y_true_flat == True
    if positive_mask.sum() > 0:
        accuracy = (y_true_flat[positive_mask] ==
                    y_pred_flat[positive_mask]).float().mean().item()
    else:
        accuracy = 0.0  # No positive pairs available

    # Calculate negative pair ratio (different organisms / total pairs)
    # Different organisms are where y_true is False (or 0)
    total_pairs = y_true.numel()
    negative_pairs = (y_true == False).sum().item()
    negative_pair_ratio = negative_pairs / total_pairs if total_pairs > 0 else 0.0

    return accuracy, negative_pair_ratio


def build_distance_matrix(organisms, distance_mat_df):
    """
    Construct a distance matrix for a batch of organisms.

    Args:
        organisms (list of str): List of organism names (batch).
        distance_mat_df (pd.DataFrame): Square DataFrame representing pairwise distances
            between organism names (must have organism names as both index and columns).
            If None, builds a binary same/different-organism matrix.

    Returns:
        torch.Tensor: 2D tensor of shape [batch_size, batch_size] where
            entry (i, j) is the phylogenetic distance between organisms[i] and organisms[j],
            or 0.0 if the same organism, or 1.0 for missing/different values.
    """
    if distance_mat_df is not None:
        # Subselect rows and columns corresponding to the input organisms with underscores
        sub_df = distance_mat_df.reindex(index=organisms, columns=organisms)

        # Set distance for same organism pairs to 0.0
        np.fill_diagonal(sub_df.values, 0.0)

        # Replace missing or NaN values with 1.0 (default max distance)
        distance_np = sub_df.fillna(1.0).to_numpy(dtype=np.float32)

        return torch.from_numpy(distance_np)
    else:
        # Create binary same-organism mask: 0 if same organism, 1 if different
        return build_same_organism_mask(organisms)


def build_same_organism_mask(organism_list):
    """Build binary mask where 0 indicates same organism, 1 indicates different organism."""
    n = len(organism_list)

    # Create mask using numpy for string comparison, then convert to tensor
    import numpy as np

    # Convert to numpy array for vectorized string comparison
    organisms = np.array(organism_list)

    # Create broadcasting comparison: organisms[i] != organisms[j] for all i,j
    # Shape: (n, 1) != (1, n) -> (n, n)
    mask_np = (organisms[:, np.newaxis] !=
               organisms[np.newaxis, :]).astype(np.float32)

    # Convert back to torch tensor with correct dtype
    return torch.from_numpy(mask_np).float()


def evaluate_adversary_accuracy(adversary, main_model, dataloader, device='cuda', hk_vae=None, distance_mat_df=None):
    """
    Evaluate adversary accuracy on same organism prediction across the entire dataset.

    Args:
        adversary: The phylogenetic adversary model
        main_model: The main DualHeadAdapter model to generate embeddings
        dataloader: DataLoader containing batches of (hk_batch, rr_batch, organism, hk_genes, rr_genes)
        device: Device to run evaluation on
        hk_vae: Optional VAE model for HK preprocessing
        distance_mat_df: Optional distance matrix DataFrame for ground truth distances

    Returns:
        tuple: (accuracy (float), negative_pair_ratio (float)) where:
            - accuracy: Overall accuracy of same organism prediction
            - negative_pair_ratio: Average ratio of negative example pairs across all batches
    """
    adversary.eval()
    main_model.eval()
    if hk_vae is not None:
        hk_vae.eval()

    total_accuracy = 0.0
    total_negative_ratio = 0.0
    batch_count = 0

    with torch.no_grad():
        for (hk_batch, rr_batch, organism, hk_genes, rr_genes), _ in tqdm(dataloader, desc="Evaluating Adversary Accuracy"):
            hk_batch, rr_batch = hk_batch.to(device), rr_batch.to(device)

            # Pass HK data through VAE encoder if VAE is provided
            if hk_vae is not None:
                mu, logvar = hk_vae.encode(hk_batch)
                hk_input_for_adapter = mu
            else:
                hk_input_for_adapter = hk_batch

            # Get embeddings from the main model
            hk_emb, rr_emb, _, _ = main_model(hk_input_for_adapter, rr_batch)

            batch_size = hk_batch.shape[0]

            # Get ground truth organism distances
            ground_truth_distances = build_distance_matrix(
                organism, distance_mat_df).to(device)

            # Predict pairwise distances using adversary
            pred_hk = hk_emb.unsqueeze(1).expand(-1, batch_size, -1)
            pred_rr = rr_emb.unsqueeze(0).expand(batch_size, -1, -1)
            predicted_distances = adversary(pred_hk, pred_rr).squeeze(-1)

            if predicted_distances.dim() == 3:
                predicted_distances = predicted_distances.squeeze(-1)

            # Evaluate accuracy for this batch
            batch_accuracy, batch_negative_ratio = evaluate_adversary_on_same_organism_prediction(
                predicted_distances, ground_truth_distances
            )

            total_accuracy += batch_accuracy
            total_negative_ratio += batch_negative_ratio
            batch_count += 1

    # Return average accuracy and negative pair ratio across all batches
    avg_accuracy = total_accuracy / batch_count if batch_count > 0 else 0.0
    avg_negative_ratio = total_negative_ratio / \
        batch_count if batch_count > 0 else 0.0
    return avg_accuracy, avg_negative_ratio


# -----------------------------------
# Utility Functions
def sample_sub_batch(hk_batch_data: torch.Tensor, rr_batch_data: torch.Tensor,
                     organism: List[str], hk_genes: List[str], rr_genes: List[str],
                     adversary_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[str]]:
    """Sample a sub-batch for adversary training by selecting organisms.

    Samples 2 organisms (or as many as available if less than 2) and includes
    all HK/RR pairs from those organisms in the sub-batch.

    Args:
        hk_batch_data: HK batch tensor
        rr_batch_data: RR batch tensor
        organism: List of organism names for each sample
        hk_genes: List of HK gene names
        rr_genes: List of RR gene names
        adversary_batch_size: Ignored (kept for backward compatibility). 
                              Currently samples 2 organisms.

    Returns:
        Tuple of sampled sub-batch data (hk_sub_batch, rr_sub_batch, organism_sub_batch, 
        hk_genes_sub_batch, rr_genes_sub_batch)
    """
    # Get unique organisms in the batch
    unique_organisms = list(set(organism))

    # Sample 2 organisms (or as many as available if less than 2)
    num_organisms_to_sample = min(2, len(unique_organisms))
    sampled_organisms = random.sample(
        unique_organisms, num_organisms_to_sample)

    # Find all indices where the organism matches one of the sampled organisms
    sub_batch_indices = [
        i for i, org in enumerate(organism) if org in sampled_organisms
    ]

    if len(sub_batch_indices) == 0:
        # Fallback: return all if somehow no matches (shouldn't happen)
        return hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes

    # Convert to tensor for indexing
    sub_batch_indices_tensor = torch.tensor(
        sub_batch_indices, device=hk_batch_data.device)

    return (hk_batch_data[sub_batch_indices_tensor],
            rr_batch_data[sub_batch_indices_tensor],
            [organism[i] for i in sub_batch_indices],
            [hk_genes[i] for i in sub_batch_indices],
            [rr_genes[i] for i in sub_batch_indices])


def get_model_embeddings(model: torch.nn.Module, hk_batch_data: torch.Tensor,
                         rr_batch_data: torch.Tensor, hk_vae: Optional[torch.nn.Module] = None,
                         freeze_model: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get embeddings from the model with optional VAE preprocessing."""
    def _forward():
        if hk_vae is not None:
            mu, logvar = hk_vae.encode(hk_batch_data)
            hk_input_for_adapter = mu
        else:
            hk_input_for_adapter = hk_batch_data
        return model(hk_input_for_adapter, rr_batch_data)

    if freeze_model:
        with torch.no_grad():
            return _forward()
    else:
        return _forward()


def train_adversary_step(adversary: torch.nn.Module, adversary_optimizer: torch.optim.Optimizer,
                         hk_emb: torch.Tensor, rr_emb: torch.Tensor,
                         ground_truth_distances: torch.Tensor, sub_batch_size: int,
                         freeze_model: bool = False) -> float:
    """Perform one step of adversary training with NORMAL gradients (not reversed)."""
    adversary_optimizer.zero_grad()

    if freeze_model:
        hk_emb.requires_grad_(True)
        rr_emb.requires_grad_(True)

    # Predict pairwise distances
    pred_hk = hk_emb.unsqueeze(1).expand(-1, sub_batch_size, -1)
    pred_rr = rr_emb.unsqueeze(0).expand(sub_batch_size, -1, -1)
    predicted_distances = adversary(pred_hk, pred_rr).squeeze(-1)

    if predicted_distances.dim() == 3:
        predicted_distances = predicted_distances.squeeze(-1)

    # Adversary wants to minimize this loss (normal gradient direction)
    adversary_loss = F.mse_loss(predicted_distances, ground_truth_distances)
    adversary_loss.backward()
    adversary_optimizer.step()

    return adversary_loss.item()


def run_validation(model: torch.nn.Module, val_dataloader: torch.utils.data.DataLoader,
                   device: torch.device, hk_vae: Optional[torch.nn.Module] = None,
                   adversary: Optional[torch.nn.Module] = None, adversary_batch_size: int = 32,
                   distance_mat_df=None, mask_inter_organism_loss=False) -> Tuple[float, float, float, float]:
    """Run validation loop and return average validation loss, adversary validation loss, adversary accuracy, and negative pair ratio."""
    model.eval()
    if hk_vae is not None:
        hk_vae.eval()
    if adversary is not None:
        adversary.eval()

    total_val_loss = 0
    total_adversary_val_loss = 0
    total_adversary_accuracy = 0
    total_negative_ratio = 0
    val_batch_count = 0
    adversary_val_batch_count = 0
    adversary_accuracy_batch_count = 0

    with torch.no_grad():
        for (hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes), indices in tqdm(val_dataloader, desc="Validation"):
            hk_batch_data, rr_batch_data = hk_batch_data.to(
                device), rr_batch_data.to(device)

            hk_emb, rr_emb, temperature, current_gamma = get_model_embeddings(
                model, hk_batch_data, rr_batch_data, hk_vae, freeze_model=True
            )
            val_loss = organism_aware_clip_loss(
                hk_emb, rr_emb, organism, hk_genes, rr_genes, temperature, current_gamma, mask_inter_organism_loss)
            total_val_loss += val_loss.item()
            val_batch_count += 1

            # Adversary validation
            if adversary is not None:
                hk_sub_batch, rr_sub_batch, organism_sub_batch, hk_genes_sub_batch, rr_genes_sub_batch = sample_sub_batch(
                    hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes, adversary_batch_size
                )
                hk_emb_sub, rr_emb_sub, _, _ = get_model_embeddings(
                    model, hk_sub_batch, rr_sub_batch, hk_vae, freeze_model=True
                )
                ground_truth_distances = build_distance_matrix(
                    organism_sub_batch, distance_mat_df).to(device)

                sub_batch_size = hk_sub_batch.shape[0]
                pred_hk = hk_emb_sub.unsqueeze(
                    1).expand(-1, sub_batch_size, -1)
                pred_rr = rr_emb_sub.unsqueeze(
                    0).expand(sub_batch_size, -1, -1)
                predicted_distances = adversary(pred_hk, pred_rr).squeeze(-1)

                if predicted_distances.dim() == 3:
                    predicted_distances = predicted_distances.squeeze(-1)

                adversary_val_loss = adversary_loss_excluding_diagonal(
                    predicted_distances, ground_truth_distances)
                total_adversary_val_loss += adversary_val_loss.item()
                adversary_val_batch_count += 1

                # Calculate accuracy
                adversary_accuracy, adversary_negative_ratio = evaluate_adversary_on_same_organism_prediction(
                    predicted_distances, ground_truth_distances)
                total_adversary_accuracy += adversary_accuracy
                total_negative_ratio += adversary_negative_ratio
                adversary_accuracy_batch_count += 1

    avg_val_loss = total_val_loss / val_batch_count if val_batch_count > 0 else 0
    avg_adversary_val_loss = total_adversary_val_loss / \
        adversary_val_batch_count if adversary_val_batch_count > 0 else 0
    avg_adversary_accuracy = total_adversary_accuracy / \
        adversary_accuracy_batch_count if adversary_accuracy_batch_count > 0 else 0
    avg_negative_ratio = total_negative_ratio / \
        adversary_accuracy_batch_count if adversary_accuracy_batch_count > 0 else 0

    return avg_val_loss, avg_adversary_val_loss, avg_adversary_accuracy, avg_negative_ratio


def save_model_checkpoint(model: torch.nn.Module, path: str, model_name: str) -> bool:
    """Save model checkpoint with error handling."""
    try:
        torch.save(model.state_dict(), path)
        print(f"Saved {model_name} to {path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to save {model_name}: {e}")
        return False


def update_top3_models(top3_heap: List[Tuple[float, int, Dict[str, torch.Tensor]]],
                       loss: float, epoch: int, state_dict: Dict[str, torch.Tensor]) -> None:
    """Update the top-3 models heap."""
    if loss is None or loss < 0:
        return

    neg_loss = -loss
    if len(top3_heap) < 3:
        heapq.heappush(top3_heap, (neg_loss, epoch, state_dict))
    elif neg_loss > top3_heap[0][0]:
        heapq.heapreplace(top3_heap, (neg_loss, epoch, state_dict))


def save_top3_models(top3_heap: List[Tuple[float, int, Dict[str, torch.Tensor]]],
                     save_dir: str, model_type: str = "model") -> None:
    """Save the top-3 models from the heap."""
    if not top3_heap:
        return

    for i, (neg_loss, epoch_save, state_dict) in enumerate(sorted(top3_heap, reverse=True)):
        if model_type == "adversary":
            topk_path = os.path.join(
                save_dir, f"top_{i+1}_adversary_epoch_{epoch_save}.pth")
            loss_name = "Adversary Loss"
        else:
            topk_path = os.path.join(
                save_dir, f"top_{i+1}_epoch_{epoch_save}.pth")
            loss_name = "Val Loss"

        save_model_checkpoint(type('obj', (object,), {'state_dict': lambda: state_dict})(
        ), topk_path, f"top {i+1} {model_type}")


def log_training_metrics(epoch: int, epochs: int, avg_train_loss: float, avg_val_loss: float,
                         avg_gamma: float, avg_adversary_loss: float, avg_combined_loss: float,
                         val_accuracy: Optional[float], optimizer: torch.optim.Optimizer,
                         freeze_dualheadadapter: bool, adversary: Optional[torch.nn.Module] = None,
                         evaluate_accuracy: bool = True, avg_adversary_val_loss: float = 0.0,
                         avg_adversary_accuracy: float = 0.0, avg_negative_ratio: float = 0.0, is_in_warmup: bool = False) -> None:
    """Log training metrics to wandb and print progress."""
    log_dict = {
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "gamma": avg_gamma,
        "learning_rate": optimizer.param_groups[0]['lr'] if not freeze_dualheadadapter else 0,
        "is_in_warmup": 1.0 if is_in_warmup else 0.0
    }

    if adversary is not None:
        log_dict["adversary_loss"] = avg_adversary_loss
        log_dict["adversary_val_loss"] = avg_adversary_val_loss
        log_dict["adversary_val_accuracy"] = avg_adversary_accuracy
        log_dict["adversary_negative_ratio"] = avg_negative_ratio
        log_dict["combined_loss"] = avg_combined_loss  # train + adversary loss

    if evaluate_accuracy and val_accuracy is not None:
        log_dict["val_accuracy"] = val_accuracy

    wandb.log(log_dict)

    # Print progress
    print(f"Epoch {epoch+1}/{epochs}")
    if not freeze_dualheadadapter:
        print(f"  Train Loss: {avg_train_loss:.6f}", end='; ')
        print(f"  Val Loss: {avg_val_loss:.6f}", end='; ')
        print(f"  Gamma: {avg_gamma:.6f}")
    if adversary is not None:
        print(f"  Adversary Loss: {avg_adversary_loss:.6f}", end='; ')
        print(f"  Adversary Val Loss: {avg_adversary_val_loss:.6f}", end='; ')
        print(
            f"  Adversary Val Accuracy: {avg_adversary_accuracy:.4f}", end='; ')
        print(
            f"  Adversary Negative Ratio: {avg_negative_ratio:.4f}", end='; ')
        print(f"  Combined Loss: {avg_combined_loss:.6f}")
    if evaluate_accuracy and val_accuracy is not None:
        print(f"  Val Accuracy: {val_accuracy:.4f}")


def load_best_models(model: torch.nn.Module, adversary: Optional[torch.nn.Module],
                     best_model_path: str, best_adversary_path: Optional[str],
                     device: torch.device, freeze_dualheadadapter: bool) -> None:
    """Load the best models from saved checkpoints."""
    # Load the best main model
    if not freeze_dualheadadapter and os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(
                best_model_path, map_location=device))
            print(f"Loaded best model from {best_model_path}")
        except Exception as e:
            print(f"Warning: Failed to load best model: {e}")

    # Load the best adversary
    if adversary is not None and best_adversary_path and os.path.exists(best_adversary_path):
        try:
            adversary.load_state_dict(torch.load(
                best_adversary_path, map_location=device))
            print(f"Loaded best adversary from {best_adversary_path}")
        except Exception as e:
            print(f"Warning: Failed to load best adversary: {e}")


def check_early_stopping(epochs_without_improvement: int, early_stopping_patience: int,
                         best_epoch: int, current_epoch: int) -> Tuple[bool, str]:
    """Check if early stopping should be triggered."""
    if epochs_without_improvement >= early_stopping_patience:
        message = f"Early stopping triggered after {current_epoch+1} epochs (no improvement for {early_stopping_patience} epochs). Best epoch was {best_epoch+1}"
        return True, message
    return False, ""

# -----------------------------------
# Main train loop


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=10, device='cuda',
          save_dir="model_weights", gamma=0.5, hk_vae=None, adversary=None, adversary_optimizer=None,
          adversary_batch_size=32, adversary_lambda=1.0, distance_mat_df=None, evaluate_accuracy=True,
          freeze_dualheadadapter=False, early_stopping_patience=5, min_delta=1e-6,
          separate_adversary_in_warmup=False, warmup_epochs=0, mask_inter_organism_loss=False):
    """Training function with optional adversarial training and cognate prediction accuracy reporting.

    Args:
        distance_mat_df: Optional distance matrix DataFrame. If None, creates binary same-organism mask
                        (0 for same organism, 1 for different organisms) for adversary training.
        separate_adversary_in_warmup: If True, during warmup epochs, adversary and main model
                                     learn separately by detaching gradients.
        warmup_epochs: Number of epochs to use for warmup phase (default: 0).
        mask_inter_organism_loss: If True, only intra-organism pairs contribute to the loss,
                                 effectively masking inter-organism loss contributions.
    """
    # Validate inputs
    if freeze_dualheadadapter and adversary is None:
        raise ValueError(
            "adversary must be provided when freeze_dualheadadapter=True")

    if adversary is not None and adversary_optimizer is None:
        raise ValueError(
            "adversary_optimizer must be provided when adversary is provided")

    wandb.init(project="clip-hk-rr", group='all_runs',
               config={
                   "epochs": epochs,
                   "gamma_initial": gamma.item() if isinstance(gamma, torch.Tensor) else gamma,
                   "organism_complete_batching": True,
                   "use_vae_mu": hk_vae is not None,
                   "adversarial_training": adversary is not None,
                   "adversary_batch_size": adversary_batch_size if adversary is not None else None,
                   "adversary_lambda": adversary_lambda if adversary is not None else None,
                   "use_distance_matrix": distance_mat_df is not None,
                   "separate_adversary_in_warmup": separate_adversary_in_warmup,
                   "warmup_epochs": warmup_epochs,
                   "evaluate_accuracy": evaluate_accuracy,
                   "freeze_dualheadadapter": freeze_dualheadadapter,
                   "early_stopping_patience": early_stopping_patience,
                   "mask_inter_organism_loss": mask_inter_organism_loss,
               })

    try:
        # Move models to device
        model.to(device)
        if hk_vae is not None:
            hk_vae.to(device)
        if adversary is not None:
            adversary.to(device)

        # Handle model freezing/unfreezing based on freeze_dualheadadapter parameter
        if freeze_dualheadadapter and adversary is not None:
            for param in model.parameters():
                param.requires_grad = False
            print("DualHeadAdapter model frozen - only training adversary")
        else:
            # Ensure model is unfrozen when freeze_dualheadadapter=False
            for param in model.parameters():
                param.requires_grad = True
        if not freeze_dualheadadapter:
            print("DualHeadAdapter model unfrozen - training main model")

        # Inform about distance matrix usage
        if adversary is not None:
            if distance_mat_df is not None:
                print("Using provided distance matrix for adversary training")
            else:
                print(
                    "Using binary same-organism mask for adversary training (0=same organism, 1=different)")

            if separate_adversary_in_warmup:
                print(
                    f"Adversary will train separately during first {warmup_epochs} epochs (detached gradients)")

        # Initialize paths and tracking variables
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f"best_{wandb.run.name}.pth")
        best_adversary_path = os.path.join(
            save_dir, f"best_adversary_{wandb.run.name}.pth") if adversary is not None else None
        top3_models_heap = []
        top3_adversaries_heap = []

        # Loss tracking
        best_val_loss = float('inf')
        best_adversary_loss = float('inf')
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        best_epoch = 0

        for epoch in range(epochs):
            # Check if we're in warmup phase
            is_in_warmup = (separate_adversary_in_warmup) and (
                epoch < warmup_epochs)

            # Set models to training mode
            model.train()
            if hk_vae is not None:
                hk_vae.train()
            if adversary is not None:
                adversary.train()

            # Initialize epoch tracking variables
            total_train_loss = 0
            total_adversary_loss = 0
            batch_count = 0
            adversary_batch_count = 0
            total_gamma_value = 0

            # Training loop
            for (hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes), indices in tqdm(train_dataloader, desc=f"Epoch {epoch+1} [Train]"):
                hk_batch_data, rr_batch_data = hk_batch_data.to(
                    device), rr_batch_data.to(device)

                if not freeze_dualheadadapter:
                    # === MAIN MODEL TRAINING ===
                    optimizer.zero_grad()

                    hk_emb, rr_emb, temperature, current_gamma = get_model_embeddings(
                        model, hk_batch_data, rr_batch_data, hk_vae, freeze_model=False
                    )
                    main_loss = organism_aware_clip_loss(
                        hk_emb, rr_emb, organism, hk_genes, rr_genes, temperature, current_gamma, mask_inter_organism_loss)
                    main_loss_value = main_loss.item()

                    # === ADVERSARIAL TRAINING ===
                    # The key insight: We want the main model to learn phylogeny-agnostic embeddings
                    # while the adversary tries to predict phylogenetic distances from these embeddings.
                    # This creates a minimax game:
                    # - Main model: minimize CLIP loss BUT maximize adversary loss (fool the adversary)
                    # - Adversary: minimize its own loss (predict phylogenetic distances correctly)
                    # The GradientReversalLayer in PhylogeneticAdversary automatically handles gradient reversal
                    adversary_loss = None
                    if adversary is not None:
                        adversary_optimizer.zero_grad()

                        hk_sub_batch, rr_sub_batch, organism_sub_batch, hk_genes_sub_batch, rr_genes_sub_batch = sample_sub_batch(
                            hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes, adversary_batch_size
                        )
                        hk_emb_sub, rr_emb_sub, _, _ = get_model_embeddings(
                            model, hk_sub_batch, rr_sub_batch, hk_vae, freeze_model=False
                        )
                        ground_truth_distances = build_distance_matrix(
                            organism_sub_batch, distance_mat_df).to(device)

                        sub_batch_size = hk_sub_batch.shape[0]
                        pred_hk = hk_emb_sub.unsqueeze(
                            1).expand(-1, sub_batch_size, -1)
                        pred_rr = rr_emb_sub.unsqueeze(
                            0).expand(sub_batch_size, -1, -1)
                        predicted_distances = adversary(
                            pred_hk, pred_rr).squeeze(-1)

                        if predicted_distances.dim() == 3:
                            predicted_distances = predicted_distances.squeeze(
                                -1)

                        adversary_loss = adversary_loss_excluding_diagonal(
                            predicted_distances, ground_truth_distances)

                        if is_in_warmup:
                            # During warmup: train adversary separately with detached gradients
                            # This prevents the main model from being affected by adversary gradients
                            hk_emb_sub_detached = hk_emb_sub.detach()
                            rr_emb_sub_detached = rr_emb_sub.detach()
                            hk_emb_sub_detached.requires_grad_(True)
                            rr_emb_sub_detached.requires_grad_(True)

                            # Train adversary with detached embeddings
                            pred_hk_detached = hk_emb_sub_detached.unsqueeze(
                                1).expand(-1, sub_batch_size, -1)
                            pred_rr_detached = rr_emb_sub_detached.unsqueeze(
                                0).expand(sub_batch_size, -1, -1)
                            predicted_distances_detached = adversary(
                                pred_hk_detached, pred_rr_detached).squeeze(-1)

                            if predicted_distances_detached.dim() == 3:
                                predicted_distances_detached = predicted_distances_detached.squeeze(
                                    -1)

                            adversary_loss_detached = adversary_loss_excluding_diagonal(
                                predicted_distances_detached, ground_truth_distances)

                            # Train adversary separately
                            adversary_loss_detached.backward()
                            adversary_optimizer.step()
                        else:
                            # Normal adversarial training: Add adversary loss to main loss (lambda is handled by GRL)
                            main_loss = main_loss + adversary_loss  # TODO: reinstate
                            # main_loss = adversary_loss  # TODO: for debugging

                    # Backward pass for combined loss
                    main_loss.backward()

                    # Step both optimizers separately for independent learning rates
                    optimizer.step()  # Main model optimizer
                    if (adversary is not None) and (not is_in_warmup):
                        adversary_optimizer.step()  # Adversary optimizer  # TODO: reinstate

                    scheduler.step()
                    total_train_loss += main_loss_value
                    batch_count += 1
                    total_gamma_value += current_gamma

                    # Track adversary loss for logging
                    if adversary is not None:
                        total_adversary_loss += adversary_loss.item()
                        adversary_batch_count += 1

                else:
                    # === ADVERSARY-ONLY TRAINING ===
                    if adversary is not None:
                        hk_sub_batch, rr_sub_batch, organism_sub_batch, hk_genes_sub_batch, rr_genes_sub_batch = sample_sub_batch(
                            hk_batch_data, rr_batch_data, organism, hk_genes, rr_genes, adversary_batch_size
                        )
                        hk_emb_sub, rr_emb_sub, _, _ = get_model_embeddings(
                            model, hk_sub_batch, rr_sub_batch, hk_vae, freeze_model=True
                        )
                        ground_truth_distances = build_distance_matrix(
                            organism_sub_batch, distance_mat_df).to(device)

                        sub_batch_size = hk_sub_batch.shape[0]
                        adversary_loss = train_adversary_step(
                            adversary, adversary_optimizer, hk_emb_sub, rr_emb_sub,
                            ground_truth_distances, sub_batch_size, freeze_model=True
                        )
                        total_adversary_loss += adversary_loss
                        adversary_batch_count += 1
                        batch_count += 1

            # Calculate average losses
            avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0
            avg_adversary_loss = total_adversary_loss / \
                adversary_batch_count if adversary_batch_count > 0 else 0
            avg_combined_loss = avg_train_loss + avg_adversary_loss
            avg_gamma = total_gamma_value / batch_count if batch_count > 0 else 0

            # === VALIDATION ===
            avg_val_loss, avg_adversary_val_loss, adversary_accuracy, avg_negative_ratio = run_validation(
                model, val_dataloader, device, hk_vae, adversary, adversary_batch_size, distance_mat_df, mask_inter_organism_loss)

            # === ACCURACY EVALUATION ===
            val_accuracy = None
            if evaluate_accuracy:
                print(f"Evaluating accuracy on validation set...")
                val_accuracy = evaluate_cognate_prediction(
                    model=model, dataloader=val_dataloader, device=device,
                    restrict_organism=True, hk_vae=hk_vae
                )

            # Log metrics and print progress
            log_training_metrics(
                epoch, epochs, avg_train_loss, avg_val_loss, avg_gamma, avg_adversary_loss, avg_combined_loss,
                val_accuracy, optimizer, freeze_dualheadadapter, adversary, evaluate_accuracy, avg_adversary_val_loss, adversary_accuracy, avg_negative_ratio, is_in_warmup
            )

            # Save best models and check for improvement
            model_improved = False
            adversary_improved = False

            # Save best main model (based on validation loss)
            if (not freeze_dualheadadapter) and (avg_val_loss < (best_val_loss - min_delta)):
                best_val_loss = avg_val_loss
                model_improved = True
                save_model_checkpoint(
                    model, best_model_path, "best model (loss)")

            # Save best adversary (based on adversary loss)
            if (adversary is not None) and (avg_adversary_loss < (best_adversary_loss - min_delta)):
                best_adversary_loss = avg_adversary_loss
                adversary_improved = True
                save_model_checkpoint(
                    adversary, best_adversary_path, "best adversary")

            # Save best model based on accuracy
            if (not freeze_dualheadadapter) and (evaluate_accuracy) and (val_accuracy is not None) and (val_accuracy > best_val_accuracy):
                best_val_accuracy = val_accuracy
                best_accuracy_model_path = os.path.join(
                    save_dir, f"best_accuracy_{wandb.run.name}.pth")
                save_model_checkpoint(
                    model, best_accuracy_model_path, "best model (accuracy)")

            # Update top-3 models (only if improved to save memory)
            if (not freeze_dualheadadapter) and (model_improved):
                update_top3_models(top3_models_heap, avg_val_loss,
                                   epoch, model.state_dict().copy())

            # Update top-3 adversaries (only if improved to save memory)
            if (adversary is not None) and (adversary_improved):
                update_top3_models(
                    top3_adversaries_heap, avg_adversary_loss, epoch, adversary.state_dict().copy())

            # Early stopping check (only when main model is not frozen)
            if not freeze_dualheadadapter:
                if model_improved or (adversary_improved and freeze_dualheadadapter):
                    epochs_without_improvement = 0
                    best_epoch = epoch
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                    print(f"Best epoch was {best_epoch+1}")
                    break
            else:
                # When main model is frozen, reset early stopping counter since we're only training adversary
                epochs_without_improvement = 0

        # Save the top-3 models
        if (not freeze_dualheadadapter) and (top3_models_heap):
            save_top3_models(top3_models_heap, save_dir, "model")

        # Save the top-3 adversaries
        if (adversary is not None) and (top3_adversaries_heap):
            save_top3_models(top3_adversaries_heap, save_dir, "adversary")

        # Load the best models
        load_best_models(model, adversary, best_model_path,
                         best_adversary_path, device, freeze_dualheadadapter)

    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    finally:
        wandb.finish()

    # Return both model and adversary if adversary is provided
    if adversary is not None:
        return model, adversary
    else:
        return model
