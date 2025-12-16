"""
Protein Embedding Generation Script

This script processes protein sequences from P2CS filtered groups and generates 
embeddings using the ESM-3 Forge model with batch processing for efficiency.
"""

import subprocess
import sys
import os
from typing import Optional, List
import re

import json

# def install_package(package_name, version=None):
#     if version is not None:
#         package_name = package_name + '==' + version
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])

# def pip_command(*args):
#     subprocess.check_call([sys.executable, "-m", "pip", *args])

# pip_command("uninstall", "transformers", "torch", "torchvision", "torchaudio", "-y")
# pip_command("install", "torch==1.12.1+cu113", "torchvision==0.13.1+cu113", "torchaudio==0.12.1",
#     "-f", "https://download.pytorch.org/whl/torch_stable.html",
#     "--quiet"
# )
# install_package('httpx')
# install_package('huggingface_hub')
# install_package('esm')

import gc
import pandas as pd
import numpy as np
import torch
import time
import warnings
from tqdm import tqdm
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk import batch_executor
import esm.sdk as esm_sdk
import glob

# Suppress specific ESM library warnings about tensor copying
# warnings.filterwarnings("ignore", message="To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()")
# warnings.filterwarnings("ignore", category=UserWarning, module="esm.utils.misc")


# =============================================================================
# CONSTANTS
# =============================================================================

# ESM-3 open embedding size is 1536
# ESM3-medium-2024-08 embedding size is 2560
# ESM3-large-2024-03 embedding size is 6144
DATA_PATH = "/zdata/user-data/noam/data/p2cs"  # Update this path as needed
# Make sure this file contains your Forge API token
API_KEY_FILE = "EvolutionaryScale.txt"
MODEL_NAME = "esm3-medium-2024-08"  # Updated to use Forge model
# Dedicated output directory
# INPUT_FILE = os.path.join(DATA_PATH, "merged_p2cs_data", "p2cs_filtered_groups.csv")
# INPUT_FILE = os.path.join(
#     DATA_PATH, "merged_p2cs_data", "_p2cs_orphan_data.pkl")
INPUT_FILE = os.path.join(
    DATA_PATH, "experimental_validation", "marginal_specificity_paper", "mutated_envZs.pkl")
OUTPUT_DIR = os.path.join(DATA_PATH, "embeddings",
                          "-".join(MODEL_NAME.split("-")[:2]))
EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR, f"mutated_envZs_mean_embeddings_{MODEL_NAME}.npy")
STATE_FILE = os.path.join(
    OUTPUT_DIR, f"mutated_envZs_mean_embeddings_{MODEL_NAME}_state.json")
EMBEDDING_SIZE = 2560  # ESM3-medium-2024-08 embedding size
MAX_SEQUENCE_LENGTH = 3500  # Maximum sequence length to avoid memory issues
SAVE_INTERVAL = 50  # Save embeddings more frequently to reduce memory usage
PBAR_UPDATE_INTERVAL = 50 * 5  # Update progress bar more frequently
MEMORY_CLEANUP_INTERVAL = 1  # Clear cache after every N batches to prevent OOM
FALLBACK_SEQUENCE_LENGTH = 500  # Fallback length for OOM sequences
BATCH_SIZE = 50  # Further reduced batch size to stay within memory limits
MAX_BATCH_TOTAL_LENGTH = 15000  # Further reduced to be more conservative
SKIP_EXISTING_BATCHES = True  # Skip sequences that already have saved batch files

# Rate limiting configuration - more aggressive to stay within 750 RPM limit
RATE_LIMIT_DELAY = 5  # delay between batches to respect rate limits

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def install_package(package_name, version=None):
    """Install a Python package using pip."""
    if version is not None:
        package_name = "{}=={}".format(package_name, version)
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", package_name, "--quiet"])


def pip_command(*args):
    """Execute a pip command with the given arguments."""
    subprocess.check_call([sys.executable, "-m", "pip"] + list(args))


def load_api_key(filepath):
    """Load API key from file."""
    with open(filepath, "r") as f:
        return f.read().strip()


def initialize_model(api_key, model_name):
    """Initialize the ESM-3 Forge model with the given API key."""
    return ESM3ForgeInferenceClient(
        model=model_name,
        url="https://forge.evolutionaryscale.ai",
        token=api_key
    )


def initialize_embeddings_file(filepath, embedding_size):
    """Initialize embeddings file if it doesn't exist or is incompatible."""
    # Ensure output directory exists
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    if not os.path.exists(filepath):
        # Create new empty file
        np.save(filepath, np.empty((0, embedding_size)))
    else:
        # Check existing file for compatibility
        try:
            existing_embeddings = np.load(filepath)
            if existing_embeddings.shape[1] != embedding_size:
                print(
                    f"Warning: Existing embeddings file has size {existing_embeddings.shape[1]}, expected {embedding_size}")
                backup_file = filepath.replace('.npy', '_backup.npy')
                np.save(backup_file, existing_embeddings)
                print(f"Backup created: {backup_file}")
                print("Creating new embeddings file with correct size...")
                np.save(filepath, np.empty((0, embedding_size)))
        except Exception as e:
            print(f"Error reading existing embeddings file: {e}")
            print("Creating new embeddings file...")
            np.save(filepath, np.empty((0, embedding_size)))


def setup_pytorch_memory():
    """Configure PyTorch memory allocation for better memory management."""
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def clear_memory_cache():
    """Clear memory cache based on device type."""
    # Force garbage collection
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Additional CUDA memory cleanup
        torch.cuda.synchronize()

    # Additional Python garbage collection
    gc.collect()


def load_processing_state(state_file):
    """Load persistent processing state from disk."""
    if not os.path.exists(state_file):
        return {
            "total_sequences_processed": 0,
            "next_batch_index": 0,
            "batch_lengths": {}
        }

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read state file {state_file}: {e}")
        return {
            "total_sequences_processed": 0,
            "next_batch_index": 0,
            "batch_lengths": {}
        }

    # Normalize state structure
    total_sequences = state.get("total_sequences_processed", 0)
    next_batch_index = state.get("next_batch_index", 0)
    batch_lengths = state.get("batch_lengths", {})

    # Ensure integer keys for batch lengths
    normalized_batch_lengths = {}
    for key, value in batch_lengths.items():
        try:
            normalized_batch_lengths[int(key)] = int(value)
        except Exception:
            pass

    return {
        "total_sequences_processed": int(total_sequences),
        "next_batch_index": int(next_batch_index),
        "batch_lengths": normalized_batch_lengths
    }


def save_processing_state(state_file, state):
    """Persist processing state to disk."""
    try:
        state_dir = os.path.dirname(state_file)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir, exist_ok=True)

        serializable_state = {
            "total_sequences_processed": int(state.get("total_sequences_processed", 0)),
            "next_batch_index": int(state.get("next_batch_index", 0)),
            "batch_lengths": {str(k): int(v) for k, v in state.get("batch_lengths", {}).items()}
        }

        with open(state_file, "w") as f:
            json.dump(serializable_state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not write state file {state_file}: {e}")


def estimate_tokens(sequences):
    """
    Estimate the number of tokens for a batch of sequences.
    Rough estimate: ~1.5 tokens per amino acid character.
    """
    total_chars = sum(len(seq) for seq in sequences)
    return int(total_chars * 1.5)


def wait_for_rate_limit(delay=RATE_LIMIT_DELAY):
    """Add delay to respect rate limits."""
    time.sleep(delay)


def preprocess_sequences(sequences):
    """
    Preprocess sequences to optimize for inference.

    Args:
        sequences: List of protein sequences

    Returns:
        List of tuples (original_index, processed_sequence) sorted by length
    """
    # Remove invalid sequences and sort by length for better batching
    valid_sequences = []
    for i, seq in enumerate(sequences):
        if seq and len(seq) >= 10:  # Filter out very short sequences
            # Remove invalid amino acid characters
            cleaned_seq = ''.join(c for c in seq.upper()
                                  if c in 'ACDEFGHIKLMNPQRSTVWY')
            if len(cleaned_seq) >= 10:
                valid_sequences.append((i, cleaned_seq))

    # Sort by length to process similar lengths together (can improve efficiency)
    valid_sequences.sort(key=lambda x: len(x[1]))

    return valid_sequences


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def embed_sequence(client: ESM3ForgeInferenceClient, sequence: str) -> np.ndarray:
    """
    Compute embedding for a single protein sequence using ESM-3 Forge model.

    Args:
        client: ESM-3 Forge inference client
        sequence: Protein sequence string

    Returns:
        Mean embedding vector as numpy array
    """
    with torch.no_grad():
        # Truncate very long sequences to avoid memory issues
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]

        # Clean sequence - remove any invalid characters
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned_seq = ''.join(c for c in sequence.upper() if c in valid_aa)

        if len(cleaned_seq) < 10:  # Skip very short sequences
            print(
                f"Warning: Sequence too short after cleaning: {len(cleaned_seq)} residues")
            return np.zeros(EMBEDDING_SIZE)

        try:
            protein = ESMProtein(sequence=cleaned_seq)
            protein_tensor = client.encode(protein)

            # Check for encoding errors (including rate limiting)
            if hasattr(protein_tensor, 'error') or str(type(protein_tensor)).endswith('ESMProteinError'):
                error_msg = protein_tensor.error if hasattr(
                    protein_tensor, 'error') else str(protein_tensor)
                print(
                    f"Warning: Encoding error for sequence of length {len(cleaned_seq)}")
                print(f"Error details: {error_msg}")

                # If it's a rate limit error, wait longer before retrying
                if "429" in str(error_msg) or "exceeded" in str(error_msg).lower():
                    print(f"Rate limit hit, waiting 60 seconds before retrying...")
                    time.sleep(60)
                    # Try encoding one more time after the wait
                    try:
                        protein_tensor = client.encode(protein)
                        if hasattr(protein_tensor, 'error') or str(type(protein_tensor)).endswith('ESMProteinError'):
                            print(
                                f"Still getting error after retry, skipping sequence")
                            return np.zeros(EMBEDDING_SIZE)
                    except Exception as retry_error:
                        print(f"Retry failed: {str(retry_error)}")
                        return np.zeros(EMBEDDING_SIZE)
                else:
                    return np.zeros(EMBEDDING_SIZE)

            # Get embeddings using logits with return_embeddings=True
            result = client.logits(protein_tensor, LogitsConfig(
                sequence=True, return_embeddings=True))

            # Check if result is an error object
            if hasattr(result, 'error'):
                print(
                    f"Warning: ESM API returned error for sequence of length {len(cleaned_seq)}")
                print(f"Error details: {result.error}")
                return np.zeros(EMBEDDING_SIZE)
            try:
                if hasattr(result, 'embeddings') and result.embeddings is not None:
                    # Convert tensor to float32 to avoid BFloat16 issues with NumPy
                    embeddings_tensor = result.embeddings
                    if hasattr(embeddings_tensor, 'dtype') and hasattr(embeddings_tensor, 'float'):
                        # Convert BFloat16 or other unsupported types to float32
                        embeddings_tensor = embeddings_tensor.float()

                    # Convert to numpy array
                    if hasattr(embeddings_tensor, 'cpu'):
                        # If it's a PyTorch tensor, move to CPU first
                        embedding = embeddings_tensor.cpu().numpy()
                    else:
                        embedding = np.array(
                            embeddings_tensor, dtype=np.float32)

                    # # Debug: Print embedding shape to understand what we're getting
                    # print(f"Debug: Raw embedding shape: {embedding.shape} for sequence length {len(cleaned_seq)}")

                    # Ensure we always get a fixed-size embedding through mean pooling
                    if len(embedding.shape) == 1:
                        # Already a 1D vector (unlikely for per-residue embeddings)
                        if embedding.shape[0] == EMBEDDING_SIZE:
                            mean_embedding = embedding
                        else:
                            print(
                                f"Warning: Unexpected 1D embedding size {embedding.shape[0]}, expected {EMBEDDING_SIZE}")
                            mean_embedding = np.zeros(EMBEDDING_SIZE)
                    elif len(embedding.shape) == 2:
                        # 2D tensor: (sequence_length, embedding_dim) - need to mean pool
                        if embedding.shape[1] == EMBEDDING_SIZE:
                            # Mean pool over the sequence dimension (axis=0)
                            mean_embedding = embedding.mean(axis=0)
                            # print(f"Debug: Mean pooled from {embedding.shape} to {mean_embedding.shape}")
                        else:
                            print(
                                f"Warning: Unexpected embedding dimension {embedding.shape[1]}, expected {EMBEDDING_SIZE}")
                            mean_embedding = np.zeros(EMBEDDING_SIZE)
                    elif len(embedding.shape) == 3:
                        # 3D tensor: (batch_size, sequence_length, embedding_dim) - need to mean pool
                        if embedding.shape[2] == EMBEDDING_SIZE:
                            # Remove batch dimension (should be 1) and mean pool over sequence dimension
                            if embedding.shape[0] == 1:
                                # Squeeze out batch dimension and mean pool over sequence
                                # Now (sequence_length, embedding_dim)
                                sequence_embeddings = embedding.squeeze(0)
                                mean_embedding = sequence_embeddings.mean(
                                    axis=0)  # Mean pool over sequence
                                # print(f"Debug: Mean pooled from {embedding.shape} to {mean_embedding.shape}")
                            else:
                                print(
                                    f"Warning: Unexpected batch size {embedding.shape[0]}, expected 1")
                                mean_embedding = np.zeros(EMBEDDING_SIZE)
                        else:
                            print(
                                f"Warning: Unexpected embedding dimension {embedding.shape[2]}, expected {EMBEDDING_SIZE}")
                            mean_embedding = np.zeros(EMBEDDING_SIZE)
                    else:
                        print(
                            f"Warning: Unexpected embedding shape {embedding.shape}, using zero embedding")
                        mean_embedding = np.zeros(EMBEDDING_SIZE)

                    # Final validation
                    if mean_embedding.shape != (EMBEDDING_SIZE,):
                        print(
                            f"Warning: Final embedding shape {mean_embedding.shape}, forcing to correct size")
                        mean_embedding = np.zeros(EMBEDDING_SIZE)

                else:
                    print(
                        f"Warning: No embeddings returned for sequence of length {len(cleaned_seq)}")
                    return np.zeros(EMBEDDING_SIZE)
            except Exception as tensor_error:
                print(
                    f"Warning: Error extracting embeddings from tensor: {str(tensor_error)}")
                print(
                    f"Tensor type: {type(result.embeddings) if hasattr(result, 'embeddings') else 'No embeddings attr'}")
                return np.zeros(EMBEDDING_SIZE)

            # Clean up tensors immediately and aggressively
            del protein, protein_tensor, result
            if 'embedding' in locals():
                del embedding
            if 'embeddings_tensor' in locals():
                del embeddings_tensor

            # Force garbage collection for this sequence
            gc.collect()

            return mean_embedding

        except Exception as e:
            print(
                f"Error processing sequence of length {len(cleaned_seq)}: {str(e)}")
            return np.zeros(EMBEDDING_SIZE)


def test_embedding_size(model, seq):
    """
    Test the embedding size for a given sequence.

    Args:
        model: ESM-3 Forge inference client
        seq: Protein sequence string

    Returns:
        Size of the embedding vector
    """
    print(f"Testing embedding size for sequence of length {len(seq)}")
    embedding = embed_sequence(model, seq)
    print(f"Resulting embedding shape: {embedding.shape}")
    print(
        f"Embedding size: {embedding.shape[-1] if hasattr(embedding, 'shape') else None}")

    # Calculate expected file size for 1000 sequences
    single_embedding_bytes = embedding.nbytes if hasattr(
        embedding, 'nbytes') else len(embedding) * 4  # 4 bytes per float32
    total_size_mb = (single_embedding_bytes * 1000) / (1024 * 1024)
    print(f"Single embedding memory: {single_embedding_bytes} bytes")
    print(f"Expected file size for 1000 sequences: {total_size_mb:.2f} MB")

    return embedding.shape[-1] if hasattr(embedding, 'shape') else None


def save_embeddings_batch(embeddings_list, embeddings_file, batch_index=None):
    """Save a batch of embeddings using separate batch files to avoid memory issues."""
    if not embeddings_list:
        return None, 0

    # Ensure output directory exists
    output_dir = os.path.dirname(embeddings_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Convert list to numpy array once
    new_embeddings = np.vstack(embeddings_list)

    saved_path = None

    # Use separate batch files to avoid loading large arrays
    if batch_index is not None:
        batch_file = embeddings_file.replace(
            '.npy', f'_batch_{batch_index:06d}.npy')
        np.save(batch_file, new_embeddings)
        print(f"Saved batch {batch_index} to {batch_file}")
        saved_path = batch_file
    else:
        # Fallback to original file for final save
        np.save(embeddings_file, new_embeddings)
        saved_path = embeddings_file

    saved_count = new_embeddings.shape[0]

    # Clear variables to free memory
    del new_embeddings

    # Force garbage collection after saving
    gc.collect()

    return saved_path, saved_count


def combine_batch_files(embeddings_file, state_file=STATE_FILE):
    """Combine all batch files into final embeddings file."""

    # Ensure output directory exists
    output_dir = os.path.dirname(embeddings_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    base_name = embeddings_file.replace('.npy', '')
    batch_pattern = f"{base_name}_batch_*.npy"
    batch_files = sorted(glob.glob(batch_pattern))

    if not batch_files:
        print("No batch files found to combine.")
        return 0

    # Extract batch indices, record lengths, and detect anomalies
    batch_indices = []
    batch_file_map = {}
    batch_lengths = {}

    for batch_file in batch_files:
        match = re.search(r'_batch_(\d+)\.npy$', batch_file)
        if not match:
            print(
                f"Warning: Skipping unrecognized batch file name {os.path.basename(batch_file)}")
            continue

        batch_idx = int(match.group(1))
        batch_file_map[batch_idx] = batch_file
        batch_indices.append(batch_idx)

        try:
            batch_array = np.load(batch_file, mmap_mode='r')
            batch_len = len(batch_array)
        except Exception as e:
            print(
                f"Warning: Could not read batch file {os.path.basename(batch_file)}: {e}")
            batch_len = 0

        batch_lengths[batch_idx] = batch_len

    if not batch_indices:
        print("No valid batch files found to combine.")
        return 0

    batch_indices.sort()
    batch_files_sorted = [batch_file_map[idx] for idx in batch_indices]

    # Check for gaps in batch indices
    expected_indices = set(range(batch_indices[0], batch_indices[-1] + 1))
    missing_indices = expected_indices - set(batch_indices)

    if missing_indices:
        print(
            f"WARNING: Found {len(missing_indices)} missing batch file(s) in the middle: {sorted(missing_indices)}")
        print(
            "The combined file may have gaps because sequences from the missing batches cannot be recovered.")
        response = input("Continue combining anyway? (y/n): ")
        if response.lower() != 'y':
            print("Combination cancelled.")
            return 0

    print(
        f"Combining {len(batch_files_sorted)} batch files into {embeddings_file}...")

    # Load and combine batch files one by one in order
    all_embeddings = []
    for i, batch_file in enumerate(batch_files_sorted):
        print(
            f"Loading batch file {i + 1}/{len(batch_files_sorted)}: {os.path.basename(batch_file)}")
        batch_embeddings = np.load(batch_file)
        all_embeddings.append(batch_embeddings)

        # Clear memory periodically
        if (i + 1) % 10 == 0:  # Every 10 files
            gc.collect()

    # Combine all embeddings
    if all_embeddings:
        print("Combining all embeddings...")
        final_embeddings = np.vstack(all_embeddings)
    else:
        print("No embeddings loaded from batch files; creating empty array.")
        final_embeddings = np.empty((0, EMBEDDING_SIZE))

    # Save final combined file
    np.save(embeddings_file, final_embeddings)
    final_count = final_embeddings.shape[0]
    print(
        f"Saved final embeddings to {embeddings_file} with shape {final_embeddings.shape}")

    # Clean up batch files
    for batch_file in batch_files_sorted:
        os.remove(batch_file)
        print(f"Removed batch file: {os.path.basename(batch_file)}")

    # Update processing state after combining
    next_batch_index = batch_indices[-1] + 1 if batch_indices else 0
    combined_sequences = sum(batch_lengths.values())
    updated_state = {
        "total_sequences_processed": max(final_count, combined_sequences),
        "next_batch_index": next_batch_index,
        "batch_lengths": {}
    }
    save_processing_state(state_file, updated_state)

    # Final cleanup
    del all_embeddings, final_embeddings
    gc.collect()

    return final_count


def process_sequences_batch(sequences, model, embeddings_file, skip_existing_batches=False):
    """Process sequences in batches using the Forge Batch Executor."""

    total_sequences_available = len(sequences)

    # Check for existing embeddings to resume from
    existing_count = 0
    if os.path.exists(embeddings_file) and os.path.getsize(embeddings_file) > 0:
        existing_embeddings = np.load(embeddings_file)
        existing_count = len(existing_embeddings)
        print(
            f"Found {existing_count} existing embeddings. Resuming from sequence {existing_count}")
        del existing_embeddings  # Free memory immediately
        gc.collect()

    # Load persistent state
    state = load_processing_state(STATE_FILE)
    processed_sequences_state = state.get("total_sequences_processed", 0)
    next_batch_index_state = state.get("next_batch_index", 0)

    # Inspect existing batch files
    base_name = embeddings_file.replace('.npy', '')
    batch_pattern = f"{base_name}_batch_*.npy"
    existing_batch_files = sorted(glob.glob(batch_pattern))

    batch_lengths = {}
    existing_batch_indices = set()

    if existing_batch_files:
        print(
            f"Found {len(existing_batch_files)} existing batch files. Inspecting...")
        for batch_file in existing_batch_files:
            match = re.search(r'_batch_(\d+)\.npy$', batch_file)
            if not match:
                print(
                    f"Warning: Skipping unrecognized batch file name {os.path.basename(batch_file)}")
                continue

            batch_idx = int(match.group(1))
            existing_batch_indices.add(batch_idx)
            try:
                batch_len = len(np.load(batch_file, mmap_mode='r'))
            except Exception as e:
                print(
                    f"Warning: Could not read batch file {os.path.basename(batch_file)}: {e}")
                batch_len = 0
            batch_lengths[batch_idx] = batch_len

        if existing_batch_indices:
            min_batch_idx = min(existing_batch_indices)
            max_batch_idx = max(existing_batch_indices)
            expected_indices = set(range(min_batch_idx, max_batch_idx + 1))
            missing_indices = expected_indices - existing_batch_indices

            if missing_indices:
                print(
                    f"WARNING: Found {len(missing_indices)} missing batch file(s) in the middle: {sorted(missing_indices)}")
                print(
                    "This may indicate incomplete processing. Some sequences might need to be regenerated.")
            else:
                print(
                    f"All batch files are contiguous from {min_batch_idx} to {max_batch_idx}.")

    total_sequences_in_batches = sum(batch_lengths.values())
    if total_sequences_in_batches:
        print(
            f"Existing batch files represent {total_sequences_in_batches} embedded sequence(s).")

    # Determine the starting point and batch counter
    next_batch_index_from_files = max(
        existing_batch_indices) + 1 if existing_batch_indices else 0
    batch_counter = max(next_batch_index_state, next_batch_index_from_files)

    if skip_existing_batches:
        base_start_offset = existing_count + total_sequences_in_batches
    else:
        base_start_offset = existing_count

    start_offset = max(base_start_offset, processed_sequences_state)
    if start_offset > total_sequences_available:
        start_offset = total_sequences_available

    print(f"Resuming embedding from global sequence index {start_offset}.")

    # Prepare current state tracking
    current_batch_lengths = dict(batch_lengths)
    processed_sequences_current = start_offset

    # Persist the normalized state before processing
    save_processing_state(STATE_FILE, {
        "total_sequences_processed": processed_sequences_current,
        "next_batch_index": batch_counter,
        "batch_lengths": current_batch_lengths
    })

    if processed_sequences_current >= total_sequences_available:
        print("All sequences already processed!")
        return

    sequences_to_process = sequences[processed_sequences_current:]

    print(
        f"Processing {len(sequences_to_process)} sequences in batches (starting from index {processed_sequences_current})")

    # Process sequences in batches using the batch executor
    batch_size = min(BATCH_SIZE, len(sequences_to_process))
    all_embeddings = []
    with tqdm(total=len(sequences_to_process), desc="Embedding sequences") as pbar:
        for i in range(0, len(sequences_to_process), batch_size):
            batch_sequences = sequences_to_process[i:i + batch_size]

            try:
                # Use batch executor for efficient processing
                with batch_executor() as executor:
                    batch_outputs = executor.execute_batch(
                        user_func=embed_sequence,
                        client=model,
                        sequence=batch_sequences
                    )

                # Convert outputs to embeddings
                print(
                    f"Processing batch {i // batch_size + 1} with {len(batch_sequences)} sequence(s)")
                batch_embeddings = []
                for j, output in enumerate(batch_outputs):
                    if isinstance(output, np.ndarray) and len(output.shape) > 0 and output.shape[-1] == EMBEDDING_SIZE:
                        batch_embeddings.append(output)
                    else:
                        # Handle errors or invalid outputs - try individual processing
                        global_index = processed_sequences_current + i + j
                        print(
                            f"Warning: Invalid output for sequence {global_index}, shape: {output.shape if isinstance(output, np.ndarray) else type(output)}")
                        print(f"Expected embedding size: {EMBEDDING_SIZE}")
                        try:
                            individual_embedding = embed_sequence(
                                model, batch_sequences[j])
                            batch_embeddings.append(individual_embedding)
                        except Exception as e:
                            print(f"Error in individual processing: {str(e)}")
                            batch_embeddings.append(np.zeros(EMBEDDING_SIZE))

                all_embeddings.extend(batch_embeddings)

                # Clear memory after every batch to prevent OOM
                clear_memory_cache()

                # Save embeddings more frequently using batch files to avoid memory issues
                should_save_batch = len(all_embeddings) >= SAVE_INTERVAL or (
                    i + batch_size) >= len(sequences_to_process)
                if should_save_batch and all_embeddings:
                    saved_path, saved_count = save_embeddings_batch(
                        all_embeddings, embeddings_file, batch_counter)

                    if saved_count > 0:
                        current_batch_lengths[batch_counter] = saved_count
                        processed_sequences_current += saved_count
                        batch_counter += 1

                        save_processing_state(STATE_FILE, {
                            "total_sequences_processed": processed_sequences_current,
                            "next_batch_index": batch_counter,
                            "batch_lengths": current_batch_lengths
                        })

                    all_embeddings = []
                    clear_memory_cache()  # Clear again after saving

                # Update progress bar
                pbar.update(len(batch_sequences))

                # Rate limiting - longer delay to avoid hitting limits
                wait_for_rate_limit()

            except Exception as e:
                error_str = str(e)
                print(
                    f"Error processing batch starting at global index {processed_sequences_current + i}: {error_str}")

                # Check if it's a rate limit error
                if "429" in error_str or "exceeded" in error_str.lower() or "rate" in error_str.lower():
                    print(
                        "Rate limit detected in batch processing. Waiting 60 seconds...")
                    time.sleep(60)
                    print("Retrying batch after rate limit wait...")

                    # Retry the batch once
                    try:
                        with batch_executor() as executor:
                            batch_outputs = executor.execute_batch(
                                user_func=embed_sequence,
                                client=model,
                                sequence=batch_sequences
                            )

                        batch_embeddings = []
                        for j, output in enumerate(batch_outputs):
                            if isinstance(output, np.ndarray) and len(output.shape) > 0 and output.shape[-1] == EMBEDDING_SIZE:
                                batch_embeddings.append(output)
                            else:
                                batch_embeddings.append(
                                    np.zeros(EMBEDDING_SIZE))

                        all_embeddings.extend(batch_embeddings)
                        clear_memory_cache()  # Clear memory after retry

                    except Exception as retry_error:
                        print(f"Batch retry failed: {str(retry_error)}")
                        print(
                            "Falling back to individual sequence processing for this batch...")

                        # Fallback: process sequences individually with delays
                        batch_embeddings = []
                        for j, seq in enumerate(batch_sequences):
                            try:
                                embedding = embed_sequence(model, seq)
                                batch_embeddings.append(embedding)
                                # Add delay between individual sequences to avoid rate limits
                                time.sleep(0.1)
                                # Clear memory after each individual sequence in fallback
                                if j % 5 == 0:  # Every 5 sequences
                                    clear_memory_cache()
                            except Exception as e2:
                                print(
                                    f"Error processing individual sequence {processed_sequences_current + i + j}: {str(e2)}")
                                batch_embeddings.append(
                                    np.zeros(EMBEDDING_SIZE))

                        all_embeddings.extend(batch_embeddings)
                        clear_memory_cache()  # Clear memory after fallback processing
                else:
                    print(
                        "Falling back to individual sequence processing for this batch...")

                    # Fallback: process sequences individually
                    batch_embeddings = []
                    for j, seq in enumerate(batch_sequences):
                        try:
                            embedding = embed_sequence(model, seq)
                            batch_embeddings.append(embedding)
                            # Clear memory every few sequences in fallback
                            if j % 5 == 0:
                                clear_memory_cache()
                        except Exception as e2:
                            print(
                                f"Error processing individual sequence {processed_sequences_current + i + j}: {str(e2)}")
                            batch_embeddings.append(np.zeros(EMBEDDING_SIZE))

                    all_embeddings.extend(batch_embeddings)
                    clear_memory_cache()  # Clear memory after fallback

                pbar.update(len(batch_sequences))

        # End of for loop

    # Save any remaining embeddings (should be empty due to saving condition, but kept for safety)
    if all_embeddings:
        saved_path, saved_count = save_embeddings_batch(
            all_embeddings, embeddings_file, batch_counter)
        if saved_count > 0:
            current_batch_lengths[batch_counter] = saved_count
            processed_sequences_current += saved_count
            batch_counter += 1

        save_processing_state(STATE_FILE, {
            "total_sequences_processed": processed_sequences_current,
            "next_batch_index": batch_counter,
            "batch_lengths": current_batch_lengths
        })

    # Combine all batch files into final embeddings file
    print("Combining batch files into final embeddings file...")
    combine_batch_files(embeddings_file, state_file=STATE_FILE)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to process protein sequences and generate embeddings."""
    print(f"Device: {DEVICE}")
    print("Loading data and model...")

    # Setup memory management
    setup_pytorch_memory()

    # Load data
    # p2cs_filtered_groups = pd.read_csv(os.path.join(DATA_PATH, "p2cs_filtered_groups.csv"))
    sequences_df = pd.read_pickle(
        os.path.join(DATA_PATH, INPUT_FILE))
    sequences = sequences_df['aa_sequence'].tolist()

    # Load API key and initialize model
    api_key = load_api_key(API_KEY_FILE)
    model = initialize_model(api_key, MODEL_NAME)

    # # Test embedding size for the model
    # print(f"Testing embedding size for model '{MODEL_NAME}'...")
    # embedding_size = test_embedding_size(model, sequences[0])
    # print(f"Confirmed embedding size: {embedding_size}")

    # print(f"Testing batch embedding with sample sequences...")
    # sample_sequences = sequences[:3] if len(sequences) >= 3 else sequences  # Test with first 3 sequences

    # # Create a temporary test file to see the output
    # test_embeddings_file = os.path.join(OUTPUT_DIR, f"test_embeddings_{MODEL_NAME}.npy")
    # process_sequences_batch(sample_sequences, model, test_embeddings_file)

    # # Check the test file size
    # if os.path.exists(test_embeddings_file):
    #     file_size_bytes = os.path.getsize(test_embeddings_file)
    #     file_size_mb = file_size_bytes / (1024 * 1024)
    #     print(f"Test embeddings file size: {file_size_mb:.2f} MB for {len(sample_sequences)} sequences")

    #     # Load and check the shape
    #     test_embeddings = np.load(test_embeddings_file)
    #     print(f"Test embeddings array shape: {test_embeddings.shape}")
    #     print(f"Per-sequence embedding size: {test_embeddings.shape[1] if len(test_embeddings.shape) > 1 else 'Invalid shape'}")
    #     del test_embeddings  # Clean up memory

    # print("Test completed. Do you want to continue with full processing? (Comment out the return below)")
    # return  # Remove this line to continue with full processing

    # Initialize embeddings file
    initialize_embeddings_file(EMBEDDINGS_FILE, EMBEDDING_SIZE)

    # Process sequences using batch executor
    process_sequences_batch(
        sequences, model, EMBEDDINGS_FILE, skip_existing_batches=SKIP_EXISTING_BATCHES)

    print("Embedding generation completed!")


if __name__ == "__main__":
    main()
