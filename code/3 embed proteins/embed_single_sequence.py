"""
Minimal script to embed a single protein sequence using ESM-3 Forge model
"""

import numpy as np
import torch
import gc
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient

# Configuration
API_KEY_FILE = "EvolutionaryScale.txt"  # File containing your Forge API token
MODEL_NAME = "esm3-medium-2024-08"
EMBEDDING_SIZE = 2560  # esm3-medium-2024-08 embedding size
MAX_SEQUENCE_LENGTH = 1500

def load_api_key(filepath):
    """Load API key from file."""
    with open(filepath, "r") as f:
        return f.read().strip()

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
        # Truncate very long sequences
        if len(sequence) > MAX_SEQUENCE_LENGTH:
            sequence = sequence[:MAX_SEQUENCE_LENGTH]
            print(f"Truncated sequence to {MAX_SEQUENCE_LENGTH} residues")
        
        # Clean sequence - remove any invalid characters
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned_seq = ''.join(c for c in sequence.upper() if c in valid_aa)
        
        if len(cleaned_seq) < 10:
            print(f"Warning: Sequence too short after cleaning: {len(cleaned_seq)} residues")
            return np.zeros(EMBEDDING_SIZE)
        
        print(f"Processing sequence of length {len(cleaned_seq)}")
        
        try:
            # Encode the protein sequence
            protein = ESMProtein(sequence=cleaned_seq)
            protein_tensor = client.encode(protein)
            
            # Check for encoding errors
            if hasattr(protein_tensor, 'error'):
                print(f"Encoding error: {protein_tensor.error}")
                return np.zeros(EMBEDDING_SIZE)
            
            # Get embeddings
            result = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
            
            # Check if result is an error object
            if hasattr(result, 'error'):
                print(f"API error: {result.error}")
                return np.zeros(EMBEDDING_SIZE)
            
            if hasattr(result, 'embeddings') and result.embeddings is not None:
                # Convert tensor to float32 and numpy
                embeddings_tensor = result.embeddings
                if hasattr(embeddings_tensor, 'dtype') and hasattr(embeddings_tensor, 'float'):
                    embeddings_tensor = embeddings_tensor.float()
                
                if hasattr(embeddings_tensor, 'cpu'):
                    embedding = embeddings_tensor.cpu().numpy()
                else:
                    embedding = np.array(embeddings_tensor, dtype=np.float32)
                
                print(f"Raw embedding shape: {embedding.shape}")
                
                # Handle different embedding shapes and mean pool
                if len(embedding.shape) == 1:
                    # Already 1D
                    if embedding.shape[0] == EMBEDDING_SIZE:
                        mean_embedding = embedding
                    else:
                        print(f"Unexpected 1D embedding size {embedding.shape[0]}, expected {EMBEDDING_SIZE}")
                        mean_embedding = np.zeros(EMBEDDING_SIZE)
                elif len(embedding.shape) == 2:
                    # 2D: (sequence_length, embedding_dim)
                    if embedding.shape[1] == EMBEDDING_SIZE:
                        mean_embedding = embedding.mean(axis=0)
                        print(f"Mean pooled from {embedding.shape} to {mean_embedding.shape}")
                    else:
                        print(f"Unexpected embedding dimension {embedding.shape[1]}, expected {EMBEDDING_SIZE}")
                        mean_embedding = np.zeros(EMBEDDING_SIZE)
                elif len(embedding.shape) == 3:
                    # 3D: (batch_size, sequence_length, embedding_dim)
                    if embedding.shape[2] == EMBEDDING_SIZE and embedding.shape[0] == 1:
                        sequence_embeddings = embedding.squeeze(0)  # Remove batch dim
                        mean_embedding = sequence_embeddings.mean(axis=0)  # Mean pool over sequence
                        print(f"Mean pooled from {embedding.shape} to {mean_embedding.shape}")
                    else:
                        print(f"Unexpected 3D embedding shape {embedding.shape}")
                        mean_embedding = np.zeros(EMBEDDING_SIZE)
                else:
                    print(f"Unexpected embedding shape {embedding.shape}")
                    mean_embedding = np.zeros(EMBEDDING_SIZE)
                
                # Final validation
                if mean_embedding.shape != (EMBEDDING_SIZE,):
                    print(f"Final embedding shape {mean_embedding.shape} is incorrect, using zero embedding")
                    mean_embedding = np.zeros(EMBEDDING_SIZE)
                
                # Clean up
                del protein, protein_tensor, result, embedding, embeddings_tensor
                gc.collect()
                
                return mean_embedding
            else:
                print("No embeddings returned from API")
                return np.zeros(EMBEDDING_SIZE)
                
        except Exception as e:
            print(f"Error processing sequence: {str(e)}")
            return np.zeros(EMBEDDING_SIZE)

def main():
    """Main function to embed a single sequence and print results."""
    
    # Example protein sequence (you can change this)
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print(f"ESM-3 Forge Single Sequence Embedding")
    print(f"Model: {MODEL_NAME}")
    print(f"Expected embedding size: {EMBEDDING_SIZE}")
    print(f"Input sequence: {test_sequence}")
    print(f"Sequence length: {len(test_sequence)}")
    print("-" * 50)
    
    try:
        # Load API key and initialize model
        api_key = load_api_key(API_KEY_FILE)
        client = ESM3ForgeInferenceClient(
            model=MODEL_NAME, 
            url="https://forge.evolutionaryscale.ai", 
            token=api_key
        )
        print("✓ Model initialized successfully")
        
        # Get embedding
        embedding = embed_sequence(client, test_sequence)
        
        print("-" * 50)
        print("RESULTS:")
        print(f"Final embedding shape: {embedding.shape}")
        print(f"Embedding type: {type(embedding)}")
        print(f"Data type: {embedding.dtype}")
        print(f"Memory usage: {embedding.nbytes} bytes ({embedding.nbytes / 1024:.2f} KB)")
        
        # Print some statistics about the embedding
        print(f"Min value: {embedding.min():.6f}")
        print(f"Max value: {embedding.max():.6f}")
        print(f"Mean value: {embedding.mean():.6f}")
        print(f"Std deviation: {embedding.std():.6f}")
        
        # Print first and last 10 values
        print(f"First 10 values: {embedding[:10]}")
        print(f"Last 10 values: {embedding[-10:]}")
        
        # Check if it's a zero embedding (indicates an error)
        if np.allclose(embedding, 0):
            print("⚠️  WARNING: Got zero embedding - there may have been an error")
        else:
            print("✓ Successfully generated non-zero embedding")
            
    except FileNotFoundError:
        print(f"❌ Error: API key file '{API_KEY_FILE}' not found")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
