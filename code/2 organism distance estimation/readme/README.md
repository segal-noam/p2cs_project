# P2CS Distance Estimation Pipeline

This pipeline downloads full genomes from NCBI and computes pairwise distance matrices using Mash for multiple k-mer sizes.

## Overview

The pipeline consists of several scripts that work together to:

1. **Extract organism names** from the p2cs dataset
2. **Download full genomes** from NCBI for each organism
3. **Sketch genomes** with Mash using multiple k-mer sizes
4. **Compute distance matrices** for each k-mer size

## Scripts

### Core Scripts

- `download_genomes.py` - Downloads full genomes from NCBI using organism names from the p2cs dataset
- `mash_distance_analysis.py` - Sketches genomes and computes distance matrices using Mash
- `aggregate_distances.py` - Aggregates genome-level distances to organism-level matrices

### Utility Scripts

- File validation and cleanup is now integrated into the download script

## Usage

### Quick Start

```bash
# Activate conda environment
conda activate python310

# Run complete pipeline
python run_full_analysis.py

# Run in test mode (limited organisms)
python run_full_analysis.py --test-mode

# Skip genome download (if already done)
python run_full_analysis.py --skip-download

# Skip Mash analysis (if already done)
python run_full_analysis.py --skip-mash
```

### Step-by-Step Execution

```bash
# 1. Download genomes (this may take a while)
python download_genomes.py

# 2. Run Mash analysis
python mash_distance_analysis.py

# 3. Aggregate distances (optional)
python aggregate_distances.py

# 4. File validation is automatic in the download script
```

## Configuration

### K-mer Sizes

The pipeline uses 5 different k-mer sizes to capture different levels of sequence similarity:

- **k=15**: Captures highly divergent sequences
- **k=21**: Good for species-level comparisons
- **k=31**: Standard for genus-level comparisons
- **k=51**: Good for strain-level comparisons
- **k=101**: Captures very closely related sequences

### Output Structure

```
/zdata/user-data/noam/data/p2cs/
├── full_genomes/                    # Downloaded genome FASTA files
│   ├── organism1_genome1.fasta
│   ├── organism2_genome1.fasta
│   └── ...
├── mash_analysis/
│   ├── sketches/                    # Mash sketch files
│   │   ├── genomes_k15.msh
│   │   ├── genomes_k21.msh
│   │   └── ...
│   └── distance_matrices/           # Distance matrices
│       ├── distance_matrix_k15.csv
│       ├── distance_matrix_k21.csv
│       └── ...
```

## Dependencies

### Required Software

- **Mash**: For genome sketching and distance computation
- **Python 3.10**: With conda environment
- **NCBI Entrez API**: For genome downloads

### Python Packages

- pandas
- numpy
- biopython

## Notes

### Rate Limiting

- NCBI API: 3 requests per second (automatically handled)
- Genome downloads: 1 second delay between organisms
- Mash operations: No rate limiting needed

### Error Handling

- Failed downloads are logged and saved to `failed_organisms.txt`
- Corrupted/empty FASTA files are automatically detected and removed
- Partial results are preserved if the pipeline is interrupted
- Each step can be run independently
- Mash analysis skips invalid files and continues processing

### Performance

- **Genome download**: ~1-2 minutes per organism (depends on genome size)
- **Mash sketching**: ~1-2 minutes per k-mer size
- **Distance computation**: ~1-5 minutes per k-mer size

For 2649 organisms with 5 k-mer sizes, expect:
- Download time: ~44-88 hours
- Analysis time: ~10-25 minutes

## Troubleshooting

### Common Issues

1. **NCBI API errors**: Check internet connection and API limits
2. **Mash not found**: Ensure Mash is installed and in PATH
3. **Memory issues**: Reduce batch size or use fewer organisms
4. **Disk space**: Ensure sufficient space for genomes and sketches

### File Validation

The pipeline automatically handles corrupted files:

- **Download script**: Validates each downloaded file and removes corrupted ones
- **Mash analysis**: Skips invalid files and continues processing
- **Automatic cleanup**: Corrupted files are detected and removed during download

