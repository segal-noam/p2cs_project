#!/usr/bin/env python3
"""
Script to identify and clean up corrupted/empty FASTA files.
"""

import os
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_fasta(fasta_file):
    """Check if a FASTA file is valid and contains sequences."""
    try:
        # Check file size (must be > 100 bytes)
        if fasta_file.stat().st_size < 100:
            return False, "File too small"
        
        # Check if file contains FASTA sequences
        with open(fasta_file, 'r') as f:
            lines = f.readlines()
        
        # Must have at least one header line and one sequence line
        has_header = any(line.startswith('>') for line in lines)
        has_sequence = any(not line.startswith('>') and line.strip() for line in lines)
        
        if not has_header:
            return False, "No FASTA headers found"
        if not has_sequence:
            return False, "No sequence data found"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error reading file: {e}"

def clean_corrupted_files(genomes_dir, logs_dir, dry_run=True):
    """
    Identify and optionally remove corrupted FASTA files.
    
    Args:
        genomes_dir: Directory containing FASTA files
        logs_dir: Directory to save the log of removed files
        dry_run: If True, only report files, don't delete them
    """
    genomes_path = Path(genomes_dir)
    logs_path = Path(logs_dir)
    
    if not genomes_path.exists():
        logger.error(f"Directory not found: {genomes_dir}")
        return 0, 0
    
    # Create logs directory if it doesn't exist
    logs_path.mkdir(parents=True, exist_ok=True)
    
    genome_files = list(genomes_path.glob("*.fasta"))
    logger.info(f"Checking {len(genome_files)} FASTA files...")
    
    corrupted_files = []
    valid_files = []
    
    for genome_file in genome_files:
        is_valid, reason = is_valid_fasta(genome_file)
        
        if is_valid:
            valid_files.append(genome_file)
        else:
            corrupted_files.append((genome_file, reason))
            logger.warning(f"Corrupted: {genome_file.name} - {reason}")
    
    logger.info(f"Found {len(valid_files)} valid files and {len(corrupted_files)} corrupted files")
    
    if corrupted_files:
        logger.info("\nCorrupted files:")
        for file_path, reason in corrupted_files:
            logger.info(f"  - {file_path.name}: {reason}")
        
        # Save list of corrupted files to log
        log_file = logs_path / "corrupted_files_removed.txt"
        with open(log_file, 'w') as f:
            f.write(f"Corrupted files removed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total corrupted files: {len(corrupted_files)}\n")
            f.write(f"Total valid files: {len(valid_files)}\n")
            f.write(f"Total files checked: {len(genome_files)}\n\n")
            f.write("Removed files:\n")
            for file_path, reason in corrupted_files:
                f.write(f"{file_path.name} - {reason}\n")
        
        logger.info(f"Corrupted files list saved to: {log_file}")
        
        if not dry_run:
            logger.info("\nRemoving corrupted files...")
            removed_count = 0
            for file_path, reason in corrupted_files:
                try:
                    file_path.unlink()
                    logger.info(f"Removed: {file_path.name}")
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove {file_path.name}: {e}")
            
            logger.info(f"Successfully removed {removed_count} corrupted files")
        else:
            logger.info("\nRun with dry_run=False to actually remove the files")
    
    return len(valid_files), len(corrupted_files)

def main():
    genomes_dir = "/zdata/user-data/noam/data/p2cs/full_genomes"
    logs_dir = "/zdata/user-data/noam/code/Phage-Mate/2 p2cs distance estimation/logs"
    
    logger.info("Checking for corrupted FASTA files...")
    valid_count, corrupted_count = clean_corrupted_files(genomes_dir, logs_dir, dry_run=True)
    
    if corrupted_count > 0:
        logger.info(f"\nFound {corrupted_count} corrupted files out of {valid_count + corrupted_count} total")
        logger.info("To remove corrupted files, run:")
        logger.info("python clean_corrupted_files.py --remove")
    else:
        logger.info("All FASTA files are valid!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up corrupted FASTA files')
    parser.add_argument('--remove', action='store_true', 
                       help='Actually remove corrupted files (default: dry run)')
    
    args = parser.parse_args()
    
    genomes_dir = "/zdata/user-data/noam/data/p2cs/full_genomes"
    logs_dir = "/zdata/user-data/noam/code/Phage-Mate/2 p2cs distance estimation/logs"
    clean_corrupted_files(genomes_dir, logs_dir, dry_run=not args.remove)
