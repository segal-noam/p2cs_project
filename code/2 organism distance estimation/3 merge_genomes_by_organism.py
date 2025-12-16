#!/usr/bin/env python3
"""
Script to merge FASTA files by organism from existing downloaded genomes.
Uses updated data structure with chromosome_assembly_mapping.csv and applies filtering conditions:
- avg_file_length_nt > 1e6
- file_count < 8
For each organism, selects at most 1 assembly (or none) and merges all genome files into single FASTA.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict
from Bio import SeqIO

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_valid_fasta(filepath):
    """Check if a FASTA file is valid and contains sequences."""
    try:
        # Check file size (must be > 100 bytes)
        if filepath.stat().st_size < 100:
            return False, "File too small"

        # Check if file contains FASTA sequences
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Must have at least one header line and one sequence line
        has_header = any(line.startswith('>') for line in lines)
        has_sequence = any(not line.startswith(
            '>') and line.strip() for line in lines)

        if not has_header:
            return False, "No FASTA headers found"
        if not has_sequence:
            return False, "No sequence data found"

        return True, "Valid"

    except Exception as e:
        return False, f"Error reading file: {e}"


def load_data(mapping_csv_path, pkl_path):
    """Load the master mapping CSV and p2cs_filtered_groups.pkl files."""
    try:
        # Load chromosome assembly mapping
        logger.info("Loading chromosome assembly mapping...")
        mapping_df = pd.read_csv(mapping_csv_path)
        logger.info(f"Loaded {len(mapping_df)} chromosome records")

        # Load p2cs filtered groups
        logger.info("Loading p2cs filtered groups...")
        with open(pkl_path, 'rb') as f:
            p2cs_data = pickle.load(f)
        logger.info(f"Loaded p2cs data with {len(p2cs_data)} records")

        return mapping_df, p2cs_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None


def calculate_assembly_statistics(mapping_df):
    """
    Calculate assembly statistics (avg file length, file count) for each organism-assembly combination.
    Uses existing nucleotide_length column from the mapping file.
    """
    logger.info("Calculating assembly statistics...")

    # Check if nucleotide_length column exists
    if 'nucleotide_length' not in mapping_df.columns:
        logger.error("nucleotide_length column not found in mapping file")
        return None, None

    # Calculate average file length per assembly
    assembly_stats = mapping_df.groupby(['organism', 'assembly_accession']).agg({
        'nucleotide_length': ['mean', 'count']
    }).reset_index()

    assembly_stats.columns = [
        'organism', 'assembly_accession', 'avg_file_length_nt', 'file_count']

    logger.info(
        f"Calculated statistics for {len(assembly_stats)} organism-assembly combinations")
    return assembly_stats, mapping_df


def filter_and_select_assemblies(assembly_stats, p2cs_data):
    """
    Apply filtering conditions and select at most 1 assembly per organism.
    Filtering conditions: avg_file_length_nt > 1e6 and file_count < 8
    """
    logger.info("Applying filtering conditions...")

    # Apply filtering conditions
    filtered_assemblies = assembly_stats[
        (assembly_stats['avg_file_length_nt'] > 0) &
        (assembly_stats['file_count'] < np.inf)
    ]

    logger.info(
        f"Found {len(filtered_assemblies)} assemblies meeting filtering criteria")

    # For each organism, select the assembly with highest avg_file_length_nt
    # (or none if no assembly meets criteria)
    selected_assemblies = []

    for organism in p2cs_data['organism'].unique():
        organism_assemblies = filtered_assemblies[filtered_assemblies['organism'] == organism]

        if len(organism_assemblies) > 0:
            # Select the assembly with highest avg_file_length_nt
            best_assembly = organism_assemblies.loc[organism_assemblies['avg_file_length_nt'].idxmax(
            )]
            selected_assemblies.append(best_assembly)
        else:
            # No assembly meets criteria for this organism
            selected_assemblies.append(pd.Series({
                'organism': organism,
                'assembly_accession': np.nan,
                'avg_file_length_nt': np.nan,
                'file_count': np.nan
            }))

    selected_df = pd.DataFrame(selected_assemblies)
    logger.info(
        f"Selected assemblies for {len(selected_df[selected_df['assembly_accession'].notna()])} organisms")
    logger.info(
        f"No suitable assembly found for {len(selected_df[selected_df['assembly_accession'].isna()])} organisms")

    return selected_df


def check_existing_merged_files(output_dir):
    """
    Check which organisms already have merged files and return a set of organism names to skip.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return set()

    existing_files = set()
    for fasta_file in output_path.glob("*_merged.fasta"):
        # Extract organism name from filename (remove _merged.fasta suffix)
        organism_name = fasta_file.stem.replace("_merged", "")
        # Convert back to original format (replace underscores with spaces)
        organism_name = organism_name.replace("_", " ")
        existing_files.add(organism_name)

    logger.info(f"Found {len(existing_files)} existing merged files")
    return existing_files


def merge_organism_genomes(organism_name, assembly_accession, mapping_df, output_dir):
    """
    Merge all genome files for a selected organism-assembly combination into a single FASTA file.
    """
    if pd.isna(assembly_accession):
        logger.info(f"No assembly selected for {organism_name}")
        return None

    # Get all files for this organism-assembly combination
    organism_files = mapping_df[
        (mapping_df['organism'] == organism_name) &
        (mapping_df['assembly_accession'] == assembly_accession)
    ]

    if len(organism_files) == 0:
        logger.warning(
            f"No files found for {organism_name} assembly {assembly_accession}")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename
    safe_name = organism_name.replace(
        " ", "_").replace("/", "_").replace("\\", "_")
    merged_filename = f"{safe_name}_merged.fasta"
    merged_filepath = output_dir / merged_filename

    try:
        with open(merged_filepath, 'w') as outfile:
            # Write single header for the organism
            outfile.write(f">{safe_name}\n")

            # Concatenate all sequences without headers
            for _, row in organism_files.iterrows():
                file_path = Path(row['file_path'])
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue

                # Validate the genome file first
                is_valid, reason = is_valid_fasta(file_path)
                if not is_valid:
                    logger.warning(
                        f"Skipping invalid file {file_path.name}: {reason}")
                    continue

                with open(file_path, 'r') as infile:
                    # Read sequences and concatenate without headers
                    for record in SeqIO.parse(infile, "fasta"):
                        # Write sequence without header
                        outfile.write(str(record.seq))

        # Validate the merged file
        is_valid, reason = is_valid_fasta(merged_filepath)
        if not is_valid:
            logger.warning(
                f"Merged file is corrupted: {merged_filename} - {reason}")
            merged_filepath.unlink()
            return None

        logger.info(
            f"Merged {len(organism_files)} genome files for {organism_name} into {merged_filepath}")
        return merged_filepath

    except Exception as e:
        logger.error(f"Error merging genome files for {organism_name}: {e}")
        return None


def main():
    # Configuration
    mapping_csv_path = "/zdata/user-data/noam/data/p2cs/full_genomes_new/chromosome_assembly_mapping_with_lengths.csv"
    pkl_path = "/zdata/user-data/noam/data/p2cs/p2cs_filtered_groups.pkl"
    output_dir = "/zdata/user-data/noam/data/p2cs/organism_merged_genomes_new"

    logger.info("Starting genome merging process with updated data structure...")

    # Load data
    mapping_df, p2cs_data = load_data(mapping_csv_path, pkl_path)
    if mapping_df is None or p2cs_data is None:
        logger.error("Failed to load data")
        sys.exit(1)

    # Calculate assembly statistics
    assembly_stats, mapping_df = calculate_assembly_statistics(mapping_df)

    # Filter and select assemblies
    selected_assemblies = filter_and_select_assemblies(
        assembly_stats, p2cs_data)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check for existing merged files to skip
    existing_files = check_existing_merged_files(output_dir)

    # Merge genomes for each organism
    successful_merges = 0
    failed_organisms = []
    assembly_tracking = []
    skipped_organisms = 0

    for _, row in selected_assemblies.iterrows():
        organism_name = row['organism']
        assembly_accession = row['assembly_accession']

        # Skip if merged file already exists
        if organism_name in existing_files:
            logger.info(
                f"Skipping {organism_name} - merged file already exists")
            skipped_organisms += 1
            assembly_tracking.append({
                'organism': organism_name,
                'assembly_accession': assembly_accession,
                'avg_file_length_nt': row['avg_file_length_nt'],
                'file_count': row['file_count'],
                'merged_file': 'SKIPPED - already exists'
            })
            continue

        logger.info(
            f"Processing {organism_name} with assembly {assembly_accession}...")

        try:
            merged_file = merge_organism_genomes(
                organism_name, assembly_accession, mapping_df, output_dir)

            if merged_file:
                successful_merges += 1
                logger.info(f"Successfully merged genomes for {organism_name}")
                assembly_tracking.append({
                    'organism': organism_name,
                    'assembly_accession': assembly_accession,
                    'avg_file_length_nt': row['avg_file_length_nt'],
                    'file_count': row['file_count'],
                    'merged_file': str(merged_file)
                })
            else:
                failed_organisms.append(organism_name)
                logger.warning(f"Failed to merge genomes for {organism_name}")
                assembly_tracking.append({
                    'organism': organism_name,
                    'assembly_accession': assembly_accession,
                    'avg_file_length_nt': row['avg_file_length_nt'],
                    'file_count': row['file_count'],
                    'merged_file': None
                })

        except Exception as e:
            logger.error(f"Error processing {organism_name}: {e}")
            failed_organisms.append(organism_name)
            assembly_tracking.append({
                'organism': organism_name,
                'assembly_accession': assembly_accession,
                'avg_file_length_nt': row['avg_file_length_nt'],
                'file_count': row['file_count'],
                'merged_file': None
            })

    # Save assembly tracking
    tracking_df = pd.DataFrame(assembly_tracking)
    tracking_path = Path(output_dir) / "assembly_tracking.csv"
    tracking_df.to_csv(tracking_path, index=False)
    logger.info(f"Assembly tracking saved to: {tracking_path}")

    # Summary
    logger.info(f"\nMerge Summary:")
    logger.info(f"Total organisms processed: {len(selected_assemblies)}")
    logger.info(f"Successfully merged organisms: {successful_merges}")
    logger.info(f"Skipped organisms (already exist): {skipped_organisms}")
    logger.info(f"Failed organisms: {len(failed_organisms)}")
    logger.info(
        f"Organisms with no suitable assembly: {len(selected_assemblies[selected_assemblies['assembly_accession'].isna()])}")

    if failed_organisms:
        logger.info("Failed organisms:")
        for org in failed_organisms:
            logger.info(f"  - {org}")

    # Save failed organisms list
    failed_path = Path(output_dir) / "failed_organisms.txt"
    with open(failed_path, 'w') as f:
        for org in failed_organisms:
            f.write(f"{org}\n")
    logger.info(f"Failed organisms list saved to: {failed_path}")

    logger.info(f"\nMerged organism files saved to: {output_dir}")


if __name__ == "__main__":
    main()
