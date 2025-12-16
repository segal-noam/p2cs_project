#!/usr/bin/env python3
"""
Script to download full genomes from NCBI for organisms in the p2cs dataset.
Uses NCBI Datasets tool directly to search for and download genome assemblies.
Maintains mapping between chromosome files, assemblies, and organisms.
"""

import os
import sys
import time
import pickle
import pandas as pd
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def search_organism_assemblies(organism_name, max_results=10):
    """
    Search NCBI for genome assemblies of a given organism using NCBI Datasets.
    Returns a list of assembly information dictionaries.
    """
    try:
        logger.info(f"Searching for assemblies: {organism_name}")

        # Use NCBI Datasets to search for assemblies
        cmd = ["datasets", "summary", "genome", "taxon",
               organism_name, "--limit", str(max_results)]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.warning(
                f"NCBI Datasets search failed for {organism_name}: {result.stderr}")
            return []

        # Parse the JSON response
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse NCBI Datasets response for {organism_name}: {e}")
            return []

        if 'reports' not in data or not data['reports']:
            logger.warning(f"No assemblies found for {organism_name}")
            return []

        # Extract assembly information
        assembly_info = []
        for report in data['reports']:
            try:
                accession = report.get('accession', '')
                organism = report.get('organism', {}).get(
                    'organism_name', organism_name)

                # Get assembly details
                assembly_info_dict = report.get('assembly_info', {})
                assembly_stats = report.get('assembly_stats', {})

                assembly_info.append({
                    'assembly_id': accession,
                    'accession': accession,
                    'name': assembly_info_dict.get('assembly_name', ''),
                    'organism': organism,
                    'original_query': organism_name,
                    'taxid': report.get('organism', {}).get('tax_id', ''),
                    'refseq_category': report.get('source_database', ''),
                    'assembly_level': assembly_info_dict.get('assembly_level', ''),
                    'contig_n50': assembly_stats.get('contig_n50', ''),
                    'scaffold_n50': assembly_stats.get('scaffold_n50', ''),
                    'genome_rep': assembly_info_dict.get('assembly_type', ''),
                    'submitter': assembly_info_dict.get('submitter', ''),
                    'release_date': assembly_info_dict.get('release_date', '')
                })

            except Exception as e:
                logger.warning(
                    f"Error processing assembly report for {organism_name}: {e}")
                continue

        if assembly_info:
            logger.info(
                f"Found {len(assembly_info)} assemblies for {organism_name}")
            # Log the assemblies found
            for i, assembly in enumerate(assembly_info[:3]):  # Show top 3
                logger.info(
                    f"  {i+1}. {assembly['organism']} - {assembly['accession']}")
            return assembly_info
        else:
            logger.warning(f"No assemblies found for {organism_name}")
            return []

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout searching for assemblies of {organism_name}")
        return []
    except Exception as e:
        logger.error(f"Error searching for assemblies of {organism_name}: {e}")
        return []


def download_assembly_chromosomes(assembly_accession, output_dir, organism_name):
    """
    Download all chromosomes for a given assembly using NCBI Datasets.
    Returns a list of downloaded chromosome files and metadata.
    """
    try:
        # Create organism-specific directory
        safe_name = organism_name.replace(
            " ", "_").replace("/", "_").replace("\\", "_")
        organism_dir = output_dir / safe_name
        organism_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_file = temp_path / f"{assembly_accession}_dataset.zip"

            # Use NCBI Datasets command-line tool
            cmd = [
                "datasets", "download", "genome", "accession", assembly_accession,
                "--include", "genome",
                "--filename", str(zip_file)
            ]

            logger.info(
                f"Downloading assembly {assembly_accession} for {organism_name}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(
                    f"Failed to download assembly {assembly_accession}: {result.stderr}")
                return [], {}

            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_path)

            # Find the genomic FASTA file
            genomic_files = list(temp_path.glob("**/*_genomic.fna"))
            if not genomic_files:
                logger.warning(
                    f"No genomic FASTA file found for {assembly_accession}")
                return [], {}

            genomic_file = genomic_files[0]

            # Parse the FASTA file to separate chromosomes
            chromosome_files = []
            chromosome_metadata = {}
            current_chromosome = None
            current_file = None

            with open(genomic_file, 'r') as f:
                for line in f:
                    if line.startswith('>'):
                        # Save previous chromosome if exists
                        if current_file:
                            current_file.close()
                            chromosome_files.append(current_file_path)

                        # Start new chromosome
                        header = line.strip()[1:]  # Remove '>'
                        # First part is usually chromosome name
                        chromosome_name = header.split()[0]

                        # Create safe filename
                        safe_chromosome = chromosome_name.replace(
                            ":", "_").replace("|", "_")
                        filename = f"{safe_name}_{assembly_accession}_{safe_chromosome}.fasta"
                        current_file_path = organism_dir / filename
                        current_file = open(current_file_path, 'w')

                        # Store metadata
                        chromosome_metadata[filename] = {
                            'organism': organism_name,
                            'assembly_accession': assembly_accession,
                            'chromosome_name': chromosome_name,
                            'header': header,
                            'file_path': str(current_file_path)
                        }

                        current_chromosome = chromosome_name

                    if current_file:
                        current_file.write(line)

            # Close the last file
            if current_file:
                current_file.close()
                chromosome_files.append(current_file_path)

            logger.info(
                f"Downloaded {len(chromosome_files)} chromosomes for {organism_name} (assembly {assembly_accession})")
            return chromosome_files, chromosome_metadata

    except subprocess.TimeoutExpired:
        logger.error(
            f"Timeout downloading assembly {assembly_accession} for {organism_name}")
        return [], {}
    except Exception as e:
        logger.error(
            f"Error downloading assembly {assembly_accession} for {organism_name}: {e}")
        return [], {}


def is_valid_fasta(filepath):
    """Check if a downloaded FASTA file is valid and contains sequences."""
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


def download_organism_assemblies(organism_name, output_dir, max_assemblies=3):
    """
    Download genome assemblies for a given organism.
    Returns a list of downloaded chromosome files and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for assemblies
    assemblies = search_organism_assemblies(
        organism_name, max_results=max_assemblies)

    if not assemblies:
        return [], {}

    all_chromosome_files = []
    all_metadata = {}

    # Download chromosomes for each assembly
    for i, assembly in enumerate(assemblies):
        assembly_accession = assembly['accession']

        # Check if we already have this assembly
        safe_name = organism_name.replace(
            " ", "_").replace("/", "_").replace("\\", "_")
        organism_dir = output_dir / safe_name
        existing_files = list(organism_dir.glob(
            f"*{assembly_accession}*.fasta"))

        if existing_files:
            logger.info(
                f"Assembly {assembly_accession} already exists for {organism_name}")
            # Load existing metadata
            metadata_file = organism_dir / \
                f"{safe_name}_{assembly_accession}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
                    all_metadata.update(existing_metadata)
                    all_chromosome_files.extend(existing_files)
            continue

        # Rate limiting between assemblies
        if i > 0:
            time.sleep(2)

        chromosome_files, chromosome_metadata = download_assembly_chromosomes(
            assembly_accession, output_dir, organism_name)

        if chromosome_files:
            all_chromosome_files.extend(chromosome_files)
            all_metadata.update(chromosome_metadata)

            # Save metadata for this assembly
            safe_name = organism_name.replace(
                " ", "_").replace("/", "_").replace("\\", "_")
            organism_dir = output_dir / safe_name
            metadata_file = organism_dir / \
                f"{safe_name}_{assembly_accession}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(chromosome_metadata, f, indent=2)

    return all_chromosome_files, all_metadata


def load_organisms_from_pkl(pkl_path):
    """Load unique organisms from the pickle file."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, pd.DataFrame) and 'organism' in data.columns:
            return data['organism'].unique()
        else:
            logger.error("Invalid data format in pickle file")
            return []
    except Exception as e:
        logger.error(f"Error loading pickle file: {e}")
        return []


def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def cleanup_corrupted_files(output_dir):
    """Clean up any corrupted FASTA files in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return 0

    genome_files = list(output_path.glob("**/*.fasta"))
    corrupted_count = 0

    logger.info(
        f"Checking {len(genome_files)} existing files for corruption...")

    for genome_file in genome_files:
        is_valid, reason = is_valid_fasta(genome_file)
        if not is_valid:
            logger.warning(
                f"Found corrupted file: {genome_file.name} - {reason}")
            try:
                genome_file.unlink()
                logger.info(f"Deleted corrupted file: {genome_file.name}")
                corrupted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {genome_file.name}: {e}")

    if corrupted_count > 0:
        logger.info(f"Cleaned up {corrupted_count} corrupted files")
    else:
        logger.info("No corrupted files found")

    return corrupted_count


def save_master_mapping(all_metadata, output_dir):
    """Save a master mapping file of all chromosome files, assemblies, and organisms."""
    output_path = Path(output_dir)
    master_mapping = []

    for filename, metadata in all_metadata.items():
        master_mapping.append({
            'filename': filename,
            'organism': metadata['organism'],
            'assembly_accession': metadata['assembly_accession'],
            'chromosome_name': metadata['chromosome_name'],
            'file_path': metadata['file_path'],
            'header': metadata['header']
        })

    # Save as CSV
    df = pd.DataFrame(master_mapping)
    csv_path = output_path / "chromosome_assembly_mapping.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Master mapping saved to: {csv_path}")

    # Save as JSON
    json_path = output_path / "chromosome_assembly_mapping.json"
    with open(json_path, 'w') as f:
        json.dump(master_mapping, f, indent=2)
    logger.info(f"Master mapping saved to: {json_path}")

    return csv_path, json_path


def main():
    # Load configuration
    config = load_config()

    # Configuration
    pkl_path = config['data_paths']['pkl_path']
    output_dir = config['data_paths']['individual_genomes_dir']
    max_organisms = config['ncbi_settings']['max_organisms']
    max_assemblies = config.get('ncbi_settings', {}).get(
        'max_assemblies_per_organism', 3)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Clean up any existing corrupted files
    logger.info("Checking for existing corrupted files...")
    cleanup_corrupted_files(output_dir)

    # Load organisms
    logger.info("Loading organisms from pickle file...")
    organisms = load_organisms_from_pkl(pkl_path)

    if organisms is None or len(organisms) == 0:
        logger.error("No organisms found in pickle file")
        sys.exit(1)

    logger.info(f"Found {len(organisms)} unique organisms")

    # Limit organisms for testing
    if max_organisms:
        organisms = organisms[:max_organisms]
        logger.info(f"Processing first {len(organisms)} organisms for testing")

    # Download assemblies and chromosomes
    successful_downloads = 0
    failed_organisms = []
    all_metadata = {}

    for i, organism in enumerate(organisms):
        logger.info(f"Processing organism {i+1}/{len(organisms)}: {organism}")

        try:
            chromosome_files, metadata = download_organism_assemblies(
                organism, output_dir, max_assemblies)

            if chromosome_files:
                successful_downloads += len(chromosome_files)
                all_metadata.update(metadata)
                logger.info(
                    f"Successfully downloaded {len(chromosome_files)} chromosomes for {organism}")
            else:
                failed_organisms.append(organism)
                logger.warning(f"No chromosomes downloaded for {organism}")

            # Rate limiting between organisms
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing {organism}: {e}")
            failed_organisms.append(organism)

    # Save master mapping
    logger.info("Saving master mapping...")
    save_master_mapping(all_metadata, output_dir)

    # Summary
    logger.info(f"\nDownload Summary:")
    logger.info(f"Total organisms processed: {len(organisms)}")
    logger.info(f"Total chromosomes downloaded: {successful_downloads}")
    logger.info(f"Failed organisms: {len(failed_organisms)}")

    if failed_organisms:
        logger.info("Failed organisms:")
        for org in failed_organisms:
            logger.info(f"  - {org}")

    # Final cleanup of any corrupted files
    logger.info("\nPerforming final cleanup of corrupted files...")
    final_cleanup_count = cleanup_corrupted_files(output_dir)

    # Save failed organisms list
    failed_path = Path(output_dir) / "failed_organisms.txt"
    with open(failed_path, 'w') as f:
        for org in failed_organisms:
            f.write(f"{org}\n")
    logger.info(f"Failed organisms list saved to: {failed_path}")

    if final_cleanup_count > 0:
        logger.info(
            f"Final cleanup removed {final_cleanup_count} additional corrupted files")


if __name__ == "__main__":
    main()
