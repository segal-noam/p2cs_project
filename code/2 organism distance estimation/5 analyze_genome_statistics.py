#!/usr/bin/env python3
"""
Script to analyze genome FASTA files and collect comprehensive statistics.
Analyzes individual genome files and merged organism files to provide:
- Length of each genome file
- Number of files belonging to each merged organism
- Length of each merged file
- Comprehensive statistics and summary

This version includes caching to avoid re-analyzing files that haven't changed.
"""

import os
import sys
import pandas as pd
from pathlib import Path
import logging
from collections import defaultdict, Counter
from Bio import SeqIO
import json
import pickle
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_directory_hash(directory_path):
    """Generate a hash of directory contents to detect changes."""
    directory = Path(directory_path)
    if not directory.exists():
        return None

    # Get all FASTA files and their modification times
    fasta_files = list(directory.glob("*.fasta"))
    file_info = []

    for file_path in sorted(fasta_files):
        stat = file_path.stat()
        file_info.append(f"{file_path.name}:{stat.st_mtime}:{stat.st_size}")

    # Create hash from file info
    content = "\n".join(file_info)
    return hashlib.md5(content.encode()).hexdigest()


def save_cache(data, cache_path):
    """Save data to cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Cached data saved to: {cache_path}")


def load_cache(cache_path):
    """Load data from cache file if it exists."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded cached data from: {cache_path}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None


def is_cache_valid(cache_path, directory_path):
    """Check if cache is still valid by comparing directory hashes."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return False

    # Load cached hash
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)

        if 'directory_hash' not in cached_data:
            return False

        # Compare with current directory hash
        current_hash = get_directory_hash(directory_path)
        return cached_data['directory_hash'] == current_hash
    except:
        return False


def get_fasta_length(filepath):
    """Calculate the total length of all sequences in a FASTA file."""
    try:
        total_length = 0
        sequence_count = 0

        with open(filepath, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                total_length += len(record.seq)
                sequence_count += 1

        return total_length, sequence_count
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return 0, 0


def analyze_individual_genomes(genomes_dir, cache_dir=None):
    """Analyze all individual genome files in the full_genomes directory."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "individual_genomes_cache.pkl"

        # Check if cache is valid
        if is_cache_valid(cache_path, genomes_dir):
            logger.info("Loading individual genome analysis from cache...")
            cached_data = load_cache(cache_path)
            if cached_data:
                return cached_data['individual_stats'], cached_data['organism_file_counts']

    logger.info("Analyzing individual genome files...")

    genomes_path = Path(genomes_dir)
    if not genomes_path.exists():
        logger.error(f"Genomes directory not found: {genomes_path}")
        return {}, {}

    # Get all FASTA files
    fasta_files = list(genomes_path.glob("*.fasta"))
    logger.info(f"Found {len(fasta_files)} individual genome files")

    individual_stats = {}
    organism_file_counts = defaultdict(int)

    for i, fasta_file in enumerate(fasta_files, 1):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(fasta_files)} files...")

        # Extract organism name from filename
        filename = fasta_file.stem
        parts = filename.split('_')

        # Try to determine organism name
        organism_name = None
        if len(parts) >= 2:
            # Look for patterns like "Genus_species_strain_accession"
            for j in range(1, len(parts)):
                potential_organism = '_'.join(parts[:-j])
                if len(potential_organism) > 3 and not potential_organism.endswith(('CP', 'NC', 'NZ', 'AP')):
                    organism_name = potential_organism
                    break

        if not organism_name:
            organism_name = '_'.join(
                parts[:-1]) if len(parts) > 1 else filename

        # Get file statistics
        file_size = fasta_file.stat().st_size
        total_length, sequence_count = get_fasta_length(fasta_file)

        individual_stats[fasta_file.name] = {
            'filepath': str(fasta_file),
            'organism': organism_name,
            'file_size_bytes': file_size,
            'total_sequence_length': total_length,
            'sequence_count': sequence_count,
            'average_sequence_length': total_length / sequence_count if sequence_count > 0 else 0
        }

        organism_file_counts[organism_name] += 1

    logger.info(f"Analyzed {len(individual_stats)} individual genome files")
    logger.info(f"Found {len(organism_file_counts)} unique organisms")

    # Save to cache if cache directory is provided
    if cache_path:
        cache_data = {
            'individual_stats': individual_stats,
            'organism_file_counts': dict(organism_file_counts),
            'directory_hash': get_directory_hash(genomes_dir)
        }
        save_cache(cache_data, cache_path)

    return individual_stats, organism_file_counts


def analyze_merged_genomes(merged_dir, cache_dir=None):
    """Analyze all merged organism files."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / "merged_genomes_cache.pkl"

        # Check if cache is valid
        if is_cache_valid(cache_path, merged_dir):
            logger.info("Loading merged genome analysis from cache...")
            cached_data = load_cache(cache_path)
            if cached_data:
                return cached_data['merged_stats']

    logger.info("Analyzing merged organism files...")

    merged_path = Path(merged_dir)
    if not merged_path.exists():
        logger.error(f"Merged genomes directory not found: {merged_path}")
        return {}

    # Get all merged FASTA files
    merged_files = list(merged_path.glob("*_merged.fasta"))
    logger.info(f"Found {len(merged_files)} merged organism files")

    merged_stats = {}

    for i, merged_file in enumerate(merged_files, 1):
        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(merged_files)} merged files...")

        # Extract organism name from filename
        organism_name = merged_file.stem.replace('_merged', '')

        # Get file statistics
        file_size = merged_file.stat().st_size
        total_length, sequence_count = get_fasta_length(merged_file)

        merged_stats[merged_file.name] = {
            'filepath': str(merged_file),
            'organism': organism_name,
            'file_size_bytes': file_size,
            'total_sequence_length': total_length,
            'sequence_count': sequence_count,
            'average_sequence_length': total_length / sequence_count if sequence_count > 0 else 0
        }

    logger.info(f"Analyzed {len(merged_stats)} merged organism files")

    # Save to cache if cache directory is provided
    if cache_path:
        cache_data = {
            'merged_stats': merged_stats,
            'directory_hash': get_directory_hash(merged_dir)
        }
        save_cache(cache_data, cache_path)

    return merged_stats


def create_comprehensive_summary(individual_stats, organism_file_counts, merged_stats):
    """Create a comprehensive summary of all genome statistics."""

    # Convert to DataFrames for easier analysis
    individual_df = pd.DataFrame.from_dict(individual_stats, orient='index')
    merged_df = pd.DataFrame.from_dict(merged_stats, orient='index')

    # Create organism summary
    organism_summary = []

    for organism, file_count in organism_file_counts.items():
        # Find corresponding merged file
        merged_file = None
        for merged_name, merged_data in merged_stats.items():
            if merged_data['organism'] == organism:
                merged_file = merged_data
                break

        # Get individual files for this organism
        organism_individual_files = individual_df[individual_df['organism'] == organism]

        organism_summary.append({
            'organism': organism,
            'individual_file_count': file_count,
            'individual_total_length': organism_individual_files['total_sequence_length'].sum(),
            'individual_sequence_count': organism_individual_files['sequence_count'].sum(),
            'merged_file_exists': merged_file is not None,
            'merged_total_length': merged_file['total_sequence_length'] if merged_file else 0,
            'merged_sequence_count': merged_file['sequence_count'] if merged_file else 0,
            'merged_file_size': merged_file['file_size_bytes'] if merged_file else 0
        })

    organism_summary_df = pd.DataFrame(organism_summary)

    return individual_df, merged_df, organism_summary_df


def save_results(individual_df, merged_df, organism_summary_df, output_dir):
    """Save all results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed dataframes
    individual_df.to_csv(
        output_path / "individual_genome_statistics.csv", index=True)
    merged_df.to_csv(output_path / "merged_genome_statistics.csv", index=True)
    organism_summary_df.to_csv(
        output_path / "organism_summary_statistics.csv", index=False)

    # Create summary statistics
    summary_stats = {
        'total_individual_files': len(individual_df),
        'total_merged_files': len(merged_df),
        'total_organisms': len(organism_summary_df),
        'total_individual_sequence_length': individual_df['total_sequence_length'].sum(),
        'total_merged_sequence_length': merged_df['total_sequence_length'].sum(),
        'average_individual_file_length': individual_df['total_sequence_length'].mean(),
        'average_merged_file_length': merged_df['total_sequence_length'].mean(),
        'organisms_with_merged_files': organism_summary_df['merged_file_exists'].sum(),
        'organisms_without_merged_files': (~organism_summary_df['merged_file_exists']).sum()
    }

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj

    summary_stats = convert_numpy_types(summary_stats)

    # Save summary statistics
    with open(output_path / "summary_statistics.json", 'w') as f:
        json.dump(summary_stats, f, indent=2)

    # Create a detailed report
    report_lines = [
        "GENOME ANALYSIS REPORT",
        "=" * 50,
        f"Total individual genome files: {summary_stats['total_individual_files']:,}",
        f"Total merged organism files: {summary_stats['total_merged_files']:,}",
        f"Total unique organisms: {summary_stats['total_organisms']:,}",
        "",
        "SEQUENCE LENGTH STATISTICS:",
        f"Total individual sequence length: {summary_stats['total_individual_sequence_length']:,} bp",
        f"Total merged sequence length: {summary_stats['total_merged_sequence_length']:,} bp",
        f"Average individual file length: {summary_stats['average_individual_file_length']:,.0f} bp",
        f"Average merged file length: {summary_stats['average_merged_file_length']:,.0f} bp",
        "",
        "ORGANISM COVERAGE:",
        f"Organisms with merged files: {summary_stats['organisms_with_merged_files']:,}",
        f"Organisms without merged files: {summary_stats['organisms_without_merged_files']:,}",
        "",
        "TOP 10 ORGANISMS BY FILE COUNT:",
    ]

    # Add top organisms by file count
    top_organisms = organism_summary_df.nlargest(10, 'individual_file_count')
    for _, row in top_organisms.iterrows():
        report_lines.append(
            f"  {row['organism']}: {row['individual_file_count']} files, {row['individual_total_length']:,} bp")

    report_lines.extend([
        "",
        "TOP 10 LARGEST MERGED FILES:",
    ])

    # Add largest merged files
    largest_merged = merged_df.nlargest(10, 'total_sequence_length')
    for _, row in largest_merged.iterrows():
        report_lines.append(
            f"  {row['organism']}: {row['total_sequence_length']:,} bp, {row['sequence_count']} sequences")

    # Save report
    with open(output_path / "genome_analysis_report.txt", 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Summary statistics: {summary_stats}")


def main():
    # Configuration
    genomes_dir = "/zdata/user-data/noam/data/p2cs/full_genomes"
    merged_dir = "/zdata/user-data/noam/data/p2cs/organism_merged_genomes"
    output_dir = "/zdata/user-data/noam/data/p2cs/genome_analysis_results"
    cache_dir = "/zdata/user-data/noam/data/p2cs/genome_analysis_cache"

    logger.info("Starting comprehensive genome analysis with caching...")

    # Analyze individual genomes
    individual_stats, organism_file_counts = analyze_individual_genomes(
        genomes_dir, cache_dir)

    if not individual_stats:
        logger.error("No individual genome files found")
        sys.exit(1)

    # Analyze merged genomes
    merged_stats = analyze_merged_genomes(merged_dir, cache_dir)

    if not merged_stats:
        logger.warning("No merged genome files found")

    # Create comprehensive summary
    individual_df, merged_df, organism_summary_df = create_comprehensive_summary(
        individual_stats, organism_file_counts, merged_stats
    )

    # Save all results
    save_results(individual_df, merged_df, organism_summary_df, output_dir)

    logger.info("Genome analysis completed successfully!")


if __name__ == "__main__":
    main()
