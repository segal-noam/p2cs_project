#!/usr/bin/env python3
"""
Script to sketch genomes with Mash and compute pairwise distance matrices.
Uses multiple k values to capture different levels of sequence similarity.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import json
from pathlib import Path
import logging
from itertools import combinations
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MashAnalyzer:
    def __init__(self, genomes_dir, output_dir, k_values=[11, 15, 19, 21],
                 sketch_size=100000, parallelism=10, seed=42,
                 warning_threshold=0.01, preserve_strand=False, sketch_individual_sequences=False):
        """
        Initialize Mash analyzer for organism-level analysis.

        Args:
            genomes_dir: Directory containing merged organism FASTA files
            output_dir: Directory to save Mash sketches and distance matrices
            k_values: List of k-mer sizes to use for sketching
            sketch_size: Number of k-mers to sketch per organism
            parallelism: Number of threads for parallel processing
            seed: Seed for hash function
            warning_threshold: Probability threshold for k-mer size warnings
            preserve_strand: Whether to preserve strand information
            sketch_individual_sequences: Whether to sketch individual sequences vs whole files
        """
        self.genomes_dir = Path(genomes_dir)
        self.output_dir = Path(output_dir)
        self.k_values = k_values
        self.sketch_size = sketch_size
        self.parallelism = parallelism
        self.seed = seed
        self.warning_threshold = warning_threshold
        self.preserve_strand = preserve_strand
        self.sketch_individual_sequences = sketch_individual_sequences

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sketches_dir = self.output_dir / "sketches"
        self.matrices_dir = self.output_dir / "distance_matrices"
        self.sketches_dir.mkdir(exist_ok=True)
        self.matrices_dir.mkdir(exist_ok=True)

        # Check if Mash is available
        self._check_mash_installation()

    def _check_mash_installation(self):
        """Check if Mash is installed and accessible."""
        try:
            result = subprocess.run(['mash', '--version'],
                                    capture_output=True, text=True, check=True)
            logger.info(f"Mash version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Mash is not installed or not in PATH")
            sys.exit(1)

    def get_organism_files(self):
        """Get list of merged organism FASTA files."""
        organism_files = list(self.genomes_dir.glob("*_merged.fasta"))
        if not organism_files:
            logger.error(
                f"No merged organism FASTA files found in {self.genomes_dir}")
            return []

        # Filter out empty or corrupted files
        valid_files = []
        for organism_file in organism_files:
            if self.is_valid_fasta(organism_file):
                valid_files.append(organism_file)
            else:
                logger.warning(
                    f"Skipping invalid/corrupted file: {organism_file.name}")

        logger.info(
            f"Found {len(organism_files)} total organism files, {len(valid_files)} valid files")
        return valid_files

    def is_valid_fasta(self, fasta_file):
        """Check if a FASTA file is valid and contains sequences."""
        try:
            # Check file size (must be > 100 bytes)
            if fasta_file.stat().st_size < 100:
                return False

            # Check if file contains FASTA sequences
            with open(fasta_file, 'r') as f:
                lines = f.readlines()

            # Must have at least one header line and one sequence line
            has_header = any(line.startswith('>') for line in lines)
            has_sequence = any(not line.startswith(
                '>') and line.strip() for line in lines)

            return has_header and has_sequence

        except Exception as e:
            logger.warning(f"Error checking {fasta_file.name}: {e}")
            return False

    def calculate_genome_size(self, fasta_file):
        """Calculate the actual genome size from a FASTA file."""
        try:
            total_length = 0
            with open(fasta_file, 'r') as f:
                for line in f:
                    if not line.startswith('>'):
                        total_length += len(line.strip())
            return total_length
        except Exception as e:
            logger.warning(
                f"Could not calculate genome size for {fasta_file}: {e}")
            return None

    def sketch_organisms(self, k_value):
        """
        Sketch all organisms with the given k-mer size.
        Returns path to the sketch file.
        """
        logger.info(f"Sketching organisms with k={k_value}")

        organism_files = self.get_organism_files()
        if not organism_files:
            return None

        # Output sketch file
        sketch_file = self.sketches_dir / f"organisms_k{k_value}.msh"

        try:
            # Sketch each organism individually with its own genome size
            individual_sketches = []

            for organism_file in organism_files:
                # Calculate total genome size for this organism
                genome_size = self.calculate_genome_size(organism_file)

                # Create individual sketch file
                individual_sketch = self.sketches_dir / \
                    f"temp_{organism_file.stem}_k{k_value}.msh"

                # Build command for this organism
                cmd = [
                    'mash', 'sketch',
                    '-k', str(k_value),
                    '-s', str(self.sketch_size),
                    '-S', str(self.seed),
                    '-w', str(self.warning_threshold),
                    # Mash adds .msh extension
                    '-o', str(individual_sketch.with_suffix('')),
                    str(organism_file)
                ]

                # Add genome size if calculated successfully
                if genome_size:
                    # Genome size in base pairs
                    cmd.extend(['-g', str(genome_size)])
                    logger.info(
                        f"Using total genome size {genome_size:,} bp for {organism_file.name}")
                else:
                    logger.warning(
                        f"Could not calculate genome size for {organism_file.name}, using Mash default")

                # Add optional parameters
                if self.preserve_strand:
                    cmd.append('-n')

                if self.sketch_individual_sequences:
                    cmd.append('-i')

                # Run sketch for this organism
                logger.info(f"Sketching {organism_file.name}...")
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, check=True)

                    if individual_sketch.exists():
                        individual_sketches.append(individual_sketch)
                        logger.info(
                            f"Successfully sketched {organism_file.name}")
                    else:
                        logger.error(
                            f"Failed to sketch {organism_file.name} - no output file created")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to sketch {organism_file.name}: {e}")
                    if e.stderr:
                        logger.error(f"Error details: {e.stderr}")
                    # Continue with other organisms

            if not individual_sketches:
                logger.error("No organisms were successfully sketched")
                return None

            # Combine all individual sketches into one file
            logger.info(
                f"Combining {len(individual_sketches)} individual sketches...")
            combine_cmd = ['mash', 'paste', str(sketch_file.with_suffix(
                '')), *[str(s) for s in individual_sketches]]

            result = subprocess.run(
                combine_cmd, capture_output=True, text=True, check=True)

            # Clean up individual sketch files
            for individual_sketch in individual_sketches:
                if individual_sketch.exists():
                    individual_sketch.unlink()

            if sketch_file.exists():
                logger.info(
                    f"Successfully created combined sketch: {sketch_file}")
                return sketch_file
            else:
                logger.error(
                    f"Failed to create combined sketch: {sketch_file}")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"Error sketching genomes: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None

    def compute_distance_matrix(self, sketch_file, k_value):
        """
        Compute pairwise distance matrix using Mash triangle.
        """
        logger.info(f"Computing distance matrix for k={k_value}")

        if not sketch_file.exists():
            logger.error(f"Sketch file not found: {sketch_file}")
            return None

        # Output matrix file
        matrix_file = self.matrices_dir / f"distance_matrix_k{k_value}.txt"

        try:
            # Run Mash triangle command with config parameters
            cmd = [
                'mash', 'triangle',
                '-k', str(k_value),  # Ensure k-mer size matches
                '-p', str(self.parallelism),  # Use config parallelism
                str(sketch_file)
            ]

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True)

            # Save the distance matrix
            with open(matrix_file, 'w') as f:
                f.write(result.stdout)

            logger.info(f"Distance matrix saved to: {matrix_file}")
            return matrix_file

        except subprocess.CalledProcessError as e:
            logger.error(f"Error computing distance matrix: {e}")
            logger.error(f"stderr: {e.stderr}")
            return None

    def clean_organism_name(self, name):
        """Clean organism name by removing path, _merged suffix, and .fasta extension."""
        # Remove any path components (get just the filename)
        if '/' in name:
            name = name.split('/')[-1]
        # Remove .fasta extension
        if name.endswith('.fasta'):
            name = name[:-6]
        # Remove _merged suffix
        if name.endswith('_merged'):
            name = name[:-7]
        return name

    def parse_distance_matrix(self, matrix_file):
        """
        Parse Mash triangle output into a proper distance matrix.
        """
        logger.info(f"Parsing distance matrix: {matrix_file}")

        try:
            with open(matrix_file, 'r') as f:
                lines = f.readlines()

            # First line contains the number of organisms
            n_orgs = int(lines[0].strip())
            logger.info(f"Matrix contains {n_orgs} organisms")

            # Extract organism names and distances
            organism_names = []
            distances = []

            for i, line in enumerate(lines[1:], 1):  # Skip first line
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    # Clean the organism name (remove path, _merged suffix, and .fasta extension)
                    clean_name = self.clean_organism_name(parts[0])
                    organism_names.append(clean_name)
                    # Parse distance values (skip organism name)
                    if len(parts) > 1:
                        dist_values = [float(x) for x in parts[1:]]
                        distances.append(dist_values)
                    else:
                        distances.append([])

            if not distances:
                logger.error("No distance data found in matrix file")
                return None

            # Create DataFrame
            matrix = np.zeros((n_orgs, n_orgs))

            # Fill lower triangle (Mash triangle format)
            for i, dist_row in enumerate(distances):
                for j, dist in enumerate(dist_row):
                    # Lower triangle: matrix[i, j] where j < i
                    matrix[i, j] = dist

            # Make symmetric (copy lower to upper triangle)
            matrix = matrix + matrix.T

            # Create DataFrame with cleaned names
            df = pd.DataFrame(matrix, index=organism_names,
                              columns=organism_names)

            # Save as CSV
            csv_file = Path(matrix_file).with_suffix('.csv')
            df.to_csv(csv_file)
            logger.info(f"Distance matrix saved as CSV: {csv_file}")

            return df

        except Exception as e:
            logger.error(f"Error parsing distance matrix: {e}")
            return None

    def run_analysis(self):
        """
        Run complete Mash analysis for all k values.
        """
        logger.info("Starting Mash distance analysis")
        logger.info(f"K values: {self.k_values}")

        results = {}

        for k_value in self.k_values:
            logger.info(f"\n=== Processing k={k_value} ===")

            # Sketch organisms
            sketch_file = self.sketch_organisms(k_value)
            if not sketch_file:
                logger.error(f"Failed to sketch organisms for k={k_value}")
                continue

            # Compute distance matrix
            matrix_file = self.compute_distance_matrix(sketch_file, k_value)
            if not matrix_file:
                logger.error(
                    f"Failed to compute distance matrix for k={k_value}")
                continue

            # Parse and save matrix
            df = self.parse_distance_matrix(matrix_file)
            if df is not None:
                results[k_value] = df
                logger.info(f"Successfully processed k={k_value}")
            else:
                logger.error(
                    f"Failed to parse distance matrix for k={k_value}")

        # Summary
        logger.info(f"\n=== Analysis Summary ===")
        logger.info(f"Successfully processed {len(results)} k values")
        for k_value, df in results.items():
            logger.info(f"k={k_value}: {df.shape[0]} organisms")

        return results


def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    # Load configuration
    config = load_config()

    # Configuration - use merged organism files from config
    genomes_dir = config['data_paths']['genomes_dir']
    output_dir = config['data_paths']['mash_output_dir']
    k_values = config['mash_settings']['k_values']
    sketch_size = config['mash_settings']['sketch_size']
    parallelism = config['mash_settings']['parallelism']
    seed = config['mash_settings']['seed']
    warning_threshold = config['mash_settings']['warning_threshold']
    preserve_strand = config['mash_settings']['preserve_strand']
    sketch_individual_sequences = config['mash_settings']['sketch_individual_sequences']

    # Check if merged organism files directory exists
    if not Path(genomes_dir).exists():
        logger.error(
            f"Merged organism files directory not found: {genomes_dir}")
        logger.error("Please run merge_genomes_by_organism.py first")
        sys.exit(1)

    # Run analysis
    analyzer = MashAnalyzer(
        genomes_dir, output_dir, k_values, sketch_size,
        parallelism, seed, warning_threshold,
        preserve_strand, sketch_individual_sequences
    )
    results = analyzer.run_analysis()

    if results:
        logger.info(f"\nAnalysis complete! Results saved to: {output_dir}")
        logger.info(
            "Distance matrices are available in both text and CSV formats")
    else:
        logger.error("Analysis failed - no results generated")


if __name__ == "__main__":
    main()
