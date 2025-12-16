#!/usr/bin/env python3
"""
Script to fetch taxonomy information for organisms using NCBI Datasets.
Uses NCBI Datasets tool (same as genome download script) to search for organisms
and retrieve taxonomy information with all taxonomic levels.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from Bio import Entrez
from Bio.Entrez import efetch
from urllib.error import HTTPError

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def search_organism_taxonomy(organism_name: str) -> Optional[Dict]:
    """
    Search NCBI for taxonomy information of a given organism using NCBI Datasets.
    Returns taxonomy ID and basic organism information from assembly search.
    Tries multiple name variations if initial search fails.

    Args:
        organism_name: Scientific name of organism (may contain underscores)

    Returns:
        Dictionary with taxonomy_id and organism_name, or None if not found
    """
    # Prepare name variations to try
    # Convert underscores to spaces (NCBI expects spaces)
    name_with_spaces = organism_name.replace('_', ' ')

    # Also try without "str." or "strain" suffixes
    name_variations = [name_with_spaces]
    if ' str.' in name_with_spaces.lower():
        name_variations.append(name_with_spaces.replace(
            ' str.', '').replace(' str ', ' ').strip())
    if ' strain ' in name_with_spaces.lower():
        name_variations.append(
            name_with_spaces.replace(' strain ', ' ').strip())

    # Try each variation
    for name_variant in name_variations:
        result = _try_search_name(name_variant, organism_name)
        if result:
            return result

    # If all variations failed, try parsing error message for suggestions
    # Sometimes NCBI gives us suggestions with taxids
    result = _try_parse_suggestions(name_with_spaces, organism_name)
    if result:
        return result

    logger.warning(f"Could not find taxonomy for: {organism_name}")
    return None


def _try_search_name(name_to_search: str, original_name: str) -> Optional[Dict]:
    """Try searching with a specific name variation."""
    try:
        # Use NCBI Datasets to search for assemblies
        cmd = ["datasets", "summary", "genome", "taxon",
               name_to_search, "--limit", "1"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            # Parse the JSON response
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return None

            if 'reports' not in data or not data['reports']:
                return None

            # Get taxonomy ID from first assembly report
            first_report = data['reports'][0]
            tax_id = first_report.get('organism', {}).get('tax_id', '')
            organism_name_found = first_report.get(
                'organism', {}).get('organism_name', name_to_search)

            if not tax_id:
                return None

            try:
                taxonomy_id = int(tax_id)
                logger.info(
                    f"Found taxonomy ID {taxonomy_id} for {original_name} using search term '{name_to_search}' (NCBI name: {organism_name_found})")
                return {
                    'taxonomy_id': taxonomy_id,
                    'organism_name_ncbi': organism_name_found,
                    'original_query': original_name
                }
            except (ValueError, TypeError):
                return None

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def _try_parse_suggestions(name_to_search: str, original_name: str) -> Optional[Dict]:
    """Try to parse taxid from NCBI error suggestions."""
    try:
        cmd = ["datasets", "summary", "genome", "taxon",
               name_to_search, "--limit", "1"]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            # Check if error contains taxid suggestions
            stderr = result.stderr
            # Look for pattern like "taxid: 914149" in suggestions
            import re
            taxid_match = re.search(r'taxid:\s*(\d+)', stderr)
            if taxid_match:
                try:
                    taxonomy_id = int(taxid_match.group(1))
                    logger.info(
                        f"Found taxonomy ID {taxonomy_id} for {original_name} from NCBI suggestions")
                    return {
                        'taxonomy_id': taxonomy_id,
                        'organism_name_ncbi': name_to_search,
                        'original_query': original_name
                    }
                except (ValueError, TypeError):
                    pass

        return None

    except Exception:
        return None


def get_taxonomy_lineage(taxonomy_id: int, email: str, delay: float = 0.4, max_retries: int = 3) -> Optional[Dict]:
    """
    Fetch full taxonomy lineage for a given taxonomy ID using Entrez API.
    Includes retry logic for HTTP errors.

    Args:
        taxonomy_id: NCBI Taxonomy ID
        email: Email address for NCBI API
        delay: Delay in seconds between API requests
        max_retries: Maximum number of retries for HTTP errors

    Returns:
        Dictionary with taxonomic ranks and names, or None if error
    """
    # Set email for Entrez
    Entrez.email = email

    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Fetch taxonomy record
            handle = efetch(db="taxonomy", id=str(taxonomy_id), retmode="xml")
            records = Entrez.read(handle)
            handle.close()

            time.sleep(delay)

            # Check if records is valid and not empty
            if not records:
                logger.warning(
                    f"Empty records returned for taxonomy ID {taxonomy_id}")
                return None

            # Check if records is a list/sequence and has at least one element
            try:
                if not hasattr(records, '__len__'):
                    logger.warning(
                        f"Invalid records structure for taxonomy ID {taxonomy_id} (no length)")
                    return None

                if len(records) == 0:
                    logger.warning(
                        f"Empty records returned for taxonomy ID {taxonomy_id}")
                    return None

                record = records[0]
            except (IndexError, TypeError) as e:
                logger.warning(
                    f"Could not access records for taxonomy ID {taxonomy_id}: {e}")
                return None

            # Safely extract scientific name
            scientific_name = record.get(
                'ScientificName', '') if hasattr(record, 'get') else ''

            # Safely extract common name
            common_name = ''
            if 'OtherNames' in record and hasattr(record, 'get'):
                other_names = record.get('OtherNames', {})
                if isinstance(other_names, dict) and 'CommonName' in other_names:
                    common_name_list = other_names.get('CommonName', [])
                    if isinstance(common_name_list, list) and len(common_name_list) > 0:
                        common_name = common_name_list[0]

            taxonomy_data = {
                'taxonomy_id': taxonomy_id,
                'scientific_name': scientific_name,
                'common_name': common_name
            }

            # Extract lineage information from LineageEx
            if 'LineageEx' in record and hasattr(record, 'get'):
                lineage_ex = record.get('LineageEx', [])
                if isinstance(lineage_ex, (list, tuple)):
                    for lineage_item in lineage_ex:
                        if hasattr(lineage_item, 'get'):
                            rank = lineage_item.get('Rank', '').lower()
                            name = lineage_item.get('ScientificName', '')
                            if rank and name:
                                taxonomy_data[rank] = name

            # Also extract from Lineage string if LineageEx is not available
            if 'Lineage' in record and hasattr(record, 'get'):
                lineage = record.get('Lineage', '')
                if lineage:
                    taxonomy_data['lineage_string'] = lineage
                    # Parse lineage string (semicolon-separated)
                    lineage_parts = [part.strip() for part in str(
                        lineage).split(';') if part.strip()]
                    # Try to map to standard ranks (approximate fallback)
                    if lineage_parts:
                        if 'superkingdom' not in taxonomy_data and len(lineage_parts) > 0:
                            taxonomy_data['superkingdom'] = lineage_parts[0]
                        if 'phylum' not in taxonomy_data and len(lineage_parts) > 1:
                            taxonomy_data['phylum'] = lineage_parts[1]

            return taxonomy_data

        except HTTPError as e:
            # Handle HTTP errors with retry logic
            retry_count += 1
            if retry_count <= max_retries:
                # Exponential backoff
                retry_delay = 2.0 * (2 ** (retry_count - 1))
                status_code = e.code if hasattr(e, 'code') else 'unknown'
                logger.warning(
                    f"HTTP error {status_code} fetching taxonomy lineage for ID {taxonomy_id}. "
                    f"Retrying in {retry_delay:.1f} seconds (attempt {retry_count}/{max_retries})..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"HTTP error fetching taxonomy lineage for ID {taxonomy_id} after {max_retries} retries: {e}")
                return None
        except IndexError as e:
            logger.error(
                f"Index error fetching taxonomy lineage for ID {taxonomy_id}: {e}. "
                f"Records structure may be invalid.")
            return None
        except Exception as e:
            logger.error(
                f"Error fetching taxonomy lineage for ID {taxonomy_id}: {e}")
            return None

    return None


def fetch_organism_taxonomy(organism_name: str, email: str, delay: float = 0.4) -> Dict:
    """
    Fetch complete taxonomy information for an organism.
    First uses NCBI Datasets to get taxonomy ID, then uses Entrez to get full lineage.

    Args:
        organism_name: Original organism name
        email: Email address for NCBI API
        delay: Delay in seconds between API requests

    Returns:
        Dictionary with all taxonomy information
    """
    # Initialize result structure with all taxonomic ranks
    result = {
        'organism_name': organism_name,
        'taxonomy_id': None,
        'scientific_name': '',
        'common_name': '',
        'superkingdom': '',
        'kingdom': '',
        'phylum': '',
        'class': '',
        'order': '',
        'family': '',
        'genus': '',
        'species': '',
        'subspecies': '',
        'strain': '',
        'lineage_string': '',
        'found': False
    }

    # Step 1: Use NCBI Datasets to find taxonomy ID (same approach as genome download)
    taxonomy_info = search_organism_taxonomy(organism_name)

    if not taxonomy_info or 'taxonomy_id' not in taxonomy_info:
        logger.warning(f"No taxonomy ID found for: {organism_name}")
        return result

    taxonomy_id = taxonomy_info['taxonomy_id']
    result['taxonomy_id'] = taxonomy_id

    # Step 2: Get full taxonomy lineage using Entrez API
    taxonomy_data = get_taxonomy_lineage(taxonomy_id, email, delay)

    if not taxonomy_data:
        logger.warning(
            f"Could not fetch taxonomy lineage for ID {taxonomy_id}")
        return result

    # Update result with taxonomy data
    result['found'] = True
    result['scientific_name'] = taxonomy_data.get('scientific_name', '')
    result['common_name'] = taxonomy_data.get('common_name', '')

    # Map all standard ranks
    standard_ranks = [
        'superkingdom', 'kingdom', 'phylum', 'class', 'order',
        'family', 'genus', 'species', 'subspecies', 'strain'
    ]
    for rank in standard_ranks:
        if rank in taxonomy_data:
            result[rank] = taxonomy_data[rank]

    if 'lineage_string' in taxonomy_data:
        result['lineage_string'] = taxonomy_data['lineage_string']

    # If strain/subspecies not found in taxonomy but present in original name, extract it
    if not result.get('strain') and not result.get('subspecies'):
        full_name = organism_name.replace('_', ' ')
        parts = full_name.split()
        if len(parts) > 2:
            additional_parts = ' '.join(parts[2:])
            if not result.get('subspecies') and len(parts) == 3:
                result['subspecies'] = additional_parts
            elif not result.get('strain'):
                result['strain'] = additional_parts

    return result


def get_organism_names_from_summary(summary_file: Path) -> List[str]:
    """Extract organism names from organism summary CSV file."""
    df = pd.read_csv(summary_file)
    if 'organism' in df.columns:
        return list(df['organism'].unique())
    else:
        logger.warning(f"'organism' column not found in {summary_file}")
        return []


def load_config():
    """Load configuration from JSON file."""
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """Main function to fetch taxonomy for organisms."""
    # Load configuration
    config = load_config()

    # Configuration
    email = config['ncbi_settings']['email']
    delay_between_requests = config['ncbi_settings']['delay_between_requests']
    delay_between_organisms = config['ncbi_settings'].get(
        'delay_between_organisms', 2.0)
    mash_output_dir = Path(config['data_paths']['mash_output_dir'])

    # Output file
    output_dir = Path(__file__).parent
    output_file = output_dir / "organism_taxonomy.csv"

    # Get organism names from organism summary statistics file (primary source)
    genome_analysis_dir = Path(config['data_paths'].get('genome_analysis_dir',
                                                        mash_output_dir.parent / "genome_analysis_results"))
    summary_file = genome_analysis_dir / "organism_summary_statistics.csv"

    if not summary_file.exists():
        logger.error(f"Organism summary file not found: {summary_file}")
        sys.exit(1)

    logger.info(f"Using organism names from: {summary_file}")
    organism_names = get_organism_names_from_summary(summary_file)

    if not organism_names:
        logger.error("No organism names found")
        sys.exit(1)

    logger.info(f"Found {len(organism_names)} organisms")
    logger.info("Fetching taxonomy for all organisms using NCBI Datasets...")

    # Fetch taxonomy for all organisms
    results = []
    total = len(organism_names)
    found_count = 0

    for i, organism_name in enumerate(organism_names, 1):
        if i % 10 == 0:
            logger.info(f"Processing organism {i}/{total}: {organism_name}")

        result = fetch_organism_taxonomy(
            organism_name, email, delay_between_requests)
        results.append(result)

        if result['found']:
            found_count += 1

        # Progress update
        if i % 50 == 0:
            logger.info(f"Progress: {i}/{total} ({found_count} found so far)")

        # Rate limiting between organisms
        if i < total:
            time.sleep(delay_between_organisms)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(output_file, index=False)
    logger.info(f"Taxonomy data saved to: {output_file}")

    # Print summary
    logger.info(f"\n=== Summary ===")
    logger.info(f"Total organisms: {total}")
    logger.info(f"Found in NCBI Taxonomy: {found_count}")
    logger.info(f"Not found: {total - found_count}")

    # Show sample of results
    if found_count > 0:
        logger.info("\n=== Sample Results ===")
        sample = df[df['found']].head()
        for _, row in sample.iterrows():
            logger.info(f"\n{row['organism_name']}:")
            logger.info(f"  Taxonomy ID: {row['taxonomy_id']}")
            logger.info(f"  Scientific Name: {row['scientific_name']}")
            logger.info(f"  Genus: {row['genus']}")
            logger.info(f"  Species: {row['species']}")
            if row['phylum']:
                logger.info(f"  Phylum: {row['phylum']}")


if __name__ == "__main__":
    main()
