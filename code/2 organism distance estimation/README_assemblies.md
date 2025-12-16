# Assembly-Based Genome Download Script

This script downloads genome assemblies and their chromosomes from NCBI using the NCBI Datasets tool, maintaining a comprehensive mapping between chromosome files, assemblies, and organisms.

## Prerequisites

1. **Install NCBI Datasets CLI**: Download and install from [NCBI Datasets](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/download-and-install/)
2. **Python dependencies**: Install with `pip install -r requirements_assemblies.txt`

## Key Features

- **Assembly Search**: Uses NCBI Entrez API to find genome assemblies for organisms
- **Chromosome Download**: Downloads all chromosomes from selected assemblies using NCBI Datasets
- **Mapping System**: Maintains detailed mapping between chromosome files, assemblies, and organisms
- **Metadata Tracking**: Saves assembly metadata including accession numbers, organism names, chromosome names, etc.
- **File Organization**: Organizes files by organism with clear naming conventions

## Usage

```bash
python 1 download_genomes_assemblies.py
```

## Configuration

The script uses `config.json` for configuration. Key settings:

- `max_organisms`: Limit number of organisms to process (null for all)
- `max_assemblies_per_organism`: Maximum assemblies to download per organism (default: 3)
- `delay_between_organisms`: Delay between processing different organisms (default: 2.0 seconds)
- `delay_between_assemblies`: Delay between downloading different assemblies (default: 2.0 seconds)

## Output Structure

```
individual_genomes_dir/
├── organism_name_1/
│   ├── organism_name_1_assembly_accession_1_chromosome_1.fasta
│   ├── organism_name_1_assembly_accession_1_chromosome_2.fasta
│   ├── organism_name_1_assembly_accession_1_metadata.json
│   └── ...
├── organism_name_2/
│   └── ...
├── chromosome_assembly_mapping.csv
├── chromosome_assembly_mapping.json
└── failed_organisms.txt
```

## Mapping Files

### CSV Format (`chromosome_assembly_mapping.csv`)
- `filename`: Name of the chromosome file
- `organism`: Organism name
- `assembly_accession`: NCBI assembly accession number
- `chromosome_name`: Name of the chromosome
- `file_path`: Full path to the chromosome file
- `header`: FASTA header line

### JSON Format (`chromosome_assembly_mapping.json`)
Detailed metadata for each chromosome file including all assembly information.

## Advantages over Nucleotide Database Approach

1. **Better Quality**: Assembly data is curated and represents complete genomes
2. **Organized Structure**: Chromosomes are properly separated and named
3. **Rich Metadata**: Assembly information provides context about genome quality and completeness
4. **Traceability**: Clear mapping between files and their source assemblies
5. **Efficiency**: Downloads complete assemblies rather than searching individual sequences

## Error Handling

- Validates downloaded FASTA files
- Cleans up corrupted files automatically
- Tracks failed organisms and assemblies
- Provides detailed logging of all operations
- Handles rate limiting to respect NCBI guidelines

## Rate Limiting

The script implements appropriate delays:
- 2 seconds between organisms
- 2 seconds between assemblies
- Respects NCBI's 3 requests per second limit for Entrez API

## Troubleshooting

1. **NCBI Datasets not found**: Ensure the `datasets` command is in your PATH
2. **Permission errors**: Check write permissions for output directory
3. **Network timeouts**: Increase timeout values in the script if needed
4. **Memory issues**: Process fewer organisms at once by setting `max_organisms`
