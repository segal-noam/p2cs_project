# Mash Configuration Parameters

## Updated Configuration Options

The `config.json` file now includes comprehensive Mash parameters for both sketching and distance computation.

### Mash Sketch Parameters

| Parameter | Default | Description | Mash Flag |
|-----------|---------|-------------|-----------|
| `k_values` | [15, 19, 21, 31] | K-mer sizes for sketching | `-k` |
| `sketch_size` | 10000 | Number of k-mers to sketch per genome | `-s` |
| `parallelism` | 4 | Number of threads for parallel processing | `-p` |
| `seed` | 42 | Seed for hash function | `-S` |
| `warning_threshold` | 0.01 | Probability threshold for k-mer size warnings | `-w` |
| `preserve_strand` | false | Whether to preserve strand information | `-n` |
| `sketch_individual_sequences` | false | Sketch individual sequences vs whole files | `-i` |
| `aggregation_methods` | ["geometric_mean", "median", "minimum", "average"] | Methods for organism-level aggregation | N/A |
| `create_organism_matrices` | true | Whether to create organism-level matrices | N/A |

### Mash Triangle Parameters

| Parameter | Description | Mash Flag |
|-----------|-------------|-----------|
| `parallelism` | Number of threads for distance computation | `-p` |
| `k_values` | K-mer size (must match sketch) | `-k` |

## Example Commands Generated

### Sketch Command (per genome):
```bash
mash sketch -k 15 -s 10000 -S 42 -w 0.01 -g <individual_genome_size> -o genome_k15 genome.fasta
```

### Triangle Command:
```bash
mash triangle -k 15 -p 4 genomes_k15.msh
```

## Configuration Benefits

### 1. **Individual Genome Size for P-values**
- **Benefit**: Calculates actual genome size for each FASTA file individually
- **Usage**: Each genome uses its own size for accurate p-value calculations
- **Advantage**: Most accurate p-values since each genome has its own size estimate

### 2. **Parallel Processing**
- **Parameter**: `parallelism: 4`
- **Benefit**: Faster processing for large datasets
- **Usage**: Both sketching and distance computation use multiple threads

### 3. **Reproducible Results**
- **Parameter**: `seed: 42`
- **Benefit**: Consistent results across runs
- **Usage**: Same seed ensures identical hash functions

### 4. **Quality Control**
- **Parameter**: `warning_threshold: 0.01`
- **Benefit**: Warnings for potentially problematic k-mer sizes
- **Usage**: Mash warns if k-mer size might be too small for genome

### 5. **Strand Handling**
- **Parameter**: `preserve_strand: false`
- **Benefit**: Canonical k-mers (ignores strand direction)
- **Usage**: Standard for most genomic analyses

### 6. **Aggregation Methods**
- **Parameter**: `aggregation_methods: ["geometric_mean", "median", "minimum", "average"]`
- **Benefit**: Multiple methods for organism-level aggregation
- **Usage**: Choose appropriate method for your analysis type

### 7. **Organism Matrix Creation**
- **Parameter**: `create_organism_matrices: true`
- **Benefit**: Automatically creates organism-level matrices
- **Usage**: Set to false if you only want genome-level matrices

## Customization Examples

### For Large Genomes (>10M bp):
```json
{
  "sketch_size": 20000,
  "parallelism": 8
}
```

### For Small Genomes (<1M bp):
```json
{
  "sketch_size": 5000,
  "parallelism": 2
}
```

### For High Precision:
```json
{
  "sketch_size": 50000,
  "parallelism": 1,
  "preserve_strand": true
}
```

### For Speed:
```json
{
  "sketch_size": 5000,
  "parallelism": 8,
  "sketch_individual_sequences": true
}
```

## Memory and Performance Considerations

- **Sketch Size**: Higher values = more memory, better accuracy
- **Parallelism**: Higher values = faster processing, more CPU usage
- **Genome Size**: Automatically estimated from FASTA content for each genome
- **K-mer Sizes**: Lower k = more sensitive, higher k = more specific

## Validation

The script automatically validates:
- K-mer size consistency between sketch and triangle
- File existence and accessibility
- Parameter ranges and types
- Mash installation and version compatibility
