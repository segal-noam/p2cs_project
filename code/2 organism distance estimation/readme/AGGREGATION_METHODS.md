# Aggregation Methods for Mash Distance Matrices

## Understanding Mash Distances

Mash distances approximate **1 - ANI (Average Nucleotide Identity)**:
- **Distance = 0.0** → **ANI = 1.0** (100% identity)
- **Distance = 0.1** → **ANI = 0.9** (90% identity)  
- **Distance = 0.5** → **ANI = 0.5** (50% identity)
- **Distance = 1.0** → **ANI = 0.0** (0% identity)

## Recommended Aggregation Methods

### 1. **Geometric Mean (Recommended)**
```python
# Convert distances to ANI, take geometric mean, convert back
anis = 1 - distances
geo_mean_ani = exp(mean(log(anis)))
aggregated_distance = 1 - geo_mean_ani
```

**Why it's best:**
- ANI is multiplicative in nature
- Preserves the biological relationship
- Most appropriate for phylogenetic analysis

**Use case:** General phylogenetic analysis, most accurate representation

### 2. **Median**
```python
aggregated_distance = median(distances)
```

**Why it's good:**
- Robust to outliers
- Good for mixed datasets with some very divergent genomes
- Conservative estimate

**Use case:** When you have some genomes that are very different from others

### 3. **Minimum**
```python
aggregated_distance = min(distances)
```

**Why it's useful:**
- Captures the most similar relationship between organisms
- Conservative estimate of relatedness
- Good for species-level comparisons

**Use case:** When you want to know the closest relationship between organisms

### 4. **Average (Arithmetic Mean)**
```python
aggregated_distance = mean(distances)
```

**Why it's included:**
- Simple and intuitive
- Good baseline for comparison
- Standard approach in many studies

**Use case:** When you want a straightforward average

## Method Comparison

| Method | Best For | Biological Interpretation |
|--------|----------|---------------------------|
| **Geometric Mean** | Phylogenetic analysis | Most accurate ANI representation |
| **Median** | Mixed datasets | Robust to outliers |
| **Minimum** | Species identification | Closest relationship |
| **Average** | General use | Simple average |

## Example Results

For distances [0.1, 0.2, 0.3] (ANIs [0.9, 0.8, 0.7]):

- **Arithmetic Mean**: 0.2 (ANI = 0.8)
- **Geometric Mean**: 0.18 (ANI = 0.82)
- **Median**: 0.2 (ANI = 0.8)
- **Minimum**: 0.1 (ANI = 0.9)

## Recommendations

### For Your P2CS Analysis:

1. **Primary**: Use **geometric_mean** - most biologically accurate
2. **Secondary**: Use **median** - robust to outliers
3. **Comparison**: Include **minimum** - shows closest relationships
4. **Baseline**: Include **average** - standard approach

### Output Files:
- `organism_distance_matrix_k15_geometric_mean.csv` (recommended)
- `organism_distance_matrix_k15_median.csv` (robust)
- `organism_distance_matrix_k15_minimum.csv` (conservative)
- `organism_distance_matrix_k15_average.csv` (baseline)

## Usage

```bash
# Run aggregation with all methods
python aggregate_distances.py

# Or run Mash analysis (includes automatic aggregation)
python mash_distance_analysis.py
```

The geometric mean is the most appropriate method for your phylogenetic analysis since it properly handles the multiplicative nature of ANI relationships.
