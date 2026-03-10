# GSEA Proteomics Visualizer

A command-line tool for generating publication-quality cohort-level summary figures from GSEA preranked output, designed for Drosophila melanogaster ASD proteomics studies.

## Overview

This tool ingests per-mutant GSEA preranked output files, performs cohort-level meta-analysis, and produces three figures:

- **Figure 1** (optional): Hypothesis-driven dot plot showing GO terms from user-specified biological categories
- **Figure 2**: Unbiased dot plot showing the top data-driven GO terms grouped by hierarchical clustering
- **Figure 3**: Meta-analysis bar plot showing representative dysregulated pathways from Fisher's combined probability test

## Installation

### Using conda (recommended)

```bash
conda create -n gseav2 python=3.11
conda activate gseav2
pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Usage

Run the tool from your project directory (the directory containing your `data/` folder):

```bash
# Produce Figure 2 and Figure 3 only (unbiased + meta-analysis)
gsea-tool

# Also produce Figure 1 using a category mapping TSV file
gsea-tool path/to/category_mapping.tsv
```

The tool expects a `data/` directory containing one subdirectory per mutant line. Each mutant subdirectory name must follow the format `<mutant_id>.GseaPreranked.<timestamp>` and must contain exactly one positive and one negative GSEA report TSV file.

Output files are written to `output/` in the current working directory.

## Configuration

Create a `config.yaml` file in your project directory to customize parameters. If no `config.yaml` is present, all defaults are used.

Example `config.yaml`:

```yaml
cherry_pick:
  - go_id: "GO:0005739"
    label: "Mitochondria"
  - go_id: "GO:0006412"
    label: "Translation"

dot_plot:
  fdr_threshold: 0.05
  top_n: 20
  n_groups: 4

fisher:
  prefilter_pvalue: 0.05
  top_n_bars: 20

clustering:
  enabled: true
  similarity_threshold: 0.7

plot:
  dpi: 300
  font_family: Arial
```

See the generated `notes.md` in the output directory for a full configuration guide.

## Input Data Format

Each mutant subfolder must contain:
- `gsea_report_for_na_pos_*.tsv` -- positive enrichment report
- `gsea_report_for_na_neg_*.tsv` -- negative enrichment report

These are standard GSEA preranked output files. The tool reads the NAME, NES, FDR q-val, NOM p-val, and SIZE columns.

## Output Files

The tool writes the following files to `output/`:

| File | Description |
|------|-------------|
| `figure1_cherry_picked.{pdf,png,svg}` | Hypothesis-driven dot plot (if produced) |
| `figure2_unbiased.{pdf,png,svg}` | Unbiased selection dot plot |
| `figure3_meta_analysis.{pdf,png,svg}` | Meta-analysis bar plot |
| `pvalue_matrix.tsv` | GO term x mutant nominal p-value matrix |
| `fisher_combined_pvalues.tsv` | Fisher combined p-values with cluster assignments |
| `notes.md` | Figure legends, methods text, and reproducibility notes |

## Requirements

- Python >= 3.11
- matplotlib >= 3.7
- numpy >= 1.24
- scipy >= 1.10
- pyyaml >= 6.0

## Reference

Figure format modeled on: Gordon et al. 2024, Figure 3a.
