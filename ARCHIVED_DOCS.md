# Archived Documentation

This file preserves the historical context and technical decisions of the Protein Explorer Atlas through its early development phases.

---

## [ARCHIVED V1] Technical Architecture: Interactive Inference Stack

> [!WARNING]
> **DEPRECATION NOTICE**: This document outlines the V1 architecture, which relied on downstream CNN training and embedding diffs. This architecture was formally deprecated in flavor of the Zero-Shot Masked Language Modeling (MLM) pipeline documented in `ARCHITECTURE.md`.

### 1. The Models (V1)
- **The Feature Extractor (ESM-2)**: Meta's 35M-parameter Transformer model (`facebook/esm2_t12_35M_UR50D`). It mapped sequences into a 320-dimensional biological latent space. This checkpoint was chosen for its balance between performance and CPU-friendliness.
- **The Classifier (PyTorch 1D CNN)**: A `Conv1D` model designed to capture local sequential motifs. 
    - **Pivot Rationale**: While sound in theory, the CNN only achieved ~34% accuracy for family classification. We dropped it to focus compute on VEP via the native ESM-2 head.
- **Dimensionality Reduction (UMAP)**: Pre-trained on a 931-sequence baseline to provide a 2D visualization of the latent space.
    - **Pivot Rationale**: Removed to streamline the UI and maintain focus on the 1D mutation labs.

### 2. Backend Infrastructure (V1)
- **`app/api/routes.py`**: Contained the original `/predict` engine.
    - **In-silico Mutagenesis**: Originally calculated the L2 Norm difference across latent space vectors.
    - **Pivot Rationale (Attention Leakage)**: L2 Norms on embeddings resulted in "blurry" heatmaps where mutating one residue affected everything. Substituted with **Masked Marginal LLR** for sparse, position-specific signals.
- **Dynamic PDB Binding**: Direct RCSB query.
    - **Pivot Rationale**: Added AlphaFold fallback to Bridge the "30% Proteome Gap" where experimental structures were missing.

### 3. Frontend Implementation (V1)
- **Plotly.js**: Originally rendered the UMAP clusters and the L2-based heatmaps.
- **3Dmol.js**: Standardized for experimental structural viewing.

### 4. Obsolete Scripting Foundation
- `data_processing.py`: Chunked parser for Pfam datasets.
- `process_and_embed.py`: Batch embedding generator.
- `generate_umap.py`: Helper for the legacy background projection.

---

## [ARCHIVED V0] Original Dataset Context

*Legacy notes recovered from `Annotations Datasets.txt` (2025)*

### 1. The Pfam-A Foundation
The project originally intended to leverage the full **Pfam-A.full** dataset (~50GB+ uncompressed). 
- **Scope**: Multi-class classification across 19,000+ protein families.
- **Challenge**: Computational cost of embedding millions of sequences via ESM-2.
- **Outcome**: Switched to a curated subset of 931 reviewed sequences for the V1 Hackathon prototype.

### 2. Secondary Datasets Explored
- **SwissProt/UniProt**: Used for ground-truth review labels.
- **RCSB PDB**: Used strictly as a manual lookup table (v0) before the dynamic cascading API implementation.

### 3. Historical Data Processing Strategy
Initially, the project used a `combine_data.py` strategy to merge sequence information with external superfamily labels. This generated the `unified_protein_data.tsv` file used in the V1 training loop. 

**Scientific Pivot**: In 2025, we realized that *superfamily classification* is a solved problem via HMMER/BLAST, and the true "Big Bet" for ML in structural biology is **live, zero-shot variant effect prediction**. We therefore purged these massive datasets to focus strictly on foundational transformer logits.