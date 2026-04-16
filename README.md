<div align="center">
  <img src="Im proud to have participated Template.png" alt="Protein Explorer Atlas" />
  <h1>Protein Explorer atlas</h1>
  
  <h3>Real-Time, Zero-Shot Variant Effect Prediction on Consumer Hardware</h3>
  
  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
    <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  </p>
</div>

## The Vision
Compute is expensive, and biology is slow. Pre-screening protein variants before wet-lab synthesis usually requires hours of compute on enterprise HPC infrastructure. 

**Protein Explorer Atlas** democratizes structural biology by bringing sub-second, interactive *in-silico* mutagenesis to edge devices. By leveraging Meta's highly efficient ESM-2 (35M parameters) foundation model and exposing its Masked Language Modeling (MLM) head, we achieve zero-shot, biologically accurate Variant Effect Prediction (VEP) and Conservation scoring without relying on custom downstream training or A100 GPU clusters. 

This empowers biotech startups, EdTech undergrad programs, and global researchers to mutate, analyze, and visualize structural constraints in real time, directly in the browser.

## Features
- **In-Silico Mutagenesis Lab:** Edit a sequence and instantly compute a Masked Marginal Log-Likelihood (LLR) variant effect.
- **Real-Time Conservation Profiling:** Identifies functional hotspots via per-position 20-AA Shannon Entropy.
- **Universal Structural Coverage:** Dynamic fallback integration across RCSB PDB (Experimental) and AlphaFold EBI APIs (Predicted) guarantees 3D visualization. 
- **Sub-Second Edge Inference:** Optimized 1D tensor batching achieves near-instantaneous neural throughput.

## Quick Start
You can run the engine directly using Python or via a containerized Docker build.

### Option 1: Native Local Installation
Requires Python 3.10+ and a standard CPU (CUDA optional but recommended).
```bash
git clone https://github.com/yourusername/Protein-Explorer-Atlas.git
cd Protein-Explorer-Atlas

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

# Boot the engine
python run.py
```
*The web app will be available at `http://localhost:5000`*

### Option 2: Docker Deployment
For an isolated, production-ready environment:
```bash
docker compose up --build
```

You can quickly test this by loading the .fasta provided in the repository or by uploading your own sequence.

## System Requirements
- 4GB System RAM minimum
- Optional: CUDA-compatible GPU (auto-detected and utilized for accelerated tensor math).

## Documentation Index
- [Architecture Details](./ARCHITECTURE.md) - System design and the Zero-Shot Pipeline.
- [Scientific Methodology](./METHODOLOGY.md) - Algorithmic explanations for LLR and Shannon Entropy.