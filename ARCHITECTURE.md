# System Architecture

Protein Explorer Atlas is designed strictly for low-latency edge deployment. Instead of coupling heavy static databases, we decouple the neural pipeline from external network calls, prioritizing compute efficiency.

## 1. Zero-Shot Inference Pipeline
We intentionally deprecated custom downstream CNN architectures in favor of **Zero-Shot inference** via the foundation model's Masked Language Modeling (MLM) head. 

**Why EsmForMaskedLM?**
Traditional architectures require embedding the entire protein, then passing those vectors through Secondary ML models (like Random Forests or CNNs) trained on vast, biased static datasets. 
By pulling predictions strictly from the native MLM head of the ESM-2 transformer, the system accesses the evolutionary constraints already learned by the foundation model. This eliminates technical debt, eliminates the risk of training-set bias, and requires significantly less VRAM.

## 2. Batched Tensor Processing
Variant effect prediction natively requires evaluating every single mutated position in isolation. To prevent a massive latency penalty when a user submits multiple simultaneous edits, the inference engine relies on **batched tensor cloning**.

1. The Wild-Type (baseline) sequence is tokenized exactly once.
2. For $N$ mutated positions, the tensor is cloned into a batch shape of `(N, seq_len + 2)`.
3. The `<mask_token>` is applied vector-wise to the specific target positions.
4. A single `torch.no_grad()` forward pass is executed on the batch, guaranteeing $O(1)$ GPU utilization scaling.

## 3. Structural Discovery Fallback Network
To provide complete end-to-end visualization of the affected protein, we implemented a cascading API fallback.

1. **RCSB PDB (Experimental):** The system parses the FASTA header or generated MD5 hash to query the RCSB database for matching UniProt IDs. If X-ray crystallography or Cryo-EM data exists, it fetches the raw PDB coordinates.
2. **AlphaFold EBI API (Predicted):** Approximately 30% of the human proteome lacks experimental data. If RCSB returns a 404/Empty result, the system automatically redirects the query to the EBI AlphaFold API, fetching the highly-accurate computationally predicted structure.
3. The frontend `3Dmol.js` viewer receives the normalized coordinate URL, alongside a generated UI Badge highlighting the structural data source to ensure scientific transparency.
