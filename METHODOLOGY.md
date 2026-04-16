# Scientific Methodology

Protein Explorer Atlas utilizes two distinct mathematical algorithms to translate raw transformer representations into interpretable biochemical signals. Both calculations are executed without secondary fine-tuning (Zero-Shot).

## 1. Variant Effect: Masked Marginal Log-Likelihood Ratio (LLR)
To quantify the functional or structural disruption introduced by a user's *in-silico* mutation, the system calculates the variant's Masked Marginal Log-Likelihood Ratio.

For an amino acid sequence with a specific residue $x$ at position $i$ mutated to $y$:

1. The engine constructs a masked sequence where position $i$ is replaced with the `<mask_token>`.
2. The foundation model executes a forward pass, outputting the predicted vocabulary distributions over that exact position based entirely on the surrounding context.
3. The LLR is computed as the difference in the log-probabilities of the mutant and wild-type amino acids:

$$ \text{LLR} = \log P_{MLM}(y \mid \text{context}) - \log P_{MLM}(x \mid \text{context}) $$

**Interpretation:**
- **Negative LLR (Red):** The network assigns a lower probability to the mutant than the wild-type. Evolution strongly selects against this change, suggesting a destabilizing or pathogenic mutation.
- **Positive LLR (Blue):** The mutant is predicted to be easily accommodated or potentially selected for.
- **Sparse Execution:** To avoid artificial noise and "rolling average" blurring across the topological sequence, the engine limits this score strictly to the indices the user actively altered.

## 2. Conservation Profile: Shannon Entropy
To establish a sequence's core stability layout—its "Hotspots"—the system computes per-position conservation via Shannon Entropy from the unmasked wild-type logits.

Rather than running heavy local MSAs (Multiple Sequence Alignments) through HMMER or BLAST, the probability distributions are directly yielded by the model:

$$ H(X) = - \sum_{i=1}^{20} P(AA_i) \log_2 P(AA_i) $$

*Note: The calculation strictly limits the distribution to the 20 biological amino acid tokens, stripping specialized pad/eos tokens from the tensor before computing the final sum.*

**Interpretation:**
- **Low Entropy (< 1.5 bits):** The foundation model assigns high confidence to only 1–2 specific amino acids holding that position. The residue is structurally vital (e.g., active site, buried core).
- **High Entropy (> 2.5 bits):** The context freely tolerates heavy substitutions. The region is likely an exposed loop or non-critical linker.
