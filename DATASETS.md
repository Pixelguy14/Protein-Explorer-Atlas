# Datasets for Protein Explorer Atlas

This document provides information about the public datasets used in the Protein Explorer Atlas project and a guide on how to download and prepare them for use with the provided scripts.

## 1. Overview of Datasets

Our project utilizes several public datasets to gather comprehensive information about human proteins and their structures:

*   **UniProt Human Proteome (Amino Acid Sequences):** Provides the amino acid sequences for all proteins in the human proteome. This is fundamental for any sequence-based analysis.
*   **Pfam-A.regions.tsv:** Contains information about protein domains (Pfam families) found within protein sequences, mapping UniProt IDs to Pfam family identifiers. This helps in understanding the functional and structural modules of proteins.
*   **Pfam-A.hmm.dat:** The Pfam Hidden Markov Model (HMM) library. These models are used to identify protein domains and families in new sequences.
*   **Human Protein Atlas (HPA) Annotation Data:** Offers extensive annotations for human proteins, including protein class, biological process, and molecular function, derived from various experimental and computational sources.
*   **InterPro Annotation Data:** Provides functional predictions for proteins, integrating information from various protein family and domain databases. This includes mappings of UniProt proteins to InterPro entries.


## 2. How to Download and Organize the Datasets

All datasets should be downloaded into specific subdirectories within the `data/` directory of your project's root.

**Important:** Many of these files are compressed (`.gz` or `.zip`). After downloading, you will need to decompress them.

1.  **Create the necessary subdirectories:**
    Navigate to your project's root directory and run:
    ```bash
    mkdir -p data/{pfam,hpa,interpro,pdb}
    ```

2.  **Navigate to your project's `data` directory:**
    ```bash
    cd /home/pixel/Downloads/Hackaton/Protein-Explorer-Atlas/data
    ```

3.  **Download Commands:**

    *   **UniProt Human Proteome (Amino Acid Sequences):**
        ```bash
        wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz
        ```
        *   **Decompress:**
            ```bash
            gunzip UP000005640_9606.fasta.gz
            ```
        *   This will result in `UP000005640_9606.fasta` in the `data/` directory.

    *   **Pfam-UniProt Mapping (Pfam-A.regions.tsv):**
        ```bash
        wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.regions.tsv.gz -P pfam/
        ```
        *   **Decompress:**
            ```bash
            gunzip pfam/Pfam-A.regions.tsv.gz
            ```
        *   This will result in `Pfam-A.regions.tsv` in the `data/pfam/` directory. **Note:** This is a very large file (approx. 9.8 GB uncompressed) and will take time to download and decompress.

    *   **Pfam HMM Library (Pfam-A.hmm.dat):**
        ```bash
        wget https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.dat.gz -P pfam/
        ```
        *   **Decompress:**
            ```bash
            gunzip pfam/Pfam-A.hmm.dat.gz
            ```
        *   This will result in `Pfam-A.hmm.dat` in the `data/pfam/` directory.

    *   **Human Protein Atlas (HPA) Annotation Data:**
        ```bash
        wget https://www.proteinatlas.org/download/proteinatlas.tsv.zip -P hpa/
        ```
        *   **Decompress:**
            ```bash
            unzip hpa/proteinatlas.tsv.zip -d hpa/
            ```
        *   This will result in `proteinatlas.tsv` in the `data/hpa/` directory.

    *   **InterPro Annotation Data (protein2ipr.dat):**
        ```bash
        wget ftp://ftp.ebi.ac.uk/pub/databases/interpro/current/protein2ipr.dat.gz -P interpro/
        ```
        *   **Decompress:**
            ```bash
            gunzip interpro/protein2ipr.dat.gz
            ```
        *   This will result in `protein2ipr.dat` in the `data/interpro/` directory.

    

## 3. Guide to Training with Downloaded Datasets

Once you have downloaded and decompressed all the necessary dataset files into their respective subdirectories within `data/`, you can proceed with the protein data processing and model training workflow:

1.  **Install Dependencies:**
    Ensure all required Python libraries are installed. Navigate to your project's root directory and run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Process and Unify Data:**
    Run the `data_processing.py` script to parse, merge, and initially clean the raw dataset files. This script will generate `unified_protein_data.tsv`.
    ```bash
    python data_processing.py
    ```
    *   **Important Note:** The `data_processing.py` script currently expects `Pfam-A.regions.tsv`, `proteinatlas.tsv`, and `UP000005640_9606.fasta` to be directly in the `data/` directory. If you have downloaded them into subdirectories (e.g., `data/pfam/Pfam-A.regions.tsv`), you will need to **update the file paths within `data_processing.py`** accordingly.
    *   **Note:** This step can take a significant amount of time due to the large Pfam file.

3.  **Clean Data and Generate ESM-2 Embeddings:**
    Execute the `process_and_embed.py` script. This will perform further data cleaning and generate ESM-2 embeddings for each protein sequence, saving the result to `cleaned_and_embedded_protein_data.tsv`.
    ```bash
    python process_and_embed.py
    ```
    *   **Note:** This step is computationally intensive and requires substantial RAM.

4.  **Train the Protein Classification Model (CNN):**
    Finally, run the `classify_proteins_cnn.py` script to train the Convolutional Neural Network model for protein family classification.
    ```bash
    python classify_proteins_cnn.py
    ```

By following these steps, you will prepare your data and train the protein classification model using the specified public datasets.