/**
 * Protein Explorer Atlas — Zero-Shot VEP Frontend
 * No CNN. No UMAP. Pure foundation model inference.
 * Conservation (Entropy) + Variant Effect (Masked Marginal LLR)
 */

const AtlasState = {
    original: { sequence: "", proteinId: "" },
    baselineEntropy: null,
    isLoading: false
};

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("proteinForm");
    const fileInput = document.getElementById("proteinFile");
    const pasteInput = document.getElementById("pasteSequence");
    const processBtn = document.getElementById("processBtn");
    const loadingSection = document.getElementById("loadingSection");
    const resultsSection = document.getElementById("resultsSection");

    // --- Input Mutex: file clears paste, paste clears file ---
    if (fileInput && pasteInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                pasteInput.value = "";
                const label = document.querySelector(".file-input-label span");
                if (label) label.textContent = `File: ${fileInput.files[0].name}`;
            }
        });
        pasteInput.addEventListener("input", () => {
            if (pasteInput.value.trim()) {
                fileInput.value = "";
                const label = document.querySelector(".file-input-label span");
                if (label) label.textContent = "Upload FASTA file";
            }
        });
    }

    // --- Global View Callbacks ---
    window.view3D = function(pdbId, pdbUrl, source) {
        if (!pdbId) { alert("No structural data available."); return; }
        const qs = pdbUrl ? `?url=${encodeURIComponent(pdbUrl)}&source=${source || 'RCSB'}` : '';
        openModal(
            `<i class="fas fa-cube"></i> 3D Structure — ${pdbId} (${source || 'RCSB'})`,
            `/view3d/${pdbId}${qs}`
        );
    };

    window.viewDomains = function(proteinId) {
        if (!proteinId) return;
        openModal(`<i class="fas fa-puzzle-piece"></i> Domains: ${proteinId}`, `/plot/${proteinId}`);
    };

    window.viewRCSB = function(pdbId) {
        if (!pdbId) return;
        window.open(`https://www.rcsb.org/structure/${pdbId}`, '_blank');
    };

    // --- Form Submission ---
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        const pastedSeq = pasteInput ? pasteInput.value.trim() : "";

        if (!file && !pastedSeq) {
            alert("Please upload a FASTA file or paste a sequence.");
            return;
        }

        const formData = new FormData();
        if (file) formData.append("protein_file", file);
        else formData.append("sequence", pastedSeq);

        setLoading(true);
        try {
            const resp = await fetch("/api/predict", { method: "POST", body: formData });
            const data = await resp.json();
            if (data.success) {
                AtlasState.original.sequence = data._extracted_sequence || "";
                AtlasState.original.proteinId = data.protein_id;
                AtlasState.baselineEntropy = data.entropy_map;
                displayResults(data, false);
            } else {
                alert(`Error: ${data.error}`);
            }
        } catch (err) {
            alert("Failed to connect to the inference engine.");
        } finally {
            setLoading(false);
        }
    });

    // --- Mutation Lab ---
    document.getElementById("analyzeMutationBtn")?.addEventListener("click", async () => {
        const mutatedSeq = document.getElementById("labSequence")?.value.trim();
        if (!mutatedSeq) { alert("Enter a sequence first."); return; }
        if (!AtlasState.original.sequence) { alert("Upload a baseline sequence first."); return; }
        if (mutatedSeq === AtlasState.original.sequence) {
            alert("Sequence is identical to baseline. Make a mutation!");
            return;
        }

        const formData = new FormData();
        formData.append("sequence", mutatedSeq);
        formData.append("protein_id", AtlasState.original.proteinId);
        formData.append("original_sequence", AtlasState.original.sequence);

        const btn = document.getElementById("analyzeMutationBtn");
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Computing LLR...';

        try {
            const resp = await fetch("/api/predict", { method: "POST", body: formData });
            const data = await resp.json();
            if (data.success) {
                displayResults(data, true);
            } else {
                alert(`Mutation Error: ${data.error}`);
            }
        } catch (err) {
            alert("Mutation analysis failed.");
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-dna"></i> Compute Variant Effect';
        }
    });

    // ============================================================
    // RENDER LOGIC
    // ============================================================

    function setLoading(loading) {
        AtlasState.isLoading = loading;
        loadingSection.style.display = loading ? "block" : "none";
        if (loading) resultsSection.style.display = "none";
        processBtn.disabled = loading;
        processBtn.innerHTML = loading
            ? '<i class="fas fa-spinner fa-spin"></i> Running Inference...'
            : '<i class="fas fa-bolt"></i> Run Zero-Shot Inference';
    }

    function displayResults(data, isMutation) {
        resultsSection.style.display = "block";

        // --- Identity ---
        setTextSafe("sequenceLength", `${data.sequence_length} aa`);
        setTextSafe("proteinId", data.protein_id);

        // --- Naturalness Score ---
        const nat = data.naturalness || 0;
        setTextSafe("naturalnessValue", nat.toFixed(1) + "%");
        const natBar = document.getElementById("naturalnessBar");
        if (natBar) setTimeout(() => { natBar.style.width = nat + "%"; }, 100);

        // --- Performance ---
        setTextSafe("processingTime", `${data.metrics.processing_time.toFixed(3)}s`);

        // --- Structure Actions ---
        renderStructureActions(data.structure);

        // --- Lab Sequence ---
        if (!isMutation) {
            const labSeq = document.getElementById("labSequence");
            if (labSeq) labSeq.value = data._extracted_sequence || "";
        }

        // --- Conservation Profile (always from baseline or current) ---
        const entropyData = isMutation ? AtlasState.baselineEntropy : data.entropy_map;
        if (entropyData) {
            renderEntropyHeatmap(entropyData);
        }

        // --- Variant Effect LLR (only on mutation) ---
        const deltaEl = document.getElementById("deltaContainer");
        if (data.delta_heatmap && data.delta_heatmap.some(v => v !== 0)) {
            renderDeltaHeatmap(data.delta_heatmap);
            if (deltaEl) deltaEl.style.display = "block";
        } else if (deltaEl && !isMutation) {
            deltaEl.style.display = "none";
        }

        // --- Hotspots ---
        renderHotspots(data.hotspots || [], isMutation);
    }

    function renderStructureActions(structure) {
        const container = document.getElementById("structureActions");
        if (!container) return;

        if (!structure) {
            container.innerHTML = `
                <p style="color: var(--text-muted); font-size: var(--text-sm);">
                    <i class="fas fa-search-minus"></i> No experimental or predicted structure found for this ID.
                </p>`;
            return;
        }

        const sourceBadge = structure.source === 'AlphaFold'
            ? `<span style="background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; margin-left: 6px;">AlphaFold</span>`
            : `<span style="background: var(--success); color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; margin-left: 6px;">Experimental</span>`;

        container.innerHTML = `
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <button class="btn btn-primary" onclick="view3D('${esc(structure.id)}', '${esc(structure.pdb_url)}', '${esc(structure.source)}')" style="padding: 8px 16px;">
                    <i class="fas fa-cube"></i> 3D Viewer
                </button>
                ${structure.source === 'RCSB' ? `
                    <button class="btn btn-accent" onclick="viewRCSB('${esc(structure.id)}')" style="padding: 8px 16px;">
                        <i class="fas fa-external-link-alt"></i> RCSB
                    </button>
                ` : ''}
                <span style="font-family: var(--font-heading); color: var(--primary);">${esc(structure.id)}</span>
                ${sourceBadge}
            </div>
        `;
    }

    function renderHotspots(hotspots, isMutation) {
        const el = document.getElementById("hotspotsSummary");
        if (!el) return;

        const critical = hotspots.filter(h => h.critical);
        const conserved = hotspots.length;

        if (!isMutation && conserved > 0) {
            el.style.display = "block";
            el.innerHTML = `
                <strong style="color: var(--accent);"><i class="fas fa-shield-alt"></i> Conservation Summary:</strong>
                <p style="margin-top: 6px;">${conserved} highly conserved residues detected (entropy &lt; 1.5 bits). 
                These positions are evolutionarily constrained and likely essential for function.</p>
            `;
        } else if (isMutation && critical.length > 0) {
            el.style.display = "block";
            const positions = critical.map(h => `<strong>${h.residue}${h.position}</strong> (LLR: ${h.llr.toFixed(2)})`).slice(0, 5);
            el.innerHTML = `
                <strong style="color: var(--danger);"><i class="fas fa-radiation"></i> Critical Mutations Detected!</strong>
                <p style="margin-top: 6px;">You mutated ${critical.length} conserved position(s). 
                These mutations hit evolutionarily constrained residues: ${positions.join(', ')}${critical.length > 5 ? '...' : ''}.</p>
            `;
        } else if (isMutation) {
            el.style.display = "block";
            el.innerHTML = `
                <strong style="color: var(--success);"><i class="fas fa-check-circle"></i> Mutations appear tolerable.</strong>
                <p style="margin-top: 6px;">No conserved hotspots were affected by this mutation set.</p>
            `;
        } else {
            el.style.display = "none";
        }
    }

    // ============================================================
    // PLOTLY RENDERERS
    // ============================================================

    function renderEntropyHeatmap(entropy) {
        if (typeof Plotly === 'undefined' || !entropy) return;
        const x = entropy.map((_, i) => i + 1);

        Plotly.newPlot('entropyContainer', [{
            z: [entropy], x: x, y: ['Conservation'],
            type: 'heatmap',
            colorscale: 'Viridis',
            reversescale: true,  // Low entropy (conserved) = yellow, high = dark
            showscale: true,
            colorbar: { title: 'Entropy (bits)', len: 1 },
            hovertemplate: 'Pos: %{x}<br>Entropy: %{z:.3f} bits<extra></extra>'
        }], {
            margin: { l: 120, r: 40, t: 10, b: 60 },
            xaxis: { title: "Residue Position" },
            yaxis: { showticklabels: true },
            height: 240
        }, { responsive: true, displaylogo: false });
    }

    function renderDeltaHeatmap(delta) {
        if (typeof Plotly === 'undefined' || !delta) return;
        const x = delta.map((_, i) => i + 1);

        // Find symmetric range for RdBu centering at 0
        const absMax = Math.max(...delta.map(Math.abs), 0.1);

        Plotly.newPlot('deltaPlot', [{
            z: [delta], x: x, y: ['Variant Effect'],
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            zmin: -absMax,
            zmax: absMax,
            showscale: true,
            colorbar: { title: 'LLR', len: 1 },
            hovertemplate: 'Pos: %{x}<br>LLR: %{z:.4f}<extra></extra>'
        }], {
            margin: { l: 120, r: 40, t: 10, b: 60 },
            xaxis: { title: "Residue Position" },
            yaxis: { showticklabels: true },
            height: 240,
            annotations: [{
                text: 'Red = Destabilizing | Blue = Tolerated',
                showarrow: false, x: 0.5, y: -0.4,
                xref: 'paper', yref: 'paper',
                font: { size: 10, color: '#999' }
            }]
        }, { responsive: true, displaylogo: false });
    }

    // --- Utilities ---
    function setTextSafe(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    function esc(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
});