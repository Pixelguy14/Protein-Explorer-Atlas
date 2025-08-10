document.addEventListener("DOMContentLoaded", () => {
    // --- Form and page elements ---
    const form = document.getElementById("proteinForm");
    const fileInput = document.getElementById("proteinFile");
    const processBtn = document.getElementById("processBtn");
    const loadingSection = document.getElementById("loadingSection");
    const resultsSection = document.getElementById("resultsSection");
    
    // --- Modal elements ---
    const modal = document.getElementById("infoModal");
    const modalTitle = document.getElementById("modalTitle");
    const modalIframe = document.getElementById("modalIframe");
    const closeModalBtn = document.querySelector(".modal-close-btn");

    // --- Event Listeners ---

    // Handle form submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const formData = new FormData();
        const file = fileInput.files[0];

        if (!file) {
            alert("Por favor selecciona un archivo");
            return;
        }

        formData.append("protein_file", file);

        loadingSection.style.display = "block";
        resultsSection.style.display = "none";
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';

        try {
            const response = await fetch("/predict", { method: "POST", body: formData });
            const data = await response.json();
            if (data.success) {
                displayResults(data);
            } else {
                throw new Error(data.error || "Error desconocido");
            }
        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            loadingSection.style.display = "none";
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fas fa-cogs"></i> Procesar Proteína';
        }
    });

    // Close modal events
    closeModalBtn.onclick = () => closeModal();
    modal.onclick = (event) => {
        if (event.target == modal) { // Click on overlay
            closeModal();
        }
    };

    // --- Functions ---

    function displayResults(data) {
        resultsSection.style.display = "block";
        document.getElementById("sequenceLength").textContent = `${data.sequence_length} aminoácidos`;
        document.getElementById("processTime").textContent = new Date(data.timestamp).toLocaleString();

        const tableBody = document.querySelector("#familiesTable tbody");
        tableBody.innerHTML = "";

        data.families.forEach((family) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${family.family}</strong></td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${family.confidence * 100}%"></div>
                    </div>
                    <small>${(family.confidence * 100).toFixed(1)}%</small>
                </td>
                <td><code>${family.pdb_id}</code></td>
                <td>${family.description}</td>
                <td>
                    <button class="view-btn" onclick="viewPDB('${family.pdb_id}')">
                        <i class="fas fa-external-link-alt"></i> RCSB
                    </button>
                    <button class="view-btn" onclick="showModal('Gráfico de Dominios', '/plot/${data.protein_id}')" style="background: #ff5722; margin-left: 5px;">
                        <i class="fas fa-puzzle-piece"></i> Dominios
                    </button>
                    <button class="view-btn" onclick="showModal('Visor 3D - ${family.pdb_id}', '/view3d/${family.pdb_id}')" style="background: #9c27b0; margin-left: 5px;">
                        <i class="fas fa-cube"></i> 3D
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });

        updateMetrics(data.metrics);
        document.getElementById("processingTime").textContent = `${data.metrics.processing_time.toFixed(2)}s`;
    }

    function updateMetrics(metrics) {
        const metricTypes = ["accuracy", "precision", "recall", "f1_score"];
        metricTypes.forEach((type) => {
            const value = metrics[type];
            const key = type === "f1_score" ? "f1" : type;
            const valueElement = document.getElementById(`${key}Value`);
            const barElement = document.getElementById(`${key}Bar`);
            if (valueElement && barElement) {
                valueElement.textContent = (value * 100).toFixed(1) + "%";
                setTimeout(() => { barElement.style.width = value * 100 + "%"; }, 100);
            }
        });
    }
    
    function closeModal() {
        modal.style.display = "none";
        modalIframe.src = "about:blank"; // Stop content from running
    }

    // --- Drag and Drop (unchanged) ---
    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) {
            document.querySelector(".file-input-label span").textContent = `Archivo: ${file.name}`;
        }
    });
});

// --- Global Functions ---

function showModal(title, url) {
    const modal = document.getElementById("infoModal");
    const modalTitle = document.getElementById("modalTitle");
    const modalIframe = document.getElementById("modalIframe");

    modalTitle.textContent = title;
    modalIframe.src = url;
    modal.style.display = "block";
}

function viewPDB(pdbId) {
    // Open PDB viewer in new tab
    const url = `https://www.rcsb.org/structure/${pdbId}`;
    window.open(url, "_blank");
}