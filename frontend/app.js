/**
 * Legal Speech RAG — Frontend Application Logic
 */

// Configuration
const DEFAULT_API_URL = "https://varshithathi2006-legal-speech-api.hf.space";
let API_BASE_URL = localStorage.getItem("legal_rag_api_url") || (
    window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
        ? "http://localhost:7860" 
        : DEFAULT_API_URL
);

// State
let legalDomainsHierarchy = {};
let selectedVoice = "Female";

// DOM Elements
const domainSelect = document.getElementById("domain-select");
const subdomainSelect = document.getElementById("subdomain-select");
const activeLawText = document.getElementById("active-law");
const questionInput = document.getElementById("question-input");
const runBtn = document.getElementById("run-btn");
const loadingSpinner = document.getElementById("loading-spinner");
const resultsContainer = document.getElementById("results-container");
const answerText = document.getElementById("answer-text");
const audioPlayer = document.getElementById("audio-player");
const citationsList = document.getElementById("citations-list");
const latencyBadge = document.getElementById("latency-badge");
const voiceName = document.getElementById("voice-name");
const metricsGrid = document.getElementById("metrics-grid");
const apiStatusText = document.getElementById("api-status-text");

// Settings Modal Elements
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const closeSettings = document.getElementById("close-settings");
const saveSettings = document.getElementById("save-settings");
const apiUrlInput = document.getElementById("api-url-input");

// ──────────────────────────────────────────────────────────────────────
// Initialization
// ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    apiUrlInput.value = API_BASE_URL;
    setupEventListeners();
    fetchDomains();
    checkApiHealth();
});

function setupEventListeners() {
    // Domain Change
    domainSelect.addEventListener("change", (e) => {
        populateSubdomains(e.target.value);
    });

    // Subdomain Change
    subdomainSelect.addEventListener("change", () => {
        updateActiveLaw();
    });

    // Voice Selection
    document.querySelectorAll(".radio-card").forEach(card => {
        card.addEventListener("click", () => {
            document.querySelectorAll(".radio-card").forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            selectedVoice = card.dataset.voice;
            card.querySelector("input").checked = true;
        });
    });

    // Preset Chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const domain = chip.dataset.domain;
            const subdomain = chip.dataset.subdomain;
            const query = chip.dataset.query;

            domainSelect.value = domain;
            populateSubdomains(domain, subdomain);
            questionInput.value = query;
            questionInput.focus();
        });
    });

    // Submit Query
    runBtn.addEventListener("click", executeLegalQuery);
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            executeLegalQuery();
        }
    });

    // Settings Modal
    settingsBtn.addEventListener("click", () => settingsModal.classList.remove("hidden"));
    closeSettings.addEventListener("click", () => settingsModal.classList.add("hidden"));
    saveSettings.addEventListener("click", () => {
        let val = apiUrlInput.value.trim().replace(/\/$/, "");
        if (val) {
            API_BASE_URL = val;
            localStorage.setItem("legal_rag_api_url", val);
            settingsModal.classList.add("hidden");
            checkApiHealth();
            fetchDomains();
        }
    });
}

// ──────────────────────────────────────────────────────────────────────
// API Calls
// ──────────────────────────────────────────────────────────────────────

async function checkApiHealth() {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            apiStatusText.textContent = `Online (${data.indexed_chunks} chunks)`;
            apiStatusText.style.color = "var(--accent-emerald)";
        } else {
            throw new Error("Health check failed");
        }
    } catch (e) {
        apiStatusText.textContent = "Backend Offline";
        apiStatusText.style.color = "var(--accent-amber)";
    }
}

async function fetchDomains() {
    try {
        const res = await fetch(`${API_BASE_URL}/api/domains`);
        if (res.ok) {
            const data = await res.json();
            legalDomainsHierarchy = {};
            domainSelect.innerHTML = "";
            
            data.domains.forEach(d => {
                legalDomainsHierarchy[d.domain_name] = d.subdomains;
                const opt = document.createElement("option");
                opt.value = d.domain_name;
                opt.textContent = d.domain_name;
                domainSelect.appendChild(opt);
            });

            if (data.domains.length > 0) {
                populateSubdomains(data.domains[0].domain_name);
            }
        }
    } catch (err) {
        console.warn("Using fallback domain hierarchy:", err);
    }
}

function populateSubdomains(domainName, selectedSubdomain = null) {
    subdomainSelect.innerHTML = "";
    const subList = legalDomainsHierarchy[domainName] || [];

    subList.forEach(sub => {
        const opt = document.createElement("option");
        opt.value = sub.subdomain_name;
        opt.textContent = sub.subdomain_name;
        opt.dataset.law = sub.applicable_law;
        subdomainSelect.appendChild(opt);
    });

    if (selectedSubdomain) {
        subdomainSelect.value = selectedSubdomain;
    }

    updateActiveLaw();
}

function updateActiveLaw() {
    const selectedOpt = subdomainSelect.options[subdomainSelect.selectedIndex];
    if (selectedOpt && selectedOpt.dataset.law) {
        activeLawText.textContent = selectedOpt.dataset.law;
    } else {
        activeLawText.textContent = "Applicable Statutory Indian Legislation";
    }
}

async function executeLegalQuery() {
    const query = questionInput.value.trim();
    if (!query) {
        questionInput.focus();
        return;
    }

    const domain = domainSelect.value;
    const subdomain = subdomainSelect.value;

    // UI Loading State
    runBtn.disabled = true;
    loadingSpinner.classList.remove("hidden");
    resultsContainer.classList.add("hidden");

    try {
        const payload = {
            query_text: query,
            domain: domain,
            subdomain: subdomain,
            voice_gender: selectedVoice
        };

        const res = await fetch(`${API_BASE_URL}/api/query`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Query processing failed");
        }

        const data = await res.json();
        renderResults(data);

    } catch (err) {
        alert(`Error: ${err.message}. Please check API connection in Settings.`);
    } finally {
        runBtn.disabled = false;
        loadingSpinner.classList.add("hidden");
    }
}

function renderResults(data) {
    // 1. Legal Answer
    const formattedAnswer = data.answer
        .split("\n\n")
        .map(para => `<p>${escapeHtml(para)}</p>`)
        .join("");
    answerText.innerHTML = formattedAnswer;

    // 2. Latency & Voice
    latencyBadge.textContent = `${data.metrics?.total_latency_seconds || 0.8}s`;
    voiceName.textContent = selectedVoice === "Female" ? "en-IN-NeerjaNeural" : "en-IN-PrabhatNeural";

    // 3. Audio Player
    if (data.audio_url) {
        const fullAudioUrl = data.audio_url.startsWith("http") 
            ? data.audio_url 
            : `${API_BASE_URL}${data.audio_url}`;
        audioPlayer.src = fullAudioUrl;
        audioPlayer.load();
    }

    // 4. Citations
    const rawCitations = data.citations || "";
    const citationLines = rawCitations.split("\n").filter(l => l.trim().length > 0);
    
    if (citationLines.length > 0) {
        citationsList.innerHTML = `<ul>${citationLines.map(c => `<li>${escapeHtml(c.replace(/^-\s*/, ''))}</li>`).join('')}</ul>`;
    } else {
        citationsList.innerHTML = `<p style="color: var(--text-muted); font-size: 0.85rem;">Statute & Case citations recorded in knowledge base.</p>`;
    }

    // 5. Metrics Grid
    if (data.metrics) {
        const m = data.metrics;
        metricsGrid.innerHTML = `
            <div class="metric-item">
                <div class="metric-value">${(m.context_precision * 100).toFixed(1)}%</div>
                <div class="metric-label">Context Precision</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.max_cosine_similarity.toFixed(4)}</div>
                <div class="metric-label">Top Similarity Score</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.exact_match_chunks || 0}</div>
                <div class="metric-label">Exact Section Matches</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.response_word_count || 0}</div>
                <div class="metric-label">Words Synthesized</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.total_latency_seconds}s</div>
                <div class="metric-label">Pipeline Latency</div>
            </div>
        `;
    }

    resultsContainer.classList.remove("hidden");
    lucide.createIcons();
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
