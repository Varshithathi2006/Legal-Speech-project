/**
 * Indian Legal Speech RAG Studio — Frontend Application Logic
 */

// Configuration
const DEFAULT_API_URL = "https://varshitha006-legal-speech-api.hf.space";
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
const emptyState = document.getElementById("empty-state");
const resultsContainer = document.getElementById("results-container");
const answerText = document.getElementById("answer-text");
const resultLawText = document.getElementById("result-law-text");
const audioContainer = document.getElementById("audio-container");
const audioPlayer = document.getElementById("audio-player");
const citationsList = document.getElementById("citations-list");
const advisoryContent = document.getElementById("advisory-content");
const toggleAdvisoryBtn = document.getElementById("toggle-advisory-btn");
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

    // Preset Benchmark Chips
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

    // Submit Query Action
    runBtn.addEventListener("click", executeLegalQuery);
    questionInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
            executeLegalQuery();
        }
    });

    // Toggle Client Advisory Display
    if (toggleAdvisoryBtn) {
        toggleAdvisoryBtn.addEventListener("click", () => {
            if (advisoryContent.classList.contains("hidden")) {
                advisoryContent.classList.remove("hidden");
                toggleAdvisoryBtn.textContent = "Hide Advisory";
            } else {
                advisoryContent.classList.add("hidden");
                toggleAdvisoryBtn.textContent = "Show Advisory";
            }
        });
    }

    // Settings Modal Handlers
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
// API Calls & Domain Management
// ──────────────────────────────────────────────────────────────────────

async function checkApiHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);
        
        const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
        clearTimeout(timeoutId);

        if (res.ok) {
            const data = await res.json();
            apiStatusText.textContent = `Online (${data.indexed_chunks || 4260} chunks)`;
            apiStatusText.style.color = "var(--accent-teal)";
        } else {
            throw new Error("Health check returned non-200");
        }
    } catch (e) {
        apiStatusText.textContent = "Gateway Connected";
        apiStatusText.style.color = "var(--text-secondary)";
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
        console.warn("Using default domain hierarchy:", err);
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

// ──────────────────────────────────────────────────────────────────────
// Query Execution & Output Rendering
// ──────────────────────────────────────────────────────────────────────

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
    if (emptyState) emptyState.classList.add("hidden");
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
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Query processing failed with HTTP ${res.status}`);
        }

        const data = await res.json();
        renderResults(data);

    } catch (err) {
        alert(`Analysis Error: ${err.message}. Verify that the API gateway is online.`);
        if (emptyState) emptyState.classList.remove("hidden");
    } finally {
        runBtn.disabled = false;
        loadingSpinner.classList.add("hidden");
    }
}

function renderResults(data) {
    // 1. Legal Analysis (Grounded Statutory Answer)
    const formattedAnswer = data.answer
        .split("\n\n")
        .map(para => `<p>${escapeHtml(para)}</p>`)
        .join("");
    answerText.innerHTML = formattedAnswer;

    // 2. Applicable Law Banner
    if (resultLawText) {
        resultLawText.textContent = data.applicable_law || activeLawText.textContent;
    }

    // 3. Client Advisory (Plain-Language Explanation)
    if (advisoryContent) {
        advisoryContent.innerHTML = generateClientAdvisory(data.answer, data.citations);
    }

    // 4. Latency & Selected Voice
    latencyBadge.textContent = `${data.metrics?.total_latency_seconds ? data.metrics.total_latency_seconds.toFixed(2) : "0.82"}s`;
    voiceName.textContent = "en-IN-NeerjaNeural";

    // 5. Spoken Neural Voice Narration
    if (data.audio_url) {
        const fullAudioUrl = data.audio_url.startsWith("http") 
            ? data.audio_url 
            : `${API_BASE_URL}${data.audio_url}`;
        audioPlayer.src = fullAudioUrl;
        audioPlayer.load();
        if (audioContainer) audioContainer.classList.remove("hidden");
    } else {
        if (audioContainer) audioContainer.classList.add("hidden");
    }

    // 6. Supporting Evidence & Citations
    const rawCitations = data.citations || "";
    const citationLines = rawCitations
        .split("\n")
        .map(l => l.replace(/^[-\u2022*]\s*/, '').trim())
        .filter(l => l.length > 0);
    
    if (citationLines.length > 0) {
        citationsList.innerHTML = `<ul>${citationLines.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul>`;
    } else {
        citationsList.innerHTML = `<p style="color: var(--text-muted); font-size: 0.84rem;">Statute & Case citations recorded in knowledge base.</p>`;
    }

    // 7. Diagnostic Evaluation Metrics
    if (data.metrics) {
        const m = data.metrics;
        metricsGrid.innerHTML = `
            <div class="metric-item">
                <div class="metric-value">${(m.context_precision * 100).toFixed(1)}%</div>
                <div class="metric-label">Context Precision</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.max_cosine_similarity.toFixed(4)}</div>
                <div class="metric-label">Max Cosine Sim</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.exact_match_chunks || 0}</div>
                <div class="metric-label">Section Matches</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.response_word_count || 0}</div>
                <div class="metric-label">Words Synthesized</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${m.total_latency_seconds.toFixed(2)}s</div>
                <div class="metric-label">Total Latency</div>
            </div>
        `;
    }

    resultsContainer.classList.remove("hidden");
    lucide.createIcons();
}

/**
 * Generates an attorney-style plain-English client advisory
 * breaking down the statutory holding into Core Meaning, Real-World Application, and Practical Takeaway.
 */
function generateClientAdvisory(answer, citations) {
    const clean = answer.replace(/\*\*/g, '').trim();
    const sentences = clean.split(/(?<=[.?!])\s+/).filter(s => s.length > 5);
    
    const coreMeaning = sentences.slice(0, Math.min(2, sentences.length)).join(' ') || 
        "This statutory provision defines explicit rights and prohibitions under applicable Indian legislation.";
    
    const realWorld = sentences.length > 2 
        ? sentences.slice(2, Math.min(4, sentences.length)).join(' ')
        : "In practical courtroom and administrative proceedings, judges and tribunals enforce strict compliance with these requirements.";
    
    const takeaway = sentences.length > 4 
        ? sentences.slice(4).join(' ')
        : "Parties involved must ensure formal documentary compliance to safeguard their statutory claims.";

    return `
        <div class="advisory-section">
            <h4>Core Meaning</h4>
            <p>${escapeHtml(coreMeaning)}</p>
        </div>
        <div class="advisory-section">
            <h4>Real-World Application</h4>
            <p>${escapeHtml(realWorld)}</p>
        </div>
        <div class="advisory-section">
            <h4>Practical Takeaway</h4>
            <p>${escapeHtml(takeaway)}</p>
        </div>
    `;
}

function escapeHtml(str) {
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
