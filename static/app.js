document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("config-form");
    const evaluateBtn = document.getElementById("evaluate-btn");
    const btnSpinner = document.getElementById("btn-spinner");
    const btnLabel = document.getElementById("btn-label");
    const statsBody = document.getElementById("stats-body");
    const statusPill = document.getElementById("status-pill");
    const statusText = document.getElementById("status-text");
    const chartPlaceholder = document.getElementById("chart-placeholder");
    const schemesContainer = document.getElementById("schemes-container");
    const addSchemeBtn = document.getElementById("add-scheme-btn");
    const shareBtn = document.getElementById("share-btn");
    const downloadCsvBtn = document.getElementById("download-csv-btn");
    const downloadCdfBtn = document.getElementById("download-cdf-btn");
    const downloadStatsCsvBtn = document.getElementById("download-stats-csv-btn");
    const biasedLog2Panel = document.getElementById("biased-log2-panel");
    const cdfPanel = document.getElementById("cdf-panel");
    const toast = document.getElementById("toast");

    let currentAppVersion = null;
    let loadedStateVersion = null;

    fetch('/api/version')
        .then(res => res.json())
        .then(data => {
            currentAppVersion = data.version;
            const versionEl = document.getElementById('app-version');
            if (versionEl) versionEl.textContent = data.version;

            if (loadedStateVersion && loadedStateVersion !== currentAppVersion) {
                console.warn(`Warning: State was saved with version ${loadedStateVersion}, but current version is ${currentAppVersion}`);
                setTimeout(() => {
                    showToast(`Warning: State saved on v${loadedStateVersion}, current app v${currentAppVersion}.`);
                }, 2000);
            }
        })
        .catch(err => console.error("Error fetching version:", err));

    fetch('/api/config')
        .then(res => res.json())
        .then(data => {
            const nInput = document.getElementById('cfg-n');
            if (nInput && data.max_n) {
                nInput.max = data.max_n;
                // update title so user knows the max
                nInput.title = `Maximum allowed is ${data.max_n}`;
            }
        })
        .catch(err => console.error("Error fetching config:", err));

    let lastEvalData = null;
    let lastEvalN = 0;
    let lastMinLog2 = -23;
    let lastExactY = -1;


    const groupK = document.getElementById("group-k");
    const groupDist = document.getElementById("group-dist");

    // Per-vector distribution selects
    const aDistSelect = document.getElementById("cfg-a-dist");
    const bDistSelect = document.getElementById("cfg-b-dist");

    /** Update parameter labels based on distribution type */
    function updateDistLabels(prefix, distType) {
        const lblAvg = document.getElementById(`lbl-${prefix}-avg`);
        const lblSigma = document.getElementById(`lbl-${prefix}-sigma`);
        if (!lblAvg || !lblSigma) return;
        switch (distType) {
            case "uniform":
                lblAvg.textContent = "Center";
                lblSigma.textContent = "Half-width";
                break;
            case "lognormal":
                lblAvg.textContent = "μ (ln)";
                lblSigma.textContent = "σ (ln)";
                break;
            default: // gaussian
                lblAvg.textContent = "Mean (μ)";
                lblSigma.textContent = "StdDev (σ)";
                break;
        }
    }

    if (aDistSelect) {
        aDistSelect.addEventListener("change", () => updateDistLabels("a", aDistSelect.value));
    }
    if (bDistSelect) {
        bDistSelect.addEventListener("change", () => updateDistLabels("b", bDistSelect.value));
    }



    let schemeCounter = 0;

    // ── Scheme type definitions ──────────────────────────────────
    // Each type has a label, description, and a list of configurable parameters.
    const SCHEME_TYPES = {
        correctly_rounded: {
            label: "Correctly Rounded Dot Product",
            desc: "No intermediate rounding. Exact accumulation, single final round.",
            badge: "CR",
            params: [
                { key: "resPrec", label: "Final Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp32" },
            ],
        },
        approx_mult: {
            label: "FP MUL + Exact Acc",
            desc: "Products rounded via FP MUL, then exactly accumulated before final rounding.",
            params: [
                { key: "multPrec", label: "Mult Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp16" },
                { key: "resPrec", label: "Result Prec", type: "select", options: ["fp32", "fp64"], default: "fp32" },
            ],
        },
        approx_mult_acc: {
            label: "FP MUL + FP Add Tree",
            desc: "Products rounded, then accumulated via binary tree of rounded additions.",
            params: [
                { key: "multPrec", label: "Mult Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp16" },
                { key: "addPrec", label: "Add Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp16" },
                { key: "resPrec", label: "Result Prec", type: "select", options: ["fp32", "fp64"], default: "fp32" },
            ],
        },
        approx_mult_linear_acc: {
            label: "FP MUL + Linear FP Add",
            desc: "Products rounded, then accumulated sequentially (linear chain) with rounded additions.",
            params: [
                { key: "multPrec", label: "Mult Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp16" },
                { key: "addPrec", label: "Add Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp16" },
                { key: "resPrec", label: "Result Prec", type: "select", options: ["fp32", "fp64"], default: "fp32" },
            ],
        },
        fma: {
            label: "Sequence of FMA",
            desc: "Fused multiply-add: one rounding per step (res = round(res + a×b)).",
            params: [
                { key: "fmaPrec", label: "FMA Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp32" },
                { key: "resPrec", label: "Result Prec", type: "select", options: ["bf16", "fp16", "fp32", "fp64"], default: "fp32" },
            ],
        },
        bulk_norm: {
            label: "Bulk Normalization",
            desc: "Products rounded to fixed-point via round-to-odd, exponent from max product.",
            params: [
                { key: "bulkNormPrec", label: "Bulk Prec (bits)", type: "number", default: 24, min: 5, max: 60 },
                { key: "finalPrec", label: "Final Prec (bits)", type: "number", default: 23, min: 5, max: 60 },
            ],
        },
    };

    // ── Plotly dark layout template ────────────────────────────────
    const PLOTLY_LAYOUT = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "'Inter', sans-serif", color: "#8b9ec1", size: 12 },
        margin: { t: 24, r: 24, b: 180, l: 60 },
        xaxis: {
            title: { text: "Sorted sample index", standoff: 10 },
            gridcolor: "rgba(255,255,255,0.05)",
            zerolinecolor: "rgba(255,255,255,0.08)",
            linecolor: "rgba(255,255,255,0.08)",
        },
        yaxis: {
            title: { text: "Relative error", standoff: 10 },
            type: "log",
            gridcolor: "rgba(255,255,255,0.05)",
            zerolinecolor: "rgba(255,255,255,0.08)",
            linecolor: "rgba(255,255,255,0.08)",
        },
        legend: {
            orientation: "h",
            yanchor: "top", y: -0.15,
            xanchor: "center", x: 0.5,
            font: { size: 11 },
            bgcolor: "rgba(0,0,0,0)",
        },
        hoverlabel: {
            bgcolor: "#0f172a",
            bordercolor: "rgba(255,255,255,0.15)",
            font: { color: "#edf2f7", family: "'JetBrains Mono', monospace", size: 12 },
        },
        hovermode: "x unified",
    };

    const PLOTLY_CONFIG = {
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
    };

    // ── Curated colour palette ────────────────────────────────────
    const SCHEME_COLORS = [
        "#f87171", "#60a5fa", "#34d399", "#a78bfa", "#fbbf24",
        "#f472b6", "#38bdf8", "#fb923c", "#4ade80", "#c084fc",
        "#e879f9", "#22d3ee", "#facc15", "#a3e635", "#fb7185",
    ];

    // ═══════════════════════════════════════════════════════════════
    //  Dynamic scheme card management
    // ═══════════════════════════════════════════════════════════════

    /** Build the parameter options HTML for a given scheme type key */
    function buildParamsHTML(typeKey, initialParams = {}) {
        const typeDef = SCHEME_TYPES[typeKey];
        if (!typeDef || typeDef.params.length === 0) return "";

        let html = '<div class="scheme-options">';
        for (const p of typeDef.params) {
            html += '<div class="input-group input-sm">';
            html += `<label>${p.label}</label>`;
            const val = initialParams[p.key] !== undefined ? initialParams[p.key] : p.default;
            if (p.type === "select") {
                html += `<select data-param="${p.key}" class="scheme-param">`;
                for (const opt of p.options) {
                    const sel = opt == val ? " selected" : "";
                    html += `<option value="${opt}"${sel}>${opt.toUpperCase()}</option>`;
                }
                html += "</select>";
            } else if (p.type === "number") {
                html += `<input type="number" data-param="${p.key}" class="scheme-param" value="${val}" min="${p.min}" max="${p.max}">`;
            }
            html += "</div>";
        }
        html += "</div>";
        return html;
    }

    /** Create and append a new scheme card */
    function addSchemeCard(initialType = "correctly_rounded", initialParams = {}, isActive = true) {
        const id = schemeCounter++;
        const card = document.createElement("div");
        card.className = "scheme-card" + (isActive ? " active" : "");
        card.dataset.schemeId = id;

        const typeDef = SCHEME_TYPES[initialType];
        const badgeHTML = typeDef.badge
            ? `<span class="scheme-badge badge-reference">${typeDef.badge}</span>`
            : "";

        card.innerHTML = `
            <div class="scheme-header">
                <div class="scheme-header-left">
                    <label class="toggle-label">
                        <input type="checkbox" class="scheme-toggle" ${isActive ? "checked" : ""}>
                        <span class="toggle-track"><span class="toggle-thumb"></span></span>
                    </label>
                    <div class="scheme-title-group">
                        <select class="scheme-type-select">
                            ${Object.entries(SCHEME_TYPES)
                                .map(([key, def]) => `<option value="${key}"${key === initialType ? " selected" : ""}>${def.label}</option>`)
                                .join("")}
                        </select>
                        <input type="text" class="scheme-name-input" placeholder="Custom name (auto)" value="${initialParams.customName || ''}" aria-label="Custom scheme name">
                    </div>
                </div>
                <div class="scheme-header-right">
                    ${badgeHTML}
                    <button type="button" class="scheme-action-btn move-up-btn" title="Move up" aria-label="Move up">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 15l-6-6-6 6"/></svg>
                    </button>
                    <button type="button" class="scheme-action-btn move-down-btn" title="Move down" aria-label="Move down">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
                    </button>
                    <button type="button" class="scheme-action-btn remove-scheme-btn" title="Remove scheme" aria-label="Remove scheme">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                </div>
            </div>
            <p class="scheme-desc">${typeDef.desc}</p>
            <div class="scheme-params-slot">
                ${buildParamsHTML(initialType, initialParams)}
            </div>
        `;

        // Wire up type selector
        const typeSelect = card.querySelector(".scheme-type-select");
        typeSelect.addEventListener("change", () => {
            const newType = typeSelect.value;
            const newDef = SCHEME_TYPES[newType];
            card.querySelector(".scheme-desc").textContent = newDef.desc;
            card.querySelector(".scheme-params-slot").innerHTML = buildParamsHTML(newType);

            // Update badge
            const headerRight = card.querySelector(".scheme-header-right");
            const existingBadge = headerRight.querySelector(".scheme-badge");
            if (existingBadge) existingBadge.remove();
            if (newDef.badge) {
                const badge = document.createElement("span");
                badge.className = "scheme-badge badge-reference";
                badge.textContent = newDef.badge;
                headerRight.insertBefore(badge, headerRight.firstChild);
            }
        });

        // Wire up toggle
        const toggle = card.querySelector(".scheme-toggle");
        toggle.addEventListener("change", () => {
            card.classList.toggle("active", toggle.checked);
        });

        // Wire up remove button
        const removeBtn = card.querySelector(".remove-scheme-btn");
        removeBtn.addEventListener("click", () => {
            card.style.opacity = "0";
            card.style.transform = "scale(0.95)";
            setTimeout(() => card.remove(), 200);
        });

        // Wire up move up button
        const moveUpBtn = card.querySelector(".move-up-btn");
        moveUpBtn.addEventListener("click", () => {
            const prev = card.previousElementSibling;
            if (prev) {
                card.parentNode.insertBefore(card, prev);
            }
        });

        // Wire up move down button
        const moveDownBtn = card.querySelector(".move-down-btn");
        moveDownBtn.addEventListener("click", () => {
            const next = card.nextElementSibling;
            if (next) {
                card.parentNode.insertBefore(next, card);
            }
        });

        schemesContainer.appendChild(card);

        // Animate entrance
        card.style.animation = "fadeSlideIn 0.25s ease";
        card.addEventListener("animationend", () => { card.style.animation = ""; }, { once: true });

        return card;
    }

    // "Add Scheme" button
    addSchemeBtn.addEventListener("click", () => {
        addSchemeCard("bulk_norm");
    });

    const genSeedBtn = document.getElementById("gen-seed-btn");
    if (genSeedBtn) {
        genSeedBtn.addEventListener("click", () => {
            const seedInput = document.getElementById("cfg-seed");
            if (seedInput) {
                seedInput.value = Math.floor(Math.random() * 2147483647);
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════
    //  State sharing & restoring
    // ═══════════════════════════════════════════════════════════════

    function showToast(msg) {
        toast.textContent = msg;
        toast.classList.add("visible");
        setTimeout(() => toast.classList.remove("visible"), 2500);
    }

    shareBtn.addEventListener("click", () => {
        const state = buildRequest();
        const json = JSON.stringify(state);
        const b64 = btoa(encodeURIComponent(json));
        const url = new URL(window.location.href);
        url.hash = "state=" + b64;
        
        navigator.clipboard.writeText(url.toString()).then(() => {
            showToast("URL copied to clipboard!");
            window.history.replaceState(null, "", url.toString());
        }).catch(() => {
            showToast("Failed to copy URL");
        });
    });

    // ── CSV download ──────────────────────────────────────────────
    downloadCsvBtn.addEventListener("click", () => {
        if (!lastEvalData) return;

        const entries = Object.entries(lastEvalData).sort(
            ([, a], [, b]) => a.geometric_mean - b.geometric_mean
        );
        const schemeNames = entries.map(([name]) => name);

        // --- Part 1: Summary stats ---
        let csv = "Summary Statistics\n";
        csv += "Scheme,Min Error,Max Error,Geometric Mean,Exact Hits,Exact Hit %\n";
        for (const [name, r] of entries) {
            const pct = (r.exact_count / lastEvalN * 100).toFixed(1);
            csv += `"${name}",${r.min},${r.max},${r.geometric_mean},${r.exact_count},${pct}\n`;
        }

        // --- Part 2: Per-sample sorted relative errors (offset log2 scale) ---
        // Linear mapping: 0 = exact, 1 = min error (most negative log2), 10 = max error (largest log2)
        let maxLog2 = -Infinity;
        for (const [, r] of entries) {
            for (const v of r.sorted_rel_errors) {
                if (v > 0) {
                    const l = Math.log2(v);
                    if (l > maxLog2) maxLog2 = l;
                }
            }
        }
        const minLog2 = lastMinLog2;
        const log2Range = maxLog2 - minLog2;

        csv += "\nSorted Relative Errors (offset log2: 0=exact, 1=min error, 10=max error)\n";
        csv += "Index," + schemeNames.map(n => `"${n}"`).join(",") + "\n";

        const maxLen = Math.max(...entries.map(([, r]) => r.sorted_rel_errors.length));
        for (let i = 0; i < maxLen; i++) {
            csv += i;
            for (const [, r] of entries) {
                if (i < r.sorted_rel_errors.length) {
                    const v = r.sorted_rel_errors[i];
                    let scaled;
                    if (v === 0) {
                        scaled = lastExactY;
                    } else if (log2Range === 0) {
                        scaled = 0; // all non-zero errors are the same
                    } else {
                        scaled = (Math.log2(v) - minLog2);
                    }
                    csv += "," + scaled;
                } else {
                    csv += ",";
                }
            }
            csv += "\n";
        }

        // Trigger download
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "rvbna_evaluation.csv";
        a.click();
        URL.revokeObjectURL(url);

        showToast("CSV downloaded!");
    });

    // ── CDF CSV download ─────────────────────────────────────────
    downloadCdfBtn.addEventListener("click", () => {
        if (!lastEvalData) return;

        const entries = Object.entries(lastEvalData).sort(
            ([, a], [, b]) => a.geometric_mean - b.geometric_mean
        );
        const schemeNames = entries.map(([name]) => name);

        // Find global min log2 for bias (same as error CSV)
        // Use global precision-based minLog2
        const minLog2 = lastMinLog2;

        // For each scheme, compute biased log2 values and build CDF
        // Collect all unique biased log2 values across schemes for a shared x-axis
        const allBiasedValues = new Set();
        const schemeData = [];

        for (const [, r] of entries) {
            const biased = [];
            for (const v of r.sorted_rel_errors) {
                if (v === 0) {
                    biased.push(lastExactY);
                } else {
                    biased.push(Math.log2(v) - minLog2);
                }
            }
            biased.sort((a, b) => a - b);
            for (const b of biased) allBiasedValues.add(b);
            schemeData.push(biased);
        }

        // Sorted unique x-values
        const xValues = [...allBiasedValues].sort((a, b) => a - b);

        // Build CSV: for each x-value, compute CDF for each scheme
        let csv = "Cumulative Distribution of Biased Log2 Relative Error\n";
        csv += "Biased Log2," + schemeNames.map(n => `"${n}"`).join(",") + "\n";

        for (const x of xValues) {
            csv += x;
            for (const biased of schemeData) {
                // CDF = fraction of values <= x
                // Binary search for upper bound
                let lo = 0, hi = biased.length;
                while (lo < hi) {
                    const mid = (lo + hi) >>> 1;
                    if (biased[mid] <= x) lo = mid + 1;
                    else hi = mid;
                }
                csv += "," + (lo / biased.length);
            }
            csv += "\n";
        }

        const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "rvbna_cdf.csv";
        a.click();
        URL.revokeObjectURL(url);

        showToast("CDF CSV downloaded!");
    });

    // ── Statistics CSV download ──────────────────────────────────
    if (downloadStatsCsvBtn) {
        downloadStatsCsvBtn.addEventListener("click", () => {
            if (!lastEvalData) return;

            const entries = Object.entries(lastEvalData).sort(
                ([, a], [, b]) => a.geometric_mean - b.geometric_mean
            );

            let csv = "Scheme,Min Error,Max Error,Geometric Mean,Exact Hits,Avg Signed Rel. Error,Avg Signed Error,Sum Signed Rel. Error,Sum Signed Error,Pos Errors,Neg Errors,Exact Pos,Exact Neg,Opp Sign\n";

            for (const [schemeName, results] of entries) {
                csv += `"${schemeName}",`;
                csv += `${results.min_error},`;
                csv += `${results.max_error},`;
                csv += `${results.geometric_mean},`;
                csv += `${results.exact_hits},`;
                csv += `${results.avg_signed_rel_error},`;
                csv += `${results.avg_signed_error},`;
                csv += `${results.sum_signed_rel_error},`;
                csv += `${results.sum_signed_error},`;
                csv += `${results.positive_errors},`;
                csv += `${results.negative_errors},`;
                csv += `${results.exact_positives},`;
                csv += `${results.exact_negatives},`;
                csv += `${results.opposite_sign}\n`;
            }

            const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "rvbna_statistics.csv";
            a.click();
            URL.revokeObjectURL(url);
            
            showToast("Statistics CSV downloaded!");
        });
    }

    function restoreStateFromHash() {
        const hash = window.location.hash;
        if (!hash.startsWith("#state=")) return false;
        
        try {
            const b64 = hash.substring(7);
            const json = decodeURIComponent(atob(b64));
            const state = JSON.parse(json);

            // adding console logging of state
            console.log("Restoring state from hash:", state);
            if (state.appVersion) {
                loadedStateVersion = state.appVersion;
            }

            // Restore form fields
            //if (state.dataSource) {
            //    document.getElementById("cfg-data-source").value = state.dataSource;
            //    document.getElementById("cfg-data-source").dispatchEvent(new Event("change"));
            //}
            if (state.n) document.getElementById("cfg-n").value = state.n;
            if (state.k) document.getElementById("cfg-k").value = state.k;
            if (state.inputPrec) document.getElementById("cfg-input-prec").value = state.inputPrec;
            if (state.seed !== undefined && state.seed !== null) {
                const seedInput = document.getElementById("cfg-seed");
                if (seedInput) seedInput.value = state.seed;
            }

            // Restore per-vector distribution params
            if (state.aDistribution) {
                document.getElementById("cfg-a-dist").value = state.aDistribution;
                updateDistLabels("a", state.aDistribution);
            }
            if (state.aAverage != null) document.getElementById("cfg-a-avg").value = state.aAverage;
            if (state.aSigma != null) document.getElementById("cfg-a-sigma").value = state.aSigma;
            if (state.bDistribution) {
                document.getElementById("cfg-b-dist").value = state.bDistribution;
                updateDistLabels("b", state.bDistribution);
            }
            if (state.bAverage != null) document.getElementById("cfg-b-avg").value = state.bAverage;
            if (state.bSigma != null) document.getElementById("cfg-b-sigma").value = state.bSigma;

            // Backward compat: restore old shared average/sigma into both vectors
            if (state.average != null && state.aAverage == null) {
                document.getElementById("cfg-a-avg").value = state.average;
                document.getElementById("cfg-b-avg").value = state.average;
            }
            if (state.sigma != null && state.aSigma == null) {
                document.getElementById("cfg-a-sigma").value = state.sigma;
                document.getElementById("cfg-b-sigma").value = state.sigma;
            }

            // Clear container and restore schemes
            schemesContainer.innerHTML = "";
            state.schemes.forEach(s => {
                addSchemeCard(s.variant, s, true);
            });
            
            return true;
        } catch (err) {
            console.error("Failed to restore state", err);
            return false;
        }
    }

    // ── Create default scheme cards on page load ─────────────────
    if (restoreStateFromHash()) {
        // Automatically evaluate if state was restored from URL
        setTimeout(() => evaluateBtn.click(), 50);
    } else {
        addSchemeCard("correctly_rounded");
        addSchemeCard("approx_mult");
        addSchemeCard("approx_mult_acc");
        addSchemeCard("fma");
        addSchemeCard("bulk_norm");
    }

    // ═══════════════════════════════════════════════════════════════
    //  Build request payload
    // ═══════════════════════════════════════════════════════════════

    function buildRequest() {
        const fd = new FormData(form);
        const schemes = [];

        schemesContainer.querySelectorAll(".scheme-card").forEach((card) => {
            const toggle = card.querySelector(".scheme-toggle");
            if (!toggle.checked) return;

            const variant = card.querySelector(".scheme-type-select").value;
            const entry = { variant };

            // Collect per-scheme parameters
            card.querySelectorAll(".scheme-param").forEach((param) => {
                entry[param.dataset.param] = param.value;
            });

            const customNameInput = card.querySelector(".scheme-name-input");
            const customName = customNameInput ? customNameInput.value.trim() : "";
            if (customName) entry.customName = customName;
            
            entry.name = customName || buildSchemeName(variant, entry);
            schemes.push(entry);
        });

        const parseNum = (val, defaultVal) => {
            const p = parseFloat(val);
            return isNaN(p) ? defaultVal : p;
        };

        const aAvg = parseNum(fd.get("aAverage"), 5.0);
        const aSig = parseNum(fd.get("aSigma"), 5.0);
        const bAvg = parseNum(fd.get("bAverage"), 5.0);
        const bSig = parseNum(fd.get("bSigma"), 5.0);

        const seedVal = fd.get("seed");
        const parsedSeed = seedVal ? parseInt(seedVal) : null;

        return {
            appVersion: currentAppVersion,
            dataSource: fd.get("dataSource") || "random",
            n: parseInt(fd.get("n")) || 1000,
            k: parseInt(fd.get("k")) || 2,
            // Keep shared average/sigma as mean of the two for backward compat
            average: (aAvg + bAvg) / 2,
            sigma: (aSig + bSig) / 2,
            inputPrec: fd.get("inputPrec") || "fp16",
            aDistribution: fd.get("aDistribution") || "gaussian",
            aAverage: aAvg,
            aSigma: aSig,
            bDistribution: fd.get("bDistribution") || "gaussian",
            bAverage: bAvg,
            bSigma: bSig,
            seed: isNaN(parsedSeed) ? null : parsedSeed,
            schemes,
        };
    }

    function buildSchemeName(variant, params) {
        switch (variant) {
            case "correctly_rounded":
                return `Correctly Rounded [${(params.resPrec || "fp32").toUpperCase()}]`;
            case "approx_mult":
                return `FP MUL [${(params.multPrec || "fp16").toUpperCase()}] + Exact Acc`;
            case "approx_mult_acc":
                return `FP MUL [${(params.multPrec || "fp16").toUpperCase()}] + Add Tree [${(params.addPrec || "fp16").toUpperCase()}]`;
            case "approx_mult_linear_acc":
                return `FP MUL [${(params.multPrec || "fp16").toUpperCase()}] + Linear Add [${(params.addPrec || "fp16").toUpperCase()}]`;
            case "fma":
                return `FMA [${(params.fmaPrec || "fp32").toUpperCase()}]`;
            case "bulk_norm":
                return `Bulk Norm [Fixed ${params.bulkNormPrec || 25}, Final ${params.finalPrec || 24}]`;
            default:
                return variant;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Status & formatting helpers
    // ═══════════════════════════════════════════════════════════════

    function setStatus(state, text) {
        statusPill.className = "status-pill" + (state ? ` ${state}` : "");
        statusText.textContent = text;
    }

    function fmtSci(num) {
        if (num === 0) return "0";
        if (!isFinite(num)) return "∞";
        return num.toExponential(3);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Evaluate
    // ═══════════════════════════════════════════════════════════════

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = buildRequest();
        if (payload.schemes.length === 0) {
            setStatus("error", "No schemes enabled");
            return;
        }

        // UI → loading
        evaluateBtn.disabled = true;
        btnLabel.style.display = "none";
        btnSpinner.style.display = "block";
        setStatus("loading", "Evaluating…");
        statsBody.innerHTML = `<tr><td colspan="14" class="empty-state">Computing…</td></tr>`;
        const dataStatsBody = document.getElementById("data-stats-body");
        if (dataStatsBody) dataStatsBody.innerHTML = `<tr><td colspan="5" class="empty-state">Computing…</td></tr>`;

        try {
            const resp = await fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) throw new Error(`Server error ${resp.status}`);
            const data = await resp.json();
            renderResults(data, payload);
            setStatus("", "Done");
        } catch (err) {
            console.error(err);
            setStatus("error", "Error");
            statsBody.innerHTML = `<tr><td colspan="14" class="empty-state" style="color:var(--error)">Evaluation failed — ${err.message}</td></tr>`;
            if (dataStatsBody) dataStatsBody.innerHTML = `<tr><td colspan="5" class="empty-state" style="color:var(--error)">Evaluation failed — ${err.message}</td></tr>`;
        } finally {
            evaluateBtn.disabled = false;
            btnLabel.style.display = "inline";
            btnSpinner.style.display = "none";
        }
    });

    // ═══════════════════════════════════════════════════════════════
    //  Render results
    // ═══════════════════════════════════════════════════════════════

    function renderResults(data, payload) {
        const n = payload.n;
        lastEvalData = data;
        lastEvalN = n;
        
        const crScheme = payload.schemes.find(s => s.variant === "correctly_rounded");
        const crPrec = crScheme ? crScheme.resPrec : "fp32";
        const precBits = {
            "bf16": 7,
            "fp16": 10,
            "fp32": 23,
            "fp64": 52
        };
        const p = precBits[crPrec] || 23;
        lastMinLog2 = -p;
        
        let minActualLog2 = Infinity;
        for (const [, r] of Object.entries(data)) {
            for (const v of r.sorted_rel_errors) {
                if (v > 0) {
                    const l = Math.log2(v);
                    if (l < minActualLog2) minActualLog2 = l;
                }
            }
        }
        lastExactY = minActualLog2 !== Infinity ? Math.floor((minActualLog2 - lastMinLog2) - 1) : -1;

        biasedLog2Panel.style.display = "";
        const signedBiasedLog2Panel = document.getElementById("signed-biased-log2-panel");
        if (signedBiasedLog2Panel) signedBiasedLog2Panel.style.display = "";
        const splitBiasedLog2Panel = document.getElementById("split-biased-log2-panel");
        if (splitBiasedLog2Panel) splitBiasedLog2Panel.style.display = "";
        const quadSplitBiasedLog2Panel = document.getElementById("quad-split-biased-log2-panel");
        if (quadSplitBiasedLog2Panel) quadSplitBiasedLog2Panel.style.display = "";
        const cdfPanel = document.getElementById("cdf-panel");
        if (cdfPanel) cdfPanel.style.display = "";
        chartPlaceholder.classList.add("hidden");

        const dataStatsPanel = document.getElementById("data-stats-panel");
        if (dataStatsPanel) dataStatsPanel.style.display = "";

        const traces = [];
        statsBody.innerHTML = "";
        
        const dataStatsBody = document.getElementById("data-stats-body");
        if (dataStatsBody) dataStatsBody.innerHTML = "";

        const entries = Object.entries(data).sort(
            ([, a], [, b]) => a.geometric_mean - b.geometric_mean
        );

        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];

            // Plotly trace
            const yData = results.sorted_rel_errors.map((v) => (v === 0 ? null : v));
            const xData = Array.from({ length: yData.length }, (_, i) => i);

            traces.push({
                x: xData,
                y: yData,
                type: "scattergl",
                mode: "lines",
                name: schemeName,
                line: { color, width: 2 },
                hovertemplate: "%{y:.4e}<extra>" + schemeName + "</extra>",
            });

            // Stats table row
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>
                    <span class="scheme-label">
                        <span class="color-dot" style="background:${color}"></span>
                        ${schemeName}
                    </span>
                </td>
                <td>${fmtSci(results.min)}</td>
                <td>${fmtSci(results.max)}</td>
                <td>${fmtSci(results.geometric_mean)}</td>
                <td>${(results.exact_count / n * 100).toFixed(1)}% (${results.exact_count})</td>
                <td>${fmtSci(results.mean_signed_rel_error)}</td>
                <td>${fmtSci(results.mean_signed_error)}</td>
                <td>${fmtSci(results.sum_signed_rel_error)}</td>
                <td>${fmtSci(results.sum_signed_error)}</td>
                <td>${results.pos_count}</td>
                <td>${results.neg_count}</td>
                <td>${results.exact_pos_count}</td>
                <td>${results.exact_neg_count}</td>
                <td>${results.opposite_sign_count}</td>
            `;
            statsBody.appendChild(tr);
        });

        if (dataStatsBody && entries.length > 0) {
            const firstRes = entries[0][1];
            const seedText = firstRes.seed !== undefined && firstRes.seed !== null ? firstRes.seed : "Random";
            dataStatsBody.innerHTML = `
                <tr>
                    <td>${seedText}</td>
                    <td>${firstRes.pos_a_count}</td>
                    <td>${firstRes.neg_a_count}</td>
                    <td>${firstRes.pos_b_count}</td>
                    <td>${firstRes.neg_b_count}</td>
                </tr>
            `;
        }

        // Render Plotly chart
        const chartDiv = document.getElementById("plotly-chart");
        const wrap = document.getElementById("chart-wrap");
        const layout = {
            ...PLOTLY_LAYOUT,
            showlegend: true,
            autosize: true,
            height: Math.max(wrap.clientHeight, 600),
        };

        Plotly.react(chartDiv, traces, layout, PLOTLY_CONFIG);

        // ── Biased Log₂ Error chart ──────────────────────────────
        // Same transformation as the CSV: log2(error) - minLog2, with lastExactY for exact
        const minLog2 = lastMinLog2;
        const exactY = lastExactY;

        const biasedTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            const yBiased = results.sorted_rel_errors.map(v =>
                v === 0 ? exactY : Math.log2(v) - minLog2
            );
            const xData = Array.from({ length: yBiased.length }, (_, i) => i);
            biasedTraces.push({
                x: xData,
                y: yBiased,
                type: "scattergl",
                mode: "lines",
                name: schemeName,
                line: { color, width: 2 },
                hovertemplate: "%{y:.2f}<extra>" + schemeName + "</extra>",
            });
        });

        const biasedWrap = document.getElementById("biased-log2-wrap");
        Plotly.react("biased-log2-chart", biasedTraces, {
            ...PLOTLY_LAYOUT,
            showlegend: true,
            autosize: true,
            height: Math.max(biasedWrap.clientHeight, 600),
            yaxis: {
                ...PLOTLY_LAYOUT.yaxis,
                type: "linear",
                title: { text: "Biased log₂(error)", standoff: 10 },
            },
            xaxis: {
                ...PLOTLY_LAYOUT.xaxis,
                title: { text: "Sorted sample index", standoff: 10 },
            },
        }, PLOTLY_CONFIG);

        // ── Signed Biased Log₂ Error chart ──────────────────────────────
        const signedBiasedTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            const ySignedBiased = results.sorted_signed_rel_errors.map(v => {
                if (v === 0) return 0;
                const mag = Math.log2(Math.abs(v)) - minLog2;
                return v > 0 ? mag : -mag;
            });
            const xData = Array.from({ length: ySignedBiased.length }, (_, i) => i);
            signedBiasedTraces.push({
                x: xData,
                y: ySignedBiased,
                type: "scattergl",
                mode: "lines",
                name: schemeName,
                line: { color, width: 2 },
                hovertemplate: "%{y:.2f}<extra>" + schemeName + "</extra>",
            });
        });

        const signedBiasedWrap = document.getElementById("signed-biased-log2-wrap");
        if (signedBiasedWrap) {
            Plotly.react("signed-biased-log2-chart", signedBiasedTraces, {
                ...PLOTLY_LAYOUT,
                showlegend: true,
                autosize: true,
                height: Math.max(signedBiasedWrap.clientHeight, 600),
                yaxis: {
                    ...PLOTLY_LAYOUT.yaxis,
                    type: "linear",
                    title: { text: "Signed Biased log₂(error)", standoff: 10 },
                },
                xaxis: {
                    ...PLOTLY_LAYOUT.xaxis,
                    title: { text: "Sorted sample index", standoff: 10 },
                },
            }, PLOTLY_CONFIG);
        }

        // ── Split Biased Log₂ Error chart ──────────────────────────────
        const splitBiasedTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            
            const posErrors = results.sorted_signed_rel_errors.filter(v => v > 0);
            const negErrors = results.sorted_signed_rel_errors.filter(v => v < 0);
            
            const yPos = posErrors.map(v => Math.log2(v) - minLog2).sort((a, b) => a - b);
            const xPos = Array.from({ length: yPos.length }, (_, i) => i);
            
            const yNeg = negErrors.map(v => Math.log2(Math.abs(v)) - minLog2).sort((a, b) => a - b);
            const xNeg = Array.from({ length: yNeg.length }, (_, i) => i);
            
            if (yPos.length > 0) {
                splitBiasedTraces.push({
                    x: xPos,
                    y: yPos,
                    type: "scattergl",
                    mode: "lines",
                    name: schemeName + " (+)",
                    line: { color, width: 2, dash: 'solid' },
                    hovertemplate: "%{y:.2f}<extra>" + schemeName + " (+)</extra>",
                });
            }
            if (yNeg.length > 0) {
                splitBiasedTraces.push({
                    x: xNeg,
                    y: yNeg,
                    type: "scattergl",
                    mode: "lines",
                    name: schemeName + " (-)",
                    line: { color, width: 2, dash: 'dot' },
                    hovertemplate: "%{y:.2f}<extra>" + schemeName + " (-)</extra>",
                });
            }
        });

        const splitBiasedWrap = document.getElementById("split-biased-log2-wrap");
        if (splitBiasedWrap) {
            Plotly.react("split-biased-log2-chart", splitBiasedTraces, {
                ...PLOTLY_LAYOUT,
                showlegend: true,
                autosize: true,
                height: Math.max(splitBiasedWrap.clientHeight, 600),
                yaxis: {
                    ...PLOTLY_LAYOUT.yaxis,
                    type: "linear",
                    title: { text: "Biased log₂(error)", standoff: 10 },
                },
                xaxis: {
                    ...PLOTLY_LAYOUT.xaxis,
                    title: { text: "Sorted sample index", standoff: 10 },
                },
            }, PLOTLY_CONFIG);
        }

        // ── 4-Way Split Biased Log₂ Error chart ──────────────────────────────
        const quadSplitBiasedTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            
            const errPosResPos = results.sorted_err_pos_res_pos || [];
            const errNegResPos = results.sorted_err_neg_res_pos || [];
            const errPosResNeg = results.sorted_err_pos_res_neg || [];
            const errNegResNeg = results.sorted_err_neg_res_neg || [];
            
            const processErrors = (arr, isNegError) => {
                return arr.map(v => isNegError ? (Math.log2(Math.abs(v)) - minLog2) : (Math.log2(v) - minLog2)).sort((a, b) => a - b);
            };
            
            const y1 = processErrors(errPosResPos, false);
            const y2 = processErrors(errNegResPos, true);
            const y3 = processErrors(errPosResNeg, false);
            const y4 = processErrors(errNegResNeg, true);
            
            const x1 = Array.from({ length: y1.length }, (_, i) => i);
            const x2 = Array.from({ length: y2.length }, (_, i) => i);
            const x3 = Array.from({ length: y3.length }, (_, i) => i);
            const x4 = Array.from({ length: y4.length }, (_, i) => i);
            
            if (y1.length > 0) quadSplitBiasedTraces.push({ x: x1, y: y1, type: "scattergl", mode: "lines", name: schemeName + " (+Err, +Res)", line: { color, width: 2, dash: 'solid' }, hovertemplate: "%{y:.2f}<extra>" + schemeName + " (+Err, +Res)</extra>" });
            if (y2.length > 0) quadSplitBiasedTraces.push({ x: x2, y: y2, type: "scattergl", mode: "lines", name: schemeName + " (-Err, +Res)", line: { color, width: 2, dash: 'dot' }, hovertemplate: "%{y:.2f}<extra>" + schemeName + " (-Err, +Res)</extra>" });
            if (y3.length > 0) quadSplitBiasedTraces.push({ x: x3, y: y3, type: "scattergl", mode: "lines", name: schemeName + " (+Err, -Res)", line: { color, width: 2, dash: 'dash' }, hovertemplate: "%{y:.2f}<extra>" + schemeName + " (+Err, -Res)</extra>" });
            if (y4.length > 0) quadSplitBiasedTraces.push({ x: x4, y: y4, type: "scattergl", mode: "lines", name: schemeName + " (-Err, -Res)", line: { color, width: 2, dash: 'dashdot' }, hovertemplate: "%{y:.2f}<extra>" + schemeName + " (-Err, -Res)</extra>" });
        });

        const quadSplitBiasedWrap = document.getElementById("quad-split-biased-log2-wrap");
        if (quadSplitBiasedWrap) {
            Plotly.react("quad-split-biased-log2-chart", quadSplitBiasedTraces, {
                ...PLOTLY_LAYOUT,
                showlegend: true,
                autosize: true,
                height: Math.max(quadSplitBiasedWrap.clientHeight, 600),
                yaxis: {
                    ...PLOTLY_LAYOUT.yaxis,
                    type: "linear",
                    title: { text: "Biased log₂(|error|)", standoff: 10 },
                },
                xaxis: {
                    ...PLOTLY_LAYOUT.xaxis,
                    title: { text: "Sorted sample index", standoff: 10 },
                }
            }, PLOTLY_CONFIG);
        }

        // ── CDF chart ────────────────────────────────────────────
        // Find global max biased log2 so all curves extend to the same right edge
        let maxBiased = -Infinity;
        for (const [, results] of entries) {
            for (const v of results.sorted_rel_errors) {
                if (v > 0) {
                    const b = Math.log2(v) - minLog2;
                    if (b > maxBiased) maxBiased = b;
                }
            }
        }

        const cdfTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            // Compute biased log2 values, sorted
            const biased = results.sorted_rel_errors.map(v =>
                v === 0 ? -1 : Math.log2(v) - minLog2
            ).sort((a, b) => a - b);

            const total = biased.length;
            // CDF: x = biased log2, y = cumulative fraction
            const xCdf = [...biased];
            const yCdf = biased.map((_, i) => (i + 1) / total);

            // Extend curve to global max so all curves end at the same x
            if (xCdf[xCdf.length - 1] < maxBiased) {
                xCdf.push(maxBiased);
                yCdf.push(1.0);
            }

            cdfTraces.push({
                x: xCdf,
                y: yCdf,
                type: "scattergl",
                mode: "lines",
                name: schemeName,
                line: { color, width: 2 },
                hovertemplate: "log₂: %{x:.2f}<br>CDF: %{y:.3f}<extra>" + schemeName + "</extra>",
            });
        });

        const cdfWrap = document.getElementById("cdf-wrap");
        Plotly.react("cdf-chart", cdfTraces, {
            ...PLOTLY_LAYOUT,
            showlegend: true,
            autosize: true,
            height: Math.max(cdfWrap.clientHeight, 600),
            yaxis: {
                ...PLOTLY_LAYOUT.yaxis,
                type: "linear",
                title: { text: "Cumulative fraction", standoff: 10 },
                range: [0, 1.05],
            },
            xaxis: {
                ...PLOTLY_LAYOUT.xaxis,
                title: { text: "Biased log₂(error)", standoff: 10 },
            },
        }, PLOTLY_CONFIG);
    }
});
