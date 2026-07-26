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
    const biasedLog2Panel = document.getElementById("biased-log2-panel");
    const cdfPanel = document.getElementById("cdf-panel");
    const toast = document.getElementById("toast");

    fetch('/api/version')
        .then(res => res.json())
        .then(data => {
            const versionEl = document.getElementById('app-version');
            if (versionEl) versionEl.textContent = data.version;
        })
        .catch(err => console.error("Error fetching version:", err));

    let lastEvalData = null;
    let lastEvalN = 0;


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
        exact: {
            label: "Exact Dot Product",
            desc: "No intermediate rounding. Exact accumulation, single final round.",
            badge: "REF",
            params: [],
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
        margin: { t: 24, r: 24, b: 52, l: 60 },
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
            yanchor: "top", y: -0.18,
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
    function addSchemeCard(initialType = "exact", initialParams = {}, isActive = true) {
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
                    <select class="scheme-type-select">
                        ${Object.entries(SCHEME_TYPES)
                            .map(([key, def]) => `<option value="${key}"${key === initialType ? " selected" : ""}>${def.label}</option>`)
                            .join("")}
                    </select>
                </div>
                <div class="scheme-header-right">
                    ${badgeHTML}
                    <button type="button" class="remove-scheme-btn" title="Remove scheme" aria-label="Remove scheme">
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
        card.querySelector(".remove-scheme-btn").addEventListener("click", () => {
            card.style.animation = "fadeSlideOut 0.25s ease forwards";
            card.addEventListener("animationend", () => card.remove(), { once: true });
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
        let minLog2 = Infinity;
        let maxLog2 = -Infinity;
        for (const [, r] of entries) {
            for (const v of r.sorted_rel_errors) {
                if (v > 0) {
                    const l = Math.log2(v);
                    if (l < minLog2) minLog2 = l;
                    if (l > maxLog2) maxLog2 = l;
                }
            }
        }
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
                        scaled = -1;
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
        let minLog2 = Infinity;
        for (const [, r] of entries) {
            for (const v of r.sorted_rel_errors) {
                if (v > 0) {
                    const l = Math.log2(v);
                    if (l < minLog2) minLog2 = l;
                }
            }
        }

        // For each scheme, compute biased log2 values and build CDF
        // Collect all unique biased log2 values across schemes for a shared x-axis
        const allBiasedValues = new Set();
        const schemeData = [];

        for (const [, r] of entries) {
            const biased = [];
            for (const v of r.sorted_rel_errors) {
                if (v === 0) {
                    biased.push(-1);
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

    function restoreStateFromHash() {
        const hash = window.location.hash;
        if (!hash.startsWith("#state=")) return false;
        
        try {
            const b64 = hash.substring(7);
            const json = decodeURIComponent(atob(b64));
            const state = JSON.parse(json);

            // Restore form fields
            if (state.dataSource) {
                document.getElementById("cfg-data-source").value = state.dataSource;
                document.getElementById("cfg-data-source").dispatchEvent(new Event("change"));
            }
            if (state.n) document.getElementById("cfg-n").value = state.n;
            if (state.k) document.getElementById("cfg-k").value = state.k;
            if (state.inputPrec) document.getElementById("cfg-input-prec").value = state.inputPrec;

            // Restore per-vector distribution params
            if (state.aDistribution) {
                document.getElementById("cfg-a-dist").value = state.aDistribution;
                updateDistLabels("a", state.aDistribution);
            }
            if (state.aAverage !== undefined) document.getElementById("cfg-a-avg").value = state.aAverage;
            if (state.aSigma !== undefined) document.getElementById("cfg-a-sigma").value = state.aSigma;
            if (state.bDistribution) {
                document.getElementById("cfg-b-dist").value = state.bDistribution;
                updateDistLabels("b", state.bDistribution);
            }
            if (state.bAverage !== undefined) document.getElementById("cfg-b-avg").value = state.bAverage;
            if (state.bSigma !== undefined) document.getElementById("cfg-b-sigma").value = state.bSigma;

            // Backward compat: restore old shared average/sigma into both vectors
            if (state.average !== undefined && state.aAverage === undefined) {
                document.getElementById("cfg-a-avg").value = state.average;
                document.getElementById("cfg-b-avg").value = state.average;
            }
            if (state.sigma !== undefined && state.aSigma === undefined) {
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
        addSchemeCard("exact");
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

            entry.name = buildSchemeName(variant, entry);
            schemes.push(entry);
        });

        const aAvg = parseFloat(fd.get("aAverage")) || 5.0;
        const aSig = parseFloat(fd.get("aSigma")) || 5.0;
        const bAvg = parseFloat(fd.get("bAverage")) || 5.0;
        const bSig = parseFloat(fd.get("bSigma")) || 5.0;

        return {
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
            schemes,
        };
    }

    function buildSchemeName(variant, params) {
        switch (variant) {
            case "exact":
                return "Exact Dot Product";
            case "approx_mult":
                return `FP MUL [${(params.multPrec || "fp16").toUpperCase()}] + Exact Acc`;
            case "approx_mult_acc":
                return `FP MUL [${(params.multPrec || "fp16").toUpperCase()}] + Add Tree [${(params.addPrec || "fp16").toUpperCase()}]`;
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
        statsBody.innerHTML = `<tr><td colspan="5" class="empty-state">Computing…</td></tr>`;

        try {
            const resp = await fetch("/api/evaluate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) throw new Error(`Server error ${resp.status}`);
            const data = await resp.json();
            renderResults(data, payload.n);
            setStatus("", "Done");
        } catch (err) {
            console.error(err);
            setStatus("error", "Error");
            statsBody.innerHTML = `<tr><td colspan="5" class="empty-state" style="color:var(--error)">Evaluation failed — ${err.message}</td></tr>`;
        } finally {
            evaluateBtn.disabled = false;
            btnLabel.style.display = "inline";
            btnSpinner.style.display = "none";
        }
    });

    // ═══════════════════════════════════════════════════════════════
    //  Render results
    // ═══════════════════════════════════════════════════════════════

    function renderResults(data, n) {
        lastEvalData = data;
        lastEvalN = n;
        biasedLog2Panel.style.display = "";
        cdfPanel.style.display = "";
        chartPlaceholder.classList.add("hidden");

        const traces = [];
        statsBody.innerHTML = "";

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
            `;
            statsBody.appendChild(tr);
        });

        // Render Plotly chart
        const chartDiv = document.getElementById("plotly-chart");
        const wrap = document.getElementById("chart-wrap");
        const layout = {
            ...PLOTLY_LAYOUT,
            showlegend: true,
            autosize: true,
            height: Math.max(wrap.clientHeight, 450),
        };

        Plotly.react(chartDiv, traces, layout, PLOTLY_CONFIG);

        // ── Biased Log₂ Error chart ──────────────────────────────
        // Same transformation as the CSV: log2(error) - minLog2, with -1 for exact
        let minLog2 = Infinity;
        for (const [, results] of entries) {
            for (const v of results.sorted_rel_errors) {
                if (v > 0) {
                    const l = Math.log2(v);
                    if (l < minLog2) minLog2 = l;
                }
            }
        }

        const biasedTraces = [];
        entries.forEach(([schemeName, results], idx) => {
            const color = SCHEME_COLORS[idx % SCHEME_COLORS.length];
            const yBiased = results.sorted_rel_errors.map(v =>
                v === 0 ? -1 : Math.log2(v) - minLog2
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
            height: Math.max(biasedWrap.clientHeight, 400),
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
            height: Math.max(cdfWrap.clientHeight, 400),
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
