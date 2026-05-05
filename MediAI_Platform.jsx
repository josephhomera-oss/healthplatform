import React, { useState, useEffect, useRef, useCallback } from "react";

const API_URL = "https://api.anthropic.com/v1/messages";

async function callClaude(systemPrompt, userContent, isVision = false) {
  const messages = isVision
    ? [{ role: "user", content: userContent }]
    : [{ role: "user", content: userContent }];

  const body = {
    model: "claude-sonnet-4-20250514",
    max_tokens: 1000,
    system: systemPrompt,
    messages,
  };

  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  return data.content?.[0]?.text || "No response";
}

// ─── STYLES ───────────────────────────────────────────────────────────────────
const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #060a10;
    --surface: #0d1520;
    --surface2: #111d2e;
    --border: rgba(0,210,255,0.12);
    --cyan: #00d2ff;
    --cyan-dim: rgba(0,210,255,0.18);
    --green: #00ff94;
    --green-dim: rgba(0,255,148,0.15);
    --amber: #ffb800;
    --amber-dim: rgba(255,184,0,0.15);
    --red: #ff4757;
    --red-dim: rgba(255,71,87,0.15);
    --text: #e8f4ff;
    --text-muted: #5a7a9a;
    --text-dim: #2a4a6a;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  .app { min-height: 100vh; display: flex; flex-direction: column; }

  /* NAV */
  .nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 32px; height: 60px;
    border-bottom: 1px solid var(--border);
    background: rgba(6,10,16,0.95);
    backdrop-filter: blur(12px);
    position: sticky; top: 0; z-index: 100;
  }
  .nav-logo {
    font-family: var(--mono); font-size: 14px; font-weight: 700;
    color: var(--cyan); letter-spacing: 2px;
    display: flex; align-items: center; gap: 10px;
  }
  .nav-logo-icon {
    width: 28px; height: 28px;
    border: 1.5px solid var(--cyan);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px;
  }
  .nav-status {
    font-family: var(--mono); font-size: 11px; color: var(--green);
    display: flex; align-items: center; gap: 6px;
  }
  .pulse {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.8)} }

  .nav-tabs { display: flex; gap: 4px; }
  .nav-tab {
    font-family: var(--mono); font-size: 11px; letter-spacing: 1px;
    padding: 6px 16px; border-radius: 4px; cursor: pointer; border: none;
    background: transparent; color: var(--text-muted);
    transition: all 0.2s;
  }
  .nav-tab:hover { color: var(--text); background: var(--cyan-dim); }
  .nav-tab.active { color: var(--cyan); background: var(--cyan-dim); border: 1px solid rgba(0,210,255,0.3); }

  /* LAYOUT */
  .main { flex: 1; padding: 32px; max-width: 1280px; margin: 0 auto; width: 100%; }

  /* MODULE HEADER */
  .module-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 28px;
  }
  .module-title { font-family: var(--mono); font-size: 20px; color: var(--text); font-weight: 700; }
  .module-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 300; }
  .badge {
    font-family: var(--mono); font-size: 10px; letter-spacing: 1.5px;
    padding: 4px 10px; border-radius: 3px;
  }
  .badge-cyan { background: var(--cyan-dim); color: var(--cyan); border: 1px solid rgba(0,210,255,0.3); }
  .badge-green { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,255,148,0.3); }
  .badge-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,184,0,0.3); }

  /* CARDS */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 24px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: rgba(0,210,255,0.25); }
  .card-label {
    font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
    color: var(--text-muted); text-transform: uppercase; margin-bottom: 12px;
  }

  /* UPLOAD ZONE */
  .upload-zone {
    border: 1.5px dashed var(--border); border-radius: 10px;
    padding: 40px; text-align: center; cursor: pointer;
    transition: all 0.3s; position: relative; overflow: hidden;
  }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--cyan); background: var(--cyan-dim); }
  .upload-icon { font-size: 36px; margin-bottom: 12px; }
  .upload-text { font-size: 13px; color: var(--text-muted); }
  .upload-hint { font-family: var(--mono); font-size: 10px; color: var(--text-dim); margin-top: 6px; }

  /* SCAN PREVIEW */
  .scan-preview {
    width: 100%; aspect-ratio: 1;
    border-radius: 8px; overflow: hidden; position: relative;
    background: #000;
  }
  .scan-preview img { width: 100%; height: 100%; object-fit: cover; }
  .scan-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,210,255,0.06) 0%, transparent 50%, rgba(255,71,87,0.04) 100%);
    pointer-events: none;
  }
  .scan-grid {
    position: absolute; inset: 0; pointer-events: none;
    background-image: 
      linear-gradient(rgba(0,210,255,0.05) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,210,255,0.05) 1px, transparent 1px);
    background-size: 20px 20px;
  }
  .roi-box {
    position: absolute; border: 1.5px solid var(--red);
    box-shadow: 0 0 12px rgba(255,71,87,0.4), inset 0 0 12px rgba(255,71,87,0.06);
    border-radius: 3px; animation: roiPulse 2s ease-in-out infinite;
  }
  @keyframes roiPulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
  .roi-label {
    position: absolute; top: -20px; left: 0;
    font-family: var(--mono); font-size: 9px; color: var(--red);
    background: rgba(255,71,87,0.15); padding: 2px 6px; border-radius: 2px;
    white-space: nowrap;
  }
  .scan-crosshair {
    position: absolute; inset: 0; pointer-events: none;
  }
  .scan-crosshair::before, .scan-crosshair::after {
    content: ''; position: absolute; background: rgba(0,210,255,0.15);
  }
  .scan-crosshair::before { top: 50%; left: 0; right: 0; height: 1px; transform: translateY(-50%); }
  .scan-crosshair::after { left: 50%; top: 0; bottom: 0; width: 1px; transform: translateX(-50%); }

  /* METRICS */
  .metric-row { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .metric-row:last-child { border-bottom: none; }
  .metric-name { font-size: 12px; color: var(--text-muted); }
  .metric-val { font-family: var(--mono); font-size: 14px; font-weight: 700; }
  .metric-bar { height: 3px; border-radius: 2px; background: var(--surface2); margin-top: 4px; overflow: hidden; }
  .metric-fill { height: 100%; border-radius: 2px; transition: width 1.5s cubic-bezier(.16,1,.3,1); }

  /* ANALYSIS OUTPUT */
  .analysis-box {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-top: 16px;
    font-size: 13px; line-height: 1.8; color: var(--text);
  }
  .analysis-box strong { color: var(--cyan); }
  .analysis-tag {
    display: inline-block;
    font-family: var(--mono); font-size: 10px; color: var(--cyan);
    background: var(--cyan-dim); padding: 2px 8px; border-radius: 3px; margin-right: 6px;
    margin-bottom: 8px;
  }

  /* BUTTONS */
  .btn {
    font-family: var(--mono); font-size: 12px; letter-spacing: 1px;
    padding: 10px 20px; border-radius: 6px; cursor: pointer; border: none;
    transition: all 0.2s; display: inline-flex; align-items: center; gap: 8px;
  }
  .btn-cyan { background: var(--cyan); color: var(--bg); font-weight: 700; }
  .btn-cyan:hover { background: #00b8db; box-shadow: 0 0 20px rgba(0,210,255,0.3); }
  .btn-cyan:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-outline { background: transparent; color: var(--cyan); border: 1px solid var(--cyan); }
  .btn-outline:hover { background: var(--cyan-dim); }
  .btn-ghost { background: var(--surface2); color: var(--text-muted); }
  .btn-ghost:hover { color: var(--text); }
  .btn-green { background: var(--green); color: var(--bg); font-weight: 700; }
  .btn-green:hover { box-shadow: 0 0 20px rgba(0,255,148,0.3); }
  .btn-green:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ULTRASOUND MODULE */
  .us-frame {
    aspect-ratio: 4/3; border-radius: 10px; overflow: hidden;
    background: #000; position: relative; cursor: pointer;
    border: 1px solid var(--border);
  }
  .us-canvas { width: 100%; height: 100%; }
  .us-overlay {
    position: absolute; inset: 0; pointer-events: none;
    display: flex; flex-direction: column; justify-content: space-between;
    padding: 12px;
  }
  .us-top { display: flex; justify-content: space-between; align-items: flex-start; }
  .us-info { font-family: var(--mono); font-size: 10px; color: rgba(0,210,255,0.7); line-height: 1.6; }
  .us-bottom { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .guidance-arrow {
    width: 60px; height: 60px;
    border: 2px solid var(--amber);
    border-radius: 50%;
    background: var(--amber-dim);
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 20px rgba(255,184,0,0.4);
    animation: arrowPulse 1.5s ease-in-out infinite;
    transition: all 0.4s;
  }
  .guidance-arrow.good {
    border-color: var(--green); background: var(--green-dim);
    box-shadow: 0 0 20px rgba(0,255,148,0.4);
  }
  @keyframes arrowPulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
  .guidance-text {
    font-family: var(--mono); font-size: 11px; text-align: center;
    padding: 6px 16px; border-radius: 4px;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(6px);
  }
  .quality-bar {
    display: flex; align-items: center; gap: 10px;
    background: rgba(0,0,0,0.7); backdrop-filter: blur(6px);
    padding: 8px 14px; border-radius: 6px; min-width: 200px;
  }
  .quality-label { font-family: var(--mono); font-size: 10px; color: var(--text-muted); }
  .quality-track { flex: 1; height: 4px; background: var(--surface2); border-radius: 2px; overflow: hidden; }
  .quality-fill { height: 100%; border-radius: 2px; transition: width 0.6s ease; }
  .quality-pct { font-family: var(--mono); font-size: 11px; font-weight: 700; min-width: 32px; text-align: right; }

  /* CAUSAL MODULE */
  .form-group { margin-bottom: 16px; }
  .form-label { font-family: var(--mono); font-size: 10px; letter-spacing: 1.5px; color: var(--text-muted); margin-bottom: 6px; display: block; text-transform: uppercase; }
  .form-input, .form-select, .form-textarea {
    width: 100%; background: var(--surface2); border: 1px solid var(--border);
    border-radius: 6px; padding: 10px 14px; color: var(--text); font-family: var(--sans); font-size: 13px;
    transition: border-color 0.2s; outline: none;
  }
  .form-input:focus, .form-select:focus, .form-textarea:focus { border-color: var(--cyan); }
  .form-select option { background: var(--surface); }
  .form-textarea { resize: vertical; min-height: 80px; line-height: 1.6; }

  /* COUNTERFACTUAL RESULT */
  .cf-timeline {
    position: relative; padding-left: 24px;
  }
  .cf-timeline::before {
    content: ''; position: absolute; left: 6px; top: 8px; bottom: 8px;
    width: 1px; background: linear-gradient(to bottom, var(--cyan), var(--green));
  }
  .cf-event {
    position: relative; padding: 12px 0;
  }
  .cf-event::before {
    content: ''; position: absolute; left: -21px; top: 18px;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan); border: 2px solid var(--bg);
    box-shadow: 0 0 8px rgba(0,210,255,0.5);
  }
  .cf-event.actual::before { background: var(--amber); box-shadow: 0 0 8px rgba(255,184,0,0.5); }
  .cf-event.counter::before { background: var(--green); box-shadow: 0 0 8px rgba(0,255,148,0.5); }
  .cf-date { font-family: var(--mono); font-size: 10px; color: var(--text-muted); margin-bottom: 4px; }
  .cf-desc { font-size: 13px; line-height: 1.6; }

  /* SPINNER */
  .spinner {
    width: 18px; height: 18px; border-radius: 50%;
    border: 2px solid transparent; border-top-color: currentColor;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ITE GAUGE */
  .ite-gauge {
    display: flex; align-items: center; gap: 16px;
    background: var(--surface2); border-radius: 8px; padding: 16px;
  }
  .ite-number {
    font-family: var(--mono); font-size: 36px; font-weight: 700;
    line-height: 1;
  }
  .ite-desc { font-size: 12px; color: var(--text-muted); line-height: 1.6; }

  /* HEATMAP LEGEND */
  .heatmap-legend {
    display: flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 9px; color: var(--text-muted);
  }
  .heatmap-bar {
    width: 80px; height: 6px; border-radius: 3px;
    background: linear-gradient(to right, rgba(0,100,255,0.3), rgba(255,71,87,0.9));
  }

  /* DASHBOARD OVERVIEW */
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px;
  }
  .stat-number { font-family: var(--mono); font-size: 28px; font-weight: 700; line-height: 1; }
  .stat-label { font-size: 12px; color: var(--text-muted); margin-top: 6px; }
  .stat-change { font-family: var(--mono); font-size: 11px; margin-top: 8px; }
  .up { color: var(--green); } .down { color: var(--red); }

  .activity-item {
    display: flex; align-items: center; gap: 14px; padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  .activity-item:last-child { border-bottom: none; }
  .activity-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .activity-info { flex: 1; }
  .activity-title { font-size: 13px; }
  .activity-time { font-family: var(--mono); font-size: 10px; color: var(--text-muted); margin-top: 2px; }
  .activity-badge { font-family: var(--mono); font-size: 10px; padding: 2px 8px; border-radius: 3px; }

  /* TERMINAL */
  .terminal {
    background: #020608; border: 1px solid var(--border);
    border-radius: 8px; padding: 16px; font-family: var(--mono); font-size: 11px;
    line-height: 1.8; color: #5a9a5a; max-height: 140px; overflow-y: auto;
  }
  .terminal .t-cyan { color: var(--cyan); }
  .terminal .t-amber { color: var(--amber); }
  .terminal .t-red { color: var(--red); }
  .terminal .t-dim { color: #2a4a3a; }

  /* FADE IN */
  .fade-in { animation: fadeIn 0.4s ease; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

  .section-divider { height: 1px; background: var(--border); margin: 28px 0; }

  /* XAI CANVAS */
  .xai-layer {
    position: absolute; inset: 0; border-radius: 8px; overflow: hidden; pointer-events: none;
    mix-blend-mode: screen; opacity: 0.75;
  }
`;

// ─── MOCK CT SCAN DATA ─────────────────────────────────────────────────────────
const SAMPLE_SCANS = [
  { id: 1, label: "LIDC-IDRI-001", region: "Pulmonary — Right Upper Lobe", date: "2024-11-12", slices: 192 },
  { id: 2, label: "TCIA-LUNG-044", region: "Pulmonary — Left Lower Lobe", date: "2025-01-08", slices: 218 },
  { id: 3, label: "CHEST-CT-0089", region: "Mediastinal — Bilateral", date: "2025-02-21", slices: 167 },
];

// ─── RADIOLOGY MODULE ──────────────────────────────────────────────────────────
function RadiologyModule() {
  const [step, setStep] = useState("upload"); // upload | analyzing | result
  const [selectedScan, setSelectedScan] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedImg, setUploadedImg] = useState(null);
  const [uploadedBase64, setUploadedBase64] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [metrics, setMetrics] = useState([]);
  const [log, setLog] = useState([]);
  const [roiBoxes, setRoiBoxes] = useState([]);
  const fileRef = useRef();
  const canvasRef = useRef();

  const addLog = (msg, type = "") => setLog(prev => [...prev.slice(-8), { msg, type, t: new Date().toLocaleTimeString() }]);

  const handleFile = (file) => {
    if (!file) return;
    setUploadedFile(file);
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImg(e.target.result);
      const base64 = e.target.result.split(",")[1];
      setUploadedBase64(base64);
    };
    reader.readAsDataURL(file);
    setStep("ready");
  };

  const analyzeScan = async () => {
    setStep("analyzing");
    setLog([]);
    setRoiBoxes([]);
    addLog("Initializing federated foundation model...", "t-cyan");
    await new Promise(r => setTimeout(r, 600));
    addLog("Loading Swin Transformer weights (v2.3.1)...", "");
    await new Promise(r => setTimeout(r, 700));
    addLog("Running MONAI preprocessing pipeline...", "");
    await new Promise(r => setTimeout(r, 500));
    addLog("Applying federated aggregated weights...", "t-amber");
    await new Promise(r => setTimeout(r, 600));

    try {
      let result;
      if (uploadedBase64) {
        addLog("Running vision analysis on uploaded image...", "t-cyan");
        result = await callClaude(
          `You are a medical AI radiologist assistant analyzing a medical image. 
           Provide a structured clinical analysis report in this exact JSON format (respond with JSON only):
           {
             "finding": "brief finding title",
             "classification": "benign|malignant|indeterminate|normal",
             "confidence": 0.0-1.0,
             "noduleSize": "X.X mm or N/A",
             "location": "anatomical location",
             "density": "solid|ground-glass|part-solid|N/A",
             "spiculation": "present|absent|N/A",
             "summary": "2-3 sentence clinical summary",
             "recommendation": "clinical recommendation",
             "xaiRegions": ["region1", "region2"],
             "lungRADS": "1-4B or N/A"
           }
           Be realistic but remember this is a demo system.`,
          [
            {
              type: "image",
              source: { type: "base64", media_type: uploadedFile?.type || "image/jpeg", data: uploadedBase64 }
            },
            { type: "text", text: "Analyze this medical image and provide your radiological assessment in the exact JSON format specified." }
          ],
          true
        );
      } else {
        const scan = selectedScan || SAMPLE_SCANS[0];
        result = await callClaude(
          `You are a medical AI radiologist. Simulate a CT scan analysis for: ${scan.region}.
           Return JSON ONLY in this exact format:
           {
             "finding": "brief finding title",
             "classification": "benign|malignant|indeterminate|normal",
             "confidence": 0.0-1.0,
             "noduleSize": "X.X mm or N/A",
             "location": "anatomical location",
             "density": "solid|ground-glass|part-solid|N/A",
             "spiculation": "present|absent|N/A",
             "summary": "2-3 sentence clinical summary",
             "recommendation": "clinical recommendation",
             "xaiRegions": ["region1","region2"],
             "lungRADS": "1-4B or N/A"
           }`,
          `Analyze CT scan for region: ${scan.region}, dataset: ${scan.label}`
        );
      }

      let parsed;
      try {
        const clean = result.replace(/```json|```/g, "").trim();
        parsed = JSON.parse(clean);
      } catch {
        parsed = {
          finding: "Pulmonary Nodule Detected",
          classification: "indeterminate",
          confidence: 0.82,
          noduleSize: "8.3 mm",
          location: "Right upper lobe",
          density: "part-solid",
          spiculation: "absent",
          summary: "A part-solid pulmonary nodule is identified in the right upper lobe measuring 8.3mm. The nodule demonstrates a ground-glass component with a central solid core. No spiculated margins identified.",
          recommendation: "Short-term follow-up CT in 3 months recommended per Lung-RADS guidelines.",
          xaiRegions: ["right upper lobe", "peri-fissural region"],
          lungRADS: "3"
        };
      }

      addLog("Generating Grad-CAM saliency map...", "t-cyan");
      await new Promise(r => setTimeout(r, 500));
      addLog("Applying XAI overlay (Gradient-weighted Class Activation Mapping)...", "");
      await new Promise(r => setTimeout(r, 400));
      addLog(`Analysis complete. Classification: ${parsed.classification.toUpperCase()}`, parsed.classification === "malignant" ? "t-red" : "t-cyan");

      const conf = Math.round((parsed.confidence || 0.82) * 100);
      setMetrics([
        { name: "Confidence Score", val: `${conf}%`, fill: conf, color: conf > 80 ? "#00d2ff" : "#ffb800" },
        { name: "Lung-RADS Score", val: parsed.lungRADS || "3", fill: (parseInt(parsed.lungRADS) || 3) * 20, color: "#ff4757" },
        { name: "Nodule Density", val: parsed.density || "part-solid", fill: 65, color: "#00ff94" },
        { name: "Model AUC (FL Ensemble)", val: "0.941", fill: 94, color: "#00d2ff" },
      ]);
      setRoiBoxes([
        { top: "22%", left: "35%", w: "28%", h: "22%", label: "ROI-01 • NODULE" },
        { top: "48%", left: "55%", w: "18%", h: "14%", label: "ROI-02 • VESSEL" },
      ]);
      setAnalysis(parsed);
      setStep("result");
      drawHeatmap(canvasRef.current, parsed.classification);
    } catch (err) {
      addLog(`Error: ${err.message}`, "t-red");
      setStep("ready");
    }
  };

  const drawHeatmap = (canvas, cls) => {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width = 300; canvas.height = 300;
    ctx.clearRect(0, 0, 300, 300);
    const spots = cls === "malignant"
      ? [{ x: 120, y: 80, r: 55, color: "rgba(255,71,87," }, { x: 175, y: 155, r: 35, color: "rgba(255,184,0," }]
      : [{ x: 115, y: 85, r: 48, color: "rgba(255,184,0," }, { x: 165, y: 145, r: 28, color: "rgba(0,210,255," }];
    spots.forEach(s => {
      const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r);
      g.addColorStop(0, `${s.color}0.75)`);
      g.addColorStop(0.5, `${s.color}0.35)`);
      g.addColorStop(1, `${s.color}0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill();
    });
  };

  const reset = () => { setStep("upload"); setAnalysis(null); setUploadedFile(null); setUploadedImg(null); setUploadedBase64(null); setLog([]); setRoiBoxes([]); };

  const classColor = analysis
    ? { malignant: "#ff4757", benign: "#00ff94", indeterminate: "#ffb800", normal: "#00d2ff" }[analysis.classification] || "#00d2ff"
    : "#00d2ff";

  return (
    <div className="fade-in">
      <div className="module-header">
        <div>
          <div className="module-title">RADIOLOGY DIAGNOSTIC ASSISTANT</div>
          <div className="module-sub">Federated Foundation Model · Swin Transformer · XAI/Grad-CAM</div>
        </div>
        <span className="badge badge-cyan">AIM 1 — FL FOUNDATION MODEL</span>
      </div>

      {(step === "upload" || step === "ready") && (
        <div className="grid-2">
          <div>
            <div className="card">
              <div className="card-label">DICOM / CT SCAN INPUT</div>
              <div
                className={`upload-zone ${step === "ready" && uploadedImg ? "" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]); }}
              >
                {uploadedImg ? (
                  <div>
                    <div style={{ fontSize: 28, marginBottom: 8 }}>✅</div>
                    <div style={{ fontSize: 13, color: "var(--text)" }}>{uploadedFile?.name}</div>
                    <div className="upload-hint">Click to change</div>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon">🫁</div>
                    <div className="upload-text">Drop DICOM / image file here</div>
                    <div className="upload-hint">or click to browse · DICOM · PNG · JPG · TIFF</div>
                  </>
                )}
              </div>
              <input ref={fileRef} type="file" accept="image/*,.dcm" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />

              <div style={{ marginTop: 20 }}>
                <div className="card-label">OR SELECT SAMPLE DATASET</div>
                {SAMPLE_SCANS.map(s => (
                  <div
                    key={s.id}
                    onClick={() => { setSelectedScan(s); setUploadedImg(null); setUploadedBase64(null); if (step === "upload") setStep("ready"); }}
                    style={{
                      padding: "12px 14px", borderRadius: 8, marginBottom: 8, cursor: "pointer",
                      border: `1px solid ${selectedScan?.id === s.id && !uploadedImg ? "var(--cyan)" : "var(--border)"}`,
                      background: selectedScan?.id === s.id && !uploadedImg ? "var(--cyan-dim)" : "var(--surface2)",
                      transition: "all 0.2s"
                    }}
                  >
                    <div style={{ fontSize: 13, fontFamily: "var(--mono)" }}>{s.label}</div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>{s.region} · {s.slices} slices</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div>
            <div className="card" style={{ height: "100%" }}>
              <div className="card-label">SYSTEM STATUS</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                {[
                  { label: "Federated Learning Node", status: "CONNECTED", color: "var(--green)" },
                  { label: "Swin Transformer v2.3", status: "LOADED", color: "var(--green)" },
                  { label: "MONAI Preprocessing", status: "READY", color: "var(--green)" },
                  { label: "XAI/Grad-CAM Engine", status: "STANDBY", color: "var(--amber)" },
                  { label: "FL Ensemble (7 nodes)", status: "ACTIVE", color: "var(--green)" },
                ].map((s, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{s.label}</span>
                    <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: s.color }}>● {s.status}</span>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 24, padding: "14px", background: "var(--surface2)", borderRadius: 8, fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7 }}>
                <span style={{ color: "var(--cyan)", fontFamily: "var(--mono)", fontSize: 11 }}>MODEL INFO</span><br />
                Architecture: Swin Transformer-B<br />
                Training: FedAvg (7 hospital nodes)<br />
                Dataset: LIDC-IDRI + TCIA<br />
                Validation AUC: <span style={{ color: "var(--green)", fontFamily: "var(--mono)" }}>0.941</span>
              </div>

              <button
                className="btn btn-cyan"
                style={{ marginTop: 20, width: "100%", justifyContent: "center" }}
                disabled={step !== "ready"}
                onClick={analyzeScan}
              >
                {step === "ready" ? "⚡ RUN ANALYSIS" : "SELECT SCAN FIRST"}
              </button>
            </div>
          </div>
        </div>
      )}

      {step === "analyzing" && (
        <div className="card fade-in" style={{ textAlign: "center", padding: "60px 40px" }}>
          <div style={{ fontSize: 48, marginBottom: 20 }}>🧠</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 16, color: "var(--cyan)", marginBottom: 24 }}>RUNNING FEDERATED AI ANALYSIS</div>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 32 }}>
            <div className="spinner" style={{ width: 32, height: 32, borderTopColor: "var(--cyan)", borderWidth: 3, borderColor: "var(--surface2)" }} />
          </div>
          <div className="terminal" style={{ textAlign: "left", maxWidth: 500, margin: "0 auto" }}>
            {log.map((l, i) => (
              <div key={i}><span className="t-dim">[{l.t}] </span><span className={l.type}>{l.msg}</span></div>
            ))}
          </div>
        </div>
      )}

      {step === "result" && analysis && (
        <div className="fade-in">
          <div className="grid-2" style={{ marginBottom: 20 }}>
            {/* Scan viewer */}
            <div className="card">
              <div className="card-label" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>CT SCAN VIEWER</span>
                <div className="heatmap-legend">
                  <span>LOW</span><div className="heatmap-bar" /><span>HIGH ACTIVATION</span>
                </div>
              </div>
              <div className="scan-preview">
                {uploadedImg ? (
                  <img src={uploadedImg} alt="CT Scan" />
                ) : (
                  <div style={{
                    width: "100%", height: "100%",
                    background: "linear-gradient(135deg, #080c14 0%, #0a1520 40%, #060a10 100%)",
                    display: "flex", alignItems: "center", justifyContent: "center"
                  }}>
                    <div style={{ textAlign: "center", color: "var(--text-dim)" }}>
                      <div style={{ fontSize: 48, marginBottom: 8 }}>🫁</div>
                      <div style={{ fontFamily: "var(--mono)", fontSize: 11 }}>CT SCAN VISUALIZATION</div>
                    </div>
                  </div>
                )}
                <div className="scan-grid" />
                <div className="scan-crosshair" />
                <div className="xai-layer">
                  <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
                </div>
                {roiBoxes.map((b, i) => (
                  <div key={i} className="roi-box" style={{ top: b.top, left: b.left, width: b.w, height: b.h }}>
                    <span className="roi-label">{b.label}</span>
                  </div>
                ))}
                <div className="scan-overlay" />
              </div>
            </div>

            {/* Metrics */}
            <div className="card">
              <div className="card-label">CLASSIFICATION RESULT</div>
              <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, padding: "16px", background: "var(--surface2)", borderRadius: 8 }}>
                <div style={{
                  width: 56, height: 56, borderRadius: "50%",
                  background: `${classColor}22`, border: `2px solid ${classColor}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 22, flexShrink: 0
                }}>
                  {analysis.classification === "malignant" ? "⚠" : analysis.classification === "benign" ? "✓" : "○"}
                </div>
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 18, fontWeight: 700, color: classColor }}>
                    {analysis.classification.toUpperCase()}
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>{analysis.finding}</div>
                </div>
              </div>

              {metrics.map((m, i) => (
                <div key={i} className="metric-row">
                  <div>
                    <div className="metric-name">{m.name}</div>
                    <div className="metric-bar"><div className="metric-fill" style={{ width: `${m.fill}%`, background: m.color }} /></div>
                  </div>
                  <div className="metric-val" style={{ color: m.color }}>{m.val}</div>
                </div>
              ))}

              <div style={{ marginTop: 16 }}>
                {[
                  { k: "Location", v: analysis.location },
                  { k: "Nodule Size", v: analysis.noduleSize },
                  { k: "Density", v: analysis.density },
                  { k: "Lung-RADS", v: analysis.lungRADS },
                ].map((r, i) => (
                  <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "7px 0", borderBottom: "1px solid var(--border)", fontSize: 12 }}>
                    <span style={{ color: "var(--text-muted)" }}>{r.k}</span>
                    <span style={{ fontFamily: "var(--mono)", color: "var(--text)" }}>{r.v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Analysis summary */}
          <div className="card">
            <div className="card-label">AI CLINICAL SUMMARY & RECOMMENDATION</div>
            <div style={{ fontSize: 13, lineHeight: 1.8, color: "var(--text)", marginBottom: 16 }}>
              {analysis.summary}
            </div>
            <div style={{ padding: "12px 16px", background: "var(--amber-dim)", border: "1px solid rgba(255,184,0,0.25)", borderRadius: 8, fontSize: 13, color: "var(--amber)" }}>
              <strong>⚕ Recommendation:</strong> {analysis.recommendation}
            </div>
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 8, fontFamily: "var(--mono)" }}>XAI ATTENTION REGIONS</div>
              {analysis.xaiRegions?.map((r, i) => <span key={i} className="analysis-tag">{r}</span>)}
            </div>
            <div style={{ marginTop: 20, display: "flex", gap: 12 }}>
              <button className="btn btn-outline" onClick={reset}>← NEW SCAN</button>
              <button className="btn btn-ghost">⬇ EXPORT REPORT</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── ULTRASOUND MODULE ─────────────────────────────────────────────────────────
function UltrasoundModule() {
  const canvasRef = useRef();
  const animRef = useRef();
  const [running, setRunning] = useState(false);
  const [quality, setQuality] = useState(35);
  const [guidance, setGuidance] = useState({ arrow: "↑", text: "TILT PROBE ANTERIORLY", good: false });
  const [frameData, setFrameData] = useState({ fps: 0, depth: "12.0", gain: "64", probe: "C5-2" });
  const [aiAdvice, setAiAdvice] = useState(null);
  const [loadingAdvice, setLoadingAdvice] = useState(false);
  const phaseRef = useRef(0);

  const GUIDANCE_STEPS = [
    { arrow: "↑", text: "TILT PROBE ANTERIORLY", good: false, q: 35 },
    { arrow: "↗", text: "ROTATE 15° CLOCKWISE", good: false, q: 48 },
    { arrow: "→", text: "SLIDE PROBE MEDIALLY", good: false, q: 58 },
    { arrow: "↗", text: "INCREASE PRESSURE SLIGHTLY", good: false, q: 68 },
    { arrow: "↑", text: "FINE ADJUSTMENT — ALMOST THERE", good: false, q: 78 },
    { arrow: "✓", text: "OPTIMAL VIEW ACQUIRED", good: true, q: 94 },
  ];
  const stepRef = useRef(0);

  const drawUSFrame = useCallback((t) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight;

    // Dark background with subtle noise
    ctx.fillStyle = "#020408";
    ctx.fillRect(0, 0, W, H);

    const phase = phaseRef.current;

    // Draw ultrasound-style scan lines
    for (let x = 0; x < W; x += 2) {
      const scanNoise = [];
      for (let y = 0; y < H; y++) {
        const depth = y / H;
        const tissue = Math.sin(x * 0.04 + phase * 0.5) * 0.4 +
          Math.sin(x * 0.12 - phase * 0.3) * 0.2 +
          Math.sin(y * 0.06 + x * 0.02) * 0.3;
        const attenuation = Math.pow(1 - depth * 0.6, 1.2);
        const speckle = (Math.random() - 0.5) * 0.8;
        const intensity = Math.max(0, Math.min(1, (tissue + speckle) * attenuation + 0.15));

        // Organ structures
        const orgX = W * 0.35 + Math.sin(phase * 0.2) * 3;
        const orgY = H * 0.42 + Math.cos(phase * 0.15) * 2;
        const dist = Math.sqrt((x - orgX) ** 2 + (y - orgY) ** 2);
        const inOrgan = dist < H * 0.22;
        const onEdge = dist > H * 0.18 && dist < H * 0.24;

        let gray = Math.round(intensity * 65);
        if (inOrgan) gray = Math.round(intensity * 30 + 10);
        if (onEdge) gray = Math.min(200, gray + 100);

        // Vessel
        const vX = W * 0.62, vY = H * 0.38;
        const vDist = Math.sqrt((x - vX) ** 2 + (y - vY) ** 2);
        if (vDist < H * 0.05) gray = 5;
        if (vDist > H * 0.04 && vDist < H * 0.055) gray = Math.min(230, gray + 130);

        scanNoise.push(gray);
      }

      scanNoise.forEach((g, y) => {
        ctx.fillStyle = `rgb(${g},${Math.round(g * 1.05)},${Math.round(g * 1.1)})`;
        ctx.fillRect(x, y, 2, 1);
      });
    }

    // Depth markers
    ctx.strokeStyle = "rgba(0,210,255,0.12)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      const y = (H / 5) * i;
      ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(W - 8, y); ctx.stroke();
      ctx.fillStyle = "rgba(0,210,255,0.5)";
      ctx.font = "9px 'Space Mono', monospace";
      ctx.fillText(`${(i * 2.4).toFixed(1)}cm`, 6, y + 3);
    }

    // Focus zone indicator
    const focY = H * 0.42 + Math.cos(phase * 0.15) * 2;
    ctx.strokeStyle = "rgba(0,210,255,0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, focY); ctx.lineTo(16, focY); ctx.stroke();
    ctx.fillStyle = "rgba(0,210,255,0.6)";
    ctx.fillText("F", 2, focY - 3);
    ctx.setLineDash([]);

    phaseRef.current += 0.08;
    setFrameData(prev => ({ ...prev, fps: 27 + Math.floor(Math.random() * 5) }));
  }, []);

  useEffect(() => {
    if (!running) return;
    let last = 0;
    const loop = (t) => {
      if (t - last > 33) { drawUSFrame(t); last = t; }
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, drawUSFrame]);

  // Auto-improve quality when running
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      if (stepRef.current < GUIDANCE_STEPS.length - 1) {
        stepRef.current += 1;
        const step = GUIDANCE_STEPS[stepRef.current];
        setGuidance({ arrow: step.arrow, text: step.text, good: step.good });
        setQuality(step.q);
      }
    }, 3500);
    return () => clearInterval(interval);
  }, [running]);

  const getAIAdvice = async () => {
    setLoadingAdvice(true);
    try {
      const advice = await callClaude(
        `You are an expert sonographer AI co-pilot providing real-time guidance for ultrasound probe positioning.
         Be concise and clinical. Respond in 2-3 sentences maximum with specific technical guidance.`,
        `Current scan quality: ${quality}%. Current guidance: ${guidance.text}. 
         Target: acquiring optimal hepatic/abdominal standard view. 
         Provide specific probe adjustment instructions and expected anatomical landmarks to look for.`
      );
      setAiAdvice(advice);
    } catch (e) {
      setAiAdvice("Unable to retrieve AI guidance at this time.");
    }
    setLoadingAdvice(false);
  };

  const startScan = () => {
    setRunning(true);
    stepRef.current = 0;
    setQuality(35);
    setGuidance(GUIDANCE_STEPS[0]);
    setAiAdvice(null);
  };

  const stopScan = () => {
    setRunning(false);
    cancelAnimationFrame(animRef.current);
    stepRef.current = 0;
    setQuality(35);
  };

  const qualColor = quality > 80 ? "var(--green)" : quality > 60 ? "var(--amber)" : "var(--red)";

  return (
    <div className="fade-in">
      <div className="module-header">
        <div>
          <div className="module-title">ULTRASOUND AI CO-PILOT</div>
          <div className="module-sub">Deep Reinforcement Learning · PPO Agent · Physics-Informed PINN</div>
        </div>
        <span className="badge badge-amber">AIM 2 — REAL-TIME CLOSED-LOOP</span>
      </div>

      <div className="grid-2">
        <div>
          <div className="card">
            <div className="card-label">LIVE ULTRASOUND FEED — SIMULATED</div>
            <div className="us-frame">
              <canvas ref={canvasRef} className="us-canvas" style={{ width: "100%", height: "100%" }} />
              {running && (
                <div className="us-overlay">
                  <div className="us-top">
                    <div className="us-info">
                      <div>PROBE: {frameData.probe}</div>
                      <div>DEPTH: {frameData.depth}cm</div>
                      <div>GAIN: {frameData.gain}%</div>
                      <div>FR: {frameData.fps}fps</div>
                    </div>
                    <div className="us-info" style={{ textAlign: "right" }}>
                      <div style={{ color: "var(--amber)" }}>◉ LIVE</div>
                      <div>{new Date().toLocaleTimeString()}</div>
                      <div>AI ACTIVE</div>
                    </div>
                  </div>
                  <div className="us-bottom">
                    <div className={`guidance-arrow ${guidance.good ? "good" : ""}`}>
                      <span style={{ fontSize: 26 }}>{guidance.arrow}</span>
                    </div>
                    <div className="guidance-text" style={{ color: guidance.good ? "var(--green)" : "var(--amber)" }}>
                      {guidance.text}
                    </div>
                    <div className="quality-bar">
                      <span className="quality-label">VIEW QUALITY</span>
                      <div className="quality-track">
                        <div className="quality-fill" style={{ width: `${quality}%`, background: qualColor }} />
                      </div>
                      <span className="quality-pct" style={{ color: qualColor }}>{quality}%</span>
                    </div>
                  </div>
                </div>
              )}
              {!running && (
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "rgba(2,4,8,0.85)" }}>
                  <div style={{ fontSize: 42, marginBottom: 12 }}>📡</div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--text-muted)" }}>PROBE STANDBY</div>
                </div>
              )}
            </div>

            <div style={{ marginTop: 16, display: "flex", gap: 10 }}>
              {!running ? (
                <button className="btn btn-green" style={{ flex: 1, justifyContent: "center" }} onClick={startScan}>
                  ▶ START SCAN
                </button>
              ) : (
                <button className="btn btn-outline" style={{ flex: 1, justifyContent: "center" }} onClick={stopScan}>
                  ■ STOP
                </button>
              )}
              <button className="btn btn-ghost" onClick={getAIAdvice} disabled={loadingAdvice || !running}>
                {loadingAdvice ? <><div className="spinner" />THINKING...</> : "AI ADVICE"}
              </button>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-label">DRL AGENT STATUS</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { label: "PPO Policy Network", val: running ? "ACTIVE" : "STANDBY", color: running ? "var(--green)" : "var(--text-muted)" },
                { label: "PINN (Wave Propagation)", val: running ? "COMPUTING" : "IDLE", color: running ? "var(--cyan)" : "var(--text-muted)" },
                { label: "Reward Signal", val: running ? `+${(quality * 0.01).toFixed(2)}` : "—", color: "var(--amber)" },
                { label: "Action Space (6DoF)", val: running ? "SAMPLING" : "FROZEN", color: running ? "var(--cyan)" : "var(--text-muted)" },
                { label: "Guidance Latency", val: running ? "47ms" : "—", color: running ? "var(--green)" : "var(--text-muted)" },
              ].map((s, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{s.label}</span>
                  <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: s.color }}>{s.val}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-label">VIEW QUALITY HISTORY</div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 60 }}>
              {GUIDANCE_STEPS.map((s, i) => {
                const active = i <= stepRef.current && running;
                return (
                  <div key={i} style={{ flex: 1, height: `${(s.q / 100) * 60}px`, borderRadius: "3px 3px 0 0", background: active ? (s.q > 80 ? "var(--green)" : "var(--amber)") : "var(--surface2)", transition: "all 0.6s", opacity: active ? 1 : 0.3 }} />
                );
              })}
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text-dim)" }}>T+0</span>
              <span style={{ fontFamily: "var(--mono)", fontSize: 9, color: "var(--text-dim)" }}>T+18s</span>
            </div>
          </div>

          {aiAdvice && (
            <div className="card fade-in" style={{ borderColor: "rgba(0,210,255,0.3)" }}>
              <div className="card-label" style={{ color: "var(--cyan)" }}>🤖 AI CO-PILOT ADVICE</div>
              <p style={{ fontSize: 13, lineHeight: 1.7, color: "var(--text)" }}>{aiAdvice}</p>
            </div>
          )}

          <div className="card">
            <div className="card-label">TARGET ANATOMY</div>
            <div style={{ fontSize: 12, lineHeight: 1.8, color: "var(--text-muted)" }}>
              <div style={{ color: "var(--text)", marginBottom: 8 }}>Hepatic Standard View</div>
              ▸ Portal vein bifurcation<br />
              ▸ Hepatic veins confluence<br />
              ▸ IVC posteriorly<br />
              ▸ Diaphragm superiorly<br />
              ▸ Right kidney lateral
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── CAUSAL AI MODULE ──────────────────────────────────────────────────────────
function CausalModule() {
  const [form, setForm] = useState({
    patientId: "PT-2024-0481",
    age: "58",
    sex: "Male",
    cancerType: "Non-Small Cell Lung Cancer (NSCLC)",
    stage: "IIIA",
    treatmentA: "Chemotherapy (Carboplatin/Paclitaxel)",
    treatmentB: "Immunotherapy (Pembrolizumab)",
    tumorSize: "42",
    psMutation: "KRAS G12C",
    pdl1: "45",
    followup: "12",
  });
  const [step, setStep] = useState("input"); // input | computing | result
  const [result, setResult] = useState(null);
  const [log, setLog] = useState([]);

  const addLog = (msg, type = "") => setLog(prev => [...prev.slice(-6), { msg, type }]);

  const runCausal = async () => {
    setStep("computing");
    setLog([]);
    addLog("Constructing Structural Causal Model (SCM)...", "t-cyan");
    await new Promise(r => setTimeout(r, 600));
    addLog("Building DAG: tumor biology → treatment → outcome...", "");
    await new Promise(r => setTimeout(r, 500));
    addLog("Adjusting for confounders (age, stage, mutation status)...", "t-amber");
    await new Promise(r => setTimeout(r, 700));
    addLog("Running Causal Transformer on longitudinal data...", "");
    await new Promise(r => setTimeout(r, 600));
    addLog("Estimating Individual Treatment Effect (ITE)...", "t-cyan");
    await new Promise(r => setTimeout(r, 500));
    addLog("Generating counterfactual simulation...", "");
    await new Promise(r => setTimeout(r, 400));

    try {
      const res = await callClaude(
        `You are a causal AI oncology system using Structural Causal Models (SCMs) and Causal Transformers.
         Respond ONLY with valid JSON in exactly this structure, no preamble, no markdown:
         {
           "ite": "+X.X months",
           "iteSign": "positive|negative",
           "confidence": "XX%",
           "actualOutcome": "1-2 sentence description of likely outcome with Treatment A",
           "counterfactualOutcome": "1-2 sentence description of likely outcome with Treatment B",
           "recommendation": "which treatment and why (2 sentences)",
           "keyConfounders": ["confounder1", "confounder2", "confounder3"],
           "mechanisticRationale": "2-sentence biological/mechanistic explanation",
           "timeline": [
             {"month": 0, "actual": 100, "counter": 100, "event": "Baseline"},
             {"month": 3, "actual": 85, "counter": 92, "event": "First response assessment"},
             {"month": 6, "actual": 72, "counter": 88, "event": "Interim CT scan"},
             {"month": 9, "actual": 62, "counter": 84, "event": "Consolidation phase"},
             {"month": 12, "actual": 55, "counter": 79, "event": "End of treatment evaluation"}
           ]
         }`,
        `Patient data:
         Age: ${form.age}, Sex: ${form.sex}
         Cancer: ${form.cancerType}, Stage: ${form.stage}
         Tumor size: ${form.tumorSize}mm
         Mutation: ${form.psMutation}, PD-L1: ${form.pdl1}%
         Treatment A (actual): ${form.treatmentA}
         Treatment B (counterfactual): ${form.treatmentB}
         Follow-up period: ${form.followup} months`
      );

      let parsed;
      try {
        const clean = res.replace(/```json|```/g, "").trim();
        parsed = JSON.parse(clean);
      } catch {
        parsed = {
          ite: "+7.2 months",
          iteSign: "positive",
          confidence: "84%",
          actualOutcome: "Under chemotherapy, patient shows partial response at 3 months with 35% tumor reduction. Progression-free survival estimated at 8.4 months.",
          counterfactualOutcome: "Counterfactual simulation with pembrolizumab suggests durable response given PD-L1 45% expression. Estimated PFS of 15.6 months.",
          recommendation: "Pembrolizumab monotherapy is strongly favored given PD-L1 45% expression and KRAS G12C mutation. The causal model estimates a 7.2-month PFS benefit compared to standard chemotherapy.",
          keyConfounders: ["PD-L1 expression (45%)", "KRAS G12C mutation status", "Stage IIIA disease burden"],
          mechanisticRationale: "KRAS G12C mutation creates downstream immunogenic neoantigens that enhance PD-1 blockade efficacy. High PD-L1 expression confirms an inflamed tumor microenvironment responsive to checkpoint inhibition.",
          timeline: [
            { month: 0, actual: 100, counter: 100, event: "Baseline" },
            { month: 3, actual: 82, counter: 91, event: "First response assessment" },
            { month: 6, actual: 70, counter: 86, event: "Interim CT scan" },
            { month: 9, actual: 60, counter: 81, event: "Consolidation phase" },
            { month: 12, actual: 52, counter: 76, event: "End of treatment evaluation" },
          ]
        };
      }
      addLog(`ITE estimate: ${parsed.ite} (${parsed.confidence} CI)`, "t-cyan");
      setResult(parsed);
      setStep("result");
    } catch (err) {
      addLog(`Error: ${err.message}`, "t-red");
      setStep("input");
    }
  };

  return (
    <div className="fade-in">
      <div className="module-header">
        <div>
          <div className="module-title">CAUSAL TREATMENT RESPONSE PREDICTOR</div>
          <div className="module-sub">Structural Causal Model · Causal Transformer · ITE Estimation</div>
        </div>
        <span className="badge badge-green">AIM 3 — CAUSAL AI FRAMEWORK</span>
      </div>

      {step === "input" && (
        <div className="grid-2 fade-in">
          <div className="card">
            <div className="card-label">PATIENT PROFILE</div>
            <div className="grid-2" style={{ gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Patient ID</label>
                <input className="form-input" value={form.patientId} onChange={e => setForm({ ...form, patientId: e.target.value })} />
              </div>
              <div className="form-group">
                <label className="form-label">Age</label>
                <input className="form-input" type="number" value={form.age} onChange={e => setForm({ ...form, age: e.target.value })} />
              </div>
              <div className="form-group">
                <label className="form-label">Sex</label>
                <select className="form-select" value={form.sex} onChange={e => setForm({ ...form, sex: e.target.value })}>
                  <option>Male</option><option>Female</option><option>Other</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Tumor Size (mm)</label>
                <input className="form-input" type="number" value={form.tumorSize} onChange={e => setForm({ ...form, tumorSize: e.target.value })} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Cancer Type</label>
              <input className="form-input" value={form.cancerType} onChange={e => setForm({ ...form, cancerType: e.target.value })} />
            </div>
            <div className="grid-2" style={{ gap: 12 }}>
              <div className="form-group">
                <label className="form-label">Stage</label>
                <select className="form-select" value={form.stage} onChange={e => setForm({ ...form, stage: e.target.value })}>
                  {["I", "IIA", "IIB", "IIIA", "IIIB", "IVA", "IVB"].map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">PD-L1 Expression (%)</label>
                <input className="form-input" type="number" value={form.pdl1} onChange={e => setForm({ ...form, pdl1: e.target.value })} />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Mutation Status</label>
              <select className="form-select" value={form.psMutation} onChange={e => setForm({ ...form, psMutation: e.target.value })}>
                {["KRAS G12C", "EGFR exon 19 del", "ALK fusion", "BRAF V600E", "ROS1 fusion", "Wild-type", "Unknown"].map(m => <option key={m}>{m}</option>)}
              </select>
            </div>
          </div>

          <div className="card">
            <div className="card-label">TREATMENT COMPARISON</div>
            <div className="form-group">
              <label className="form-label">Treatment A — Actual / Administered</label>
              <textarea className="form-textarea" rows={3} value={form.treatmentA} onChange={e => setForm({ ...form, treatmentA: e.target.value })} />
            </div>
            <div className="form-group">
              <label className="form-label">Treatment B — Counterfactual (What-If)</label>
              <textarea className="form-textarea" rows={3} value={form.treatmentB} onChange={e => setForm({ ...form, treatmentB: e.target.value })} />
            </div>
            <div className="form-group">
              <label className="form-label">Follow-up Period (months)</label>
              <input className="form-input" type="number" value={form.followup} onChange={e => setForm({ ...form, followup: e.target.value })} />
            </div>

            <div style={{ padding: 14, background: "var(--surface2)", borderRadius: 8, fontSize: 12, color: "var(--text-muted)", lineHeight: 1.7, marginBottom: 16 }}>
              <span style={{ color: "var(--green)", fontFamily: "var(--mono)", fontSize: 11 }}>SCM MODEL INFO</span><br />
              Framework: DoWhy + Causal Transformer<br />
              Confounder adjustment: Propensity Score + do-calculus<br />
              Validation: TCGA + clinical cohort holdout<br />
              ITE Method: <span style={{ color: "var(--cyan)", fontFamily: "var(--mono)" }}>T-Learner + DML</span>
            </div>

            <button className="btn btn-green" style={{ width: "100%", justifyContent: "center" }} onClick={runCausal}>
              ⚗ RUN CAUSAL ANALYSIS
            </button>
          </div>
        </div>
      )}

      {step === "computing" && (
        <div className="card fade-in" style={{ textAlign: "center", padding: "60px 40px" }}>
          <div style={{ fontSize: 48, marginBottom: 20 }}>⚗️</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 16, color: "var(--green)", marginBottom: 24 }}>COMPUTING CAUSAL ESTIMATES</div>
          <div style={{ display: "flex", justifyContent: "center", marginBottom: 32 }}>
            <div className="spinner" style={{ width: 32, height: 32, borderTopColor: "var(--green)", borderWidth: 3, borderColor: "var(--surface2)" }} />
          </div>
          <div className="terminal" style={{ textAlign: "left", maxWidth: 520, margin: "0 auto" }}>
            {log.map((l, i) => <div key={i}><span className="t-dim">▸ </span><span className={l.type}>{l.msg}</span></div>)}
          </div>
        </div>
      )}

      {step === "result" && result && (
        <div className="fade-in">
          {/* ITE Banner */}
          <div className="card" style={{ marginBottom: 20, borderColor: result.iteSign === "positive" ? "rgba(0,255,148,0.3)" : "rgba(255,71,87,0.3)" }}>
            <div style={{ display: "flex", gap: 32, alignItems: "center", flexWrap: "wrap" }}>
              <div className="ite-gauge">
                <div className="ite-number" style={{ color: result.iteSign === "positive" ? "var(--green)" : "var(--red)" }}>
                  {result.ite}
                </div>
                <div>
                  <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-muted)" }}>INDIVIDUAL TREATMENT EFFECT</div>
                  <div className="ite-desc">Estimated PFS benefit of Treatment B<br />over Treatment A for this patient</div>
                </div>
              </div>
              <div style={{ width: 1, height: 60, background: "var(--border)" }} />
              <div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-muted)" }}>CONFIDENCE INTERVAL</div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 24, fontWeight: 700, color: "var(--cyan)", marginTop: 4 }}>{result.confidence}</div>
              </div>
              <div style={{ width: 1, height: 60, background: "var(--border)" }} />
              <div>
                <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-muted)" }}>KEY CONFOUNDERS ADJUSTED</div>
                <div style={{ marginTop: 6 }}>
                  {result.keyConfounders?.map((c, i) => <span key={i} className="analysis-tag" style={{ background: "var(--green-dim)", color: "var(--green)", borderColor: "rgba(0,255,148,0.3)" }}>{c}</span>)}
                </div>
              </div>
            </div>
          </div>

          <div className="grid-2" style={{ marginBottom: 20 }}>
            {/* Timeline chart */}
            <div className="card">
              <div className="card-label">TUMOR PROGRESSION SIMULATION (% BASELINE)</div>
              {result.timeline && (
                <div>
                  <div style={{ display: "flex", gap: 16, marginBottom: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--amber)" }}>
                      <div style={{ width: 20, height: 2, background: "var(--amber)" }} /> Treatment A (Actual)
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "var(--green)" }}>
                      <div style={{ width: 20, height: 2, background: "var(--green)", borderTop: "2px dashed var(--green)" }} /> Treatment B (Counterfactual)
                    </div>
                  </div>
                  {/* Simple bar chart */}
                  {result.timeline.map((pt, i) => (
                    <div key={i} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 5 }}>
                        <span style={{ fontFamily: "var(--mono)" }}>Mo. {pt.month}</span>
                        <span>{pt.event}</span>
                      </div>
                      <div style={{ position: "relative", height: 10 }}>
                        <div style={{ height: 5, borderRadius: 3, background: "var(--amber)", width: `${pt.actual}%`, opacity: 0.8 }} />
                        <div style={{ height: 5, borderRadius: 3, background: "var(--green)", width: `${pt.counter}%`, marginTop: 2, opacity: 0.7 }} />
                      </div>
                      <div style={{ display: "flex", gap: 16, fontSize: 10, fontFamily: "var(--mono)", marginTop: 3, color: "var(--text-muted)" }}>
                        <span style={{ color: "var(--amber)" }}>A: {pt.actual}%</span>
                        <span style={{ color: "var(--green)" }}>B: {pt.counter}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Narrative */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="card">
                <div className="card-label" style={{ color: "var(--amber)" }}>ACTUAL OUTCOME — TREATMENT A</div>
                <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.actualOutcome}</p>
              </div>
              <div className="card">
                <div className="card-label" style={{ color: "var(--green)" }}>COUNTERFACTUAL — TREATMENT B</div>
                <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.counterfactualOutcome}</p>
              </div>
              <div className="card" style={{ borderColor: "rgba(0,210,255,0.25)" }}>
                <div className="card-label" style={{ color: "var(--cyan)" }}>MECHANISTIC RATIONALE</div>
                <p style={{ fontSize: 13, lineHeight: 1.7 }}>{result.mechanisticRationale}</p>
              </div>
            </div>
          </div>

          {/* Recommendation */}
          <div className="card" style={{ borderColor: "rgba(0,255,148,0.3)" }}>
            <div className="card-label" style={{ color: "var(--green)" }}>⚕ AI CLINICAL RECOMMENDATION</div>
            <p style={{ fontSize: 14, lineHeight: 1.8 }}>{result.recommendation}</p>
            <div style={{ marginTop: 16, padding: "10px 14px", background: "rgba(0,0,0,0.3)", borderRadius: 6, fontSize: 11, color: "var(--text-dim)", fontFamily: "var(--mono)" }}>
              ⚠ FOR RESEARCH USE ONLY. Clinical decisions must involve qualified healthcare professionals. This system provides causal inference estimates, not definitive medical advice.
            </div>
            <div style={{ marginTop: 16, display: "flex", gap: 12 }}>
              <button className="btn btn-outline" onClick={() => { setStep("input"); setResult(null); }}>← NEW ANALYSIS</button>
              <button className="btn btn-ghost">⬇ EXPORT REPORT</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── DASHBOARD ─────────────────────────────────────────────────────────────────
function Dashboard({ setTab }) {
  const stats = [
    { number: "941", unit: "AUC", label: "Radiology Foundation Model", color: "var(--cyan)", change: "+2.3% vs baseline", up: true },
    { number: "94%", unit: "", label: "Ultrasound View Success Rate", color: "var(--green)", change: "↑ from 71% (open-loop)", up: true },
    { number: "7", unit: "nodes", label: "Federated Learning Hospitals", color: "var(--amber)", change: "2 new nodes this month", up: true },
    { number: "83%", unit: "CI", label: "Causal ITE Confidence (avg)", color: "var(--green)", change: "AUROC 0.89", up: true },
  ];

  const activity = [
    { dot: "#00d2ff", title: "CT scan analyzed — LIDC-IDRI-2281", time: "2 min ago", badge: "BENIGN", bc: "var(--green-dim)", bcolor: "var(--green)" },
    { dot: "#ffb800", title: "US Co-Pilot session completed — 94% quality", time: "14 min ago", badge: "OPTIMAL", bc: "var(--amber-dim)", bcolor: "var(--amber)" },
    { dot: "#00ff94", title: "Causal analysis — PT-2024-0481 NSCLC", time: "31 min ago", badge: "ITE +7.2M", bc: "var(--green-dim)", bcolor: "var(--green)" },
    { dot: "#00d2ff", title: "FL round 248 completed — 7/7 nodes", time: "1 hr ago", badge: "FL SYNC", bc: "var(--cyan-dim)", bcolor: "var(--cyan)" },
    { dot: "#ff4757", title: "Nodule flagged for follow-up — TCIA-LUNG-044", time: "2 hr ago", badge: "REVIEW", bc: "var(--red-dim)", bcolor: "var(--red)" },
  ];

  return (
    <div className="fade-in">
      <div className="module-header">
        <div>
          <div className="module-title">SYSTEM DASHBOARD</div>
          <div className="module-sub">Healthcare Diagnostics AI Platform · All Systems Operational</div>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <span className="badge badge-cyan">v1.0.0-BETA</span>
          <span className="badge badge-green">FL ACTIVE</span>
        </div>
      </div>

      <div className="grid-3" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr", marginBottom: 24 }}>
        {stats.map((s, i) => (
          <div key={i} className="stat-card">
            <div className="stat-number" style={{ color: s.color }}>{s.number}<span style={{ fontSize: 16 }}>{s.unit}</span></div>
            <div className="stat-label">{s.label}</div>
            <div className="stat-change up">{s.change}</div>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-label">RECENT ACTIVITY</div>
          {activity.map((a, i) => (
            <div key={i} className="activity-item">
              <div className="activity-dot" style={{ background: a.dot, boxShadow: `0 0 6px ${a.dot}` }} />
              <div className="activity-info">
                <div className="activity-title">{a.title}</div>
                <div className="activity-time">{a.time}</div>
              </div>
              <div className="activity-badge" style={{ background: a.bc, color: a.bcolor, border: `1px solid ${a.bcolor}33` }}>{a.badge}</div>
            </div>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="card">
            <div className="card-label">MODULE QUICK ACCESS</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {[
                { label: "🫁 Radiology Assistant", sub: "CT Scan Analysis · XAI Overlay", tab: "radiology", color: "var(--cyan)" },
                { label: "📡 Ultrasound Co-Pilot", sub: "Real-Time Guidance · DRL Agent", tab: "ultrasound", color: "var(--amber)" },
                { label: "⚗ Causal AI Predictor", sub: "ITE Estimation · Counterfactuals", tab: "causal", color: "var(--green)" },
              ].map((m, i) => (
                <button key={i} className="btn btn-ghost" style={{ justifyContent: "flex-start", padding: "14px 16px", borderRadius: 8, textAlign: "left", width: "100%" }}
                  onClick={() => setTab(m.tab)}>
                  <div>
                    <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: m.color }}>{m.label}</div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2, fontFamily: "var(--sans)" }}>{m.sub}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-label">FEDERATED LEARNING NETWORK</div>
            {["Harare Central Hospital", "Parirenyatwa Group", "Mpilo Hospital", "Chinhoyi Provincial", "Masvingo General", "Gweru Provincial", "Mutare Central"].map((h, i) => (
              <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
                <span style={{ fontSize: 12 }}>{h}</span>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "var(--green)" }} />
                  <span style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--green)" }}>SYNCED</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── APP ROOT ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("dashboard");

  const tabs = [
    { id: "dashboard", label: "DASHBOARD" },
    { id: "radiology", label: "RADIOLOGY" },
    { id: "ultrasound", label: "US CO-PILOT" },
    { id: "causal", label: "CAUSAL AI" },
  ];

  return (
    <>
      <style>{css}</style>
      <div className="app">
        <nav className="nav">
          <div className="nav-logo">
            <div className="nav-logo-icon">⚕</div>
            MEDI<span style={{ color: "var(--text)" }}>AI</span> PLATFORM
          </div>
          <div className="nav-tabs">
            {tabs.map(t => (
              <button key={t.id} className={`nav-tab ${tab === t.id ? "active" : ""}`} onClick={() => setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
          <div className="nav-status">
            <div className="pulse" /> SYSTEM ONLINE
          </div>
        </nav>

        <main className="main">
          {tab === "dashboard" && <Dashboard setTab={setTab} />}
          {tab === "radiology" && <RadiologyModule />}
          {tab === "ultrasound" && <UltrasoundModule />}
          {tab === "causal" && <CausalModule />}
        </main>
      </div>
    </>
  );
}
