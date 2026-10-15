import { useState } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import CompareSlider from "./components/CompareSlider.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [reportStatus, setReportStatus] = useState("idle"); // idle | loading | done | error
  const [report, setReport] = useState(null);
  const [reportError, setReportError] = useState("");

  const handleFileSelected = async (file) => {
    setStatus("loading");
    setErrorMsg("");
    setResult(null);
    setReportStatus("idle");
    setReport(null);
    setReportError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMsg(err.message || "Something went wrong.");
      setStatus("error");
    }
  };


  const handleGenerateReport = async () => {
    if (!result) return;

    setReportStatus("loading");
    setReport(null);
    setReportError("");

    const note =
      result.gradcam_note ||
      "The prediction response did not include the raw Grad-CAM activation map, so no region-level activation summary is available.";

    try {
      const res = await fetch(`${API_URL}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: result.predicted_class,
          probs: result.probabilities,
          gradcam_note: note,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || "The report service could not generate an explanation.");
      }

      const data = await res.json();
      setReport(data);
      setReportStatus("done");
    } catch (err) {
      setReportError(
        err.message === "Failed to fetch"
          ? "Unable to reach the report service. Check your connection and try again."
          : err.message || "Something went wrong while generating the report."
      );
      setReportStatus("error");
    }
  };

  const reset = () => {
    setStatus("idle");
    setResult(null);
    setErrorMsg("");
    setReportStatus("idle");
    setReport(null);
    setReportError("");
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" />
          <span className="brand-name">NEURIO</span>
        </div>
        <span className="header-tagline">
          Explainable Alzheimer's MRI Staging · ResNet-18 + Grad-CAM
        </span>
      </header>

      <main className="app-main">
        {status === "idle" && (
          <div className="stage-center">
            <UploadPanel onFileSelected={handleFileSelected} />
          </div>
        )}

        {status === "loading" && (
          <div className="stage-center">
            <div className="loading-block">
              <div className="scan-line" />
              <p>Running inference &amp; computing Grad-CAM…</p>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="stage-center">
            <div className="error-block">
              <p className="error-title">Inference failed</p>
              <p className="error-detail">{errorMsg}</p>
              <button className="btn-secondary" onClick={reset}>
                Try again
              </button>
            </div>
          </div>
        )}

        {status === "done" && result && (
          <div className="results-layout">
            <CompareSlider
              originalSrc={result.original_image}
              overlaySrc={result.gradcam_overlay}
            />
            <ResultPanel
              result={result}
              onGenerateReport={handleGenerateReport}
              reportStatus={reportStatus}
              report={report}
              reportError={reportError}
            />
            <button className="btn-secondary new-scan-btn" onClick={reset}>
              ↺ Analyze another scan
            </button>
          </div>
        )}
      </main>
      <footer className="medical-disclaimer" role="note">
        AI-generated explanation. Not medical advice. Consult a doctor.
      </footer>
    </div>
  );
}
