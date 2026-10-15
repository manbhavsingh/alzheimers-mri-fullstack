const CLASS_ORDER = [
  "Non-Demented",
  "Very Mild Demented",
  "Mild Demented",
  "Moderate Demented",
];

export default function ResultPanel({ result, onGenerateReport, reportStatus, report, reportError }) {
  const { predicted_class, confidence, probabilities } = result;

  return (
    <div className="result-panel">
      <div className="result-headline">
        <span className="result-eyebrow">PREDICTED STAGE</span>
        <h2>{predicted_class}</h2>
        <span className="result-confidence">
          {(confidence * 100).toFixed(1)}% confidence
        </span>
      </div>

      <div className="probability-list">
        {CLASS_ORDER.map((cls) => {
          const p = probabilities[cls] ?? 0;
          const isPredicted = cls === predicted_class;
          return (
            <div className="probability-row" key={cls}>
              <span className="probability-label">{cls}</span>
              <div className="probability-track">
                <div
                  className={`probability-fill ${isPredicted ? "active" : ""}`}
                  style={{ width: `${p * 100}%` }}
                />
              </div>
              <span className="probability-value">{(p * 100).toFixed(1)}%</span>
            </div>
          );
        })}
      </div>

      <p className="result-note">
        Heatmap shows regions the model weighted most heavily for this
        prediction. This is a research tool, not a diagnostic device.
      </p>

      <div className="report-generator">
        <div className="report-disclaimer">
          AI-generated explanation. Not medical advice. Consult a doctor.
        </div>

        {reportStatus !== "error" && (
          <button
            className="btn-primary report-generate-btn"
            onClick={onGenerateReport}
            disabled={reportStatus === "loading"}
            aria-busy={reportStatus === "loading"}
          >
            {reportStatus === "loading"
              ? "Generating report…"
              : reportStatus === "done"
                ? "Regenerate report"
                : "Generate report"}
          </button>
        )}

        {reportStatus === "loading" && (
          <p className="report-status" role="status">
            Asking Gemini to explain the model output…
          </p>
        )}

        {reportStatus === "error" && (
          <div className="report-error" role="alert">
            <p>{reportError || "We couldn’t generate the report. Please try again."}</p>
            <button className="btn-secondary report-retry-btn" onClick={onGenerateReport}>
              Retry report
            </button>
          </div>
        )}

        {reportStatus === "done" && report && (
          <section className="report-card" aria-live="polite">
            <h3>AI report explainer</h3>
            <div className="report-section">
              <h4>SUMMARY</h4>
              <p>{report.summary}</p>
            </div>
            <div className="report-section">
              <h4>CONFIDENCE NOTE</h4>
              <p>{report.confidence_note}</p>
            </div>
            <div className="report-section">
              <h4>CAVEATS</h4>
              <ul>
                {report.caveats.map((item, index) => <li key={`caveat-${index}`}>{item}</li>)}
              </ul>
            </div>
            <div className="report-section">
              <h4>NEXT STEPS</h4>
              <ul>
                {report.next_steps.map((item, index) => <li key={`step-${index}`}>{item}</li>)}
              </ul>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
