const CLASS_ORDER = [
  "Non-Demented",
  "Very Mild Demented",
  "Mild Demented",
  "Moderate Demented",
];

export default function ResultPanel({ result }) {
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
    </div>
  );
}
