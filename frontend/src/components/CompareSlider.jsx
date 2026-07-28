import { useRef, useState, useCallback } from "react";

export default function CompareSlider({ originalSrc, overlaySrc }) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.min(100, Math.max(0, pct)));
  }, []);

  const onPointerDown = (e) => {
    dragging.current = true;
    updateFromClientX(e.clientX);
  };
  const onPointerMove = (e) => {
    if (dragging.current) updateFromClientX(e.clientX);
  };
  const stopDragging = () => {
    dragging.current = false;
  };

  return (
    <div className="compare-slider-wrap">
      <div className="compare-labels">
        <span>RAW SCAN</span>
        <span>GRAD-CAM OVERLAY</span>
      </div>
      <div
        className="compare-slider"
        ref={containerRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={stopDragging}
        onPointerLeave={stopDragging}
        role="slider"
        aria-label="Compare raw MRI scan against Grad-CAM heatmap overlay"
        aria-valuenow={Math.round(position)}
        aria-valuemin={0}
        aria-valuemax={100}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setPosition((p) => Math.max(0, p - 5));
          if (e.key === "ArrowRight") setPosition((p) => Math.min(100, p + 5));
        }}
      >
        <img src={overlaySrc} alt="Grad-CAM heatmap overlay" draggable={false} />
        <div
          className="compare-clip"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        >
          <img src={originalSrc} alt="Raw MRI scan" draggable={false} />
        </div>
        <div className="compare-handle" style={{ left: `${position}%` }}>
          <div className="compare-handle-grip" />
        </div>
      </div>
    </div>
  );
}
