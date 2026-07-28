import { useCallback, useRef, useState } from "react";

export default function UploadPanel({ onFileSelected, disabled }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(
    (files) => {
      if (!files || files.length === 0) return;
      onFileSelected(files[0]);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`upload-panel ${isDragOver ? "drag-over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragOver(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />
      <div className="upload-icon">▤</div>
      <p className="upload-title">Drop an MRI scan or click to browse</p>
      <p className="upload-subtitle">PNG or JPEG · axial brain slice · max 10MB</p>
    </div>
  );
}
