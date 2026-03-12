import { useEffect, useState } from "react";
import CompareSlider from "./CompareSlider";

function FullscreenCompareModal({ open, title, sample, initialMode, availableModes, metaItems, onClose }) {
  const [mode, setMode] = useState(initialMode || availableModes?.[0]?.key || "prediction");
  const [reveal, setReveal] = useState(54);

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    setMode(initialMode || availableModes?.[0]?.key || "prediction");
    setReveal(54);

    function handleKeyDown(event) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [availableModes, initialMode, onClose, open, sample]);

  if (!open || !sample) {
    return null;
  }

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div className="modal-shell" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true">
        <div className="modal-header">
          <div>
            <p className="eyebrow">Fullscreen compare</p>
            <h2>{title}</h2>
          </div>
          <button className="modal-close" onClick={onClose} type="button">
            Close
          </button>
        </div>
        <CompareSlider
          sample={sample}
          mode={mode}
          reveal={reveal}
          onRevealChange={setReveal}
          onModeChange={setMode}
          availableModes={availableModes}
          metaItems={metaItems}
          headerLabel="Compare workspace"
          className="compare-panel-modal"
          hideFullscreenButton
        />
      </div>
    </div>
  );
}

export default FullscreenCompareModal;
