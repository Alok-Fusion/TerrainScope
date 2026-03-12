import { useMemo } from "react";
import { resolveAssetUrl } from "../api";

function CompareSlider({
  sample,
  mode,
  reveal,
  onRevealChange,
  onModeChange,
  availableModes = [
    { key: "prediction", label: "Prediction" },
    { key: "groundTruth", label: "Ground Truth" },
  ],
  metaItems = [],
  headerLabel = "Live compare",
  onOpenFullscreen,
  hideFullscreenButton = false,
  className = "",
}) {
  const overlayImage = useMemo(() => {
    if (!sample) {
      return "";
    }

    return sample[mode] || sample.prediction;
  }, [mode, sample]);

  const activeMode = useMemo(
    () => availableModes.find((availableMode) => availableMode.key === mode)?.label || "Prediction",
    [availableModes, mode]
  );

  if (!sample) {
    return (
      <section className={`panel compare-panel ${className}`.trim()}>
        <p>No sample loaded.</p>
      </section>
    );
  }

  const fallbackMetaItems = [
    sample.iou !== undefined ? { label: "Score", value: `${(sample.iou * 100).toFixed(1)}%` } : null,
    sample.bucket ? { label: "Bucket", value: sample.bucket } : null,
    sample.comparison ? { label: "Artifacts", value: "Open composite", href: sample.comparison } : null,
  ].filter(Boolean);
  const items = metaItems.length ? metaItems : fallbackMetaItems;

  return (
    <section className={`panel compare-panel reveal ${className}`.trim()}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{headerLabel}</p>
          <h3>{sample.id}</h3>
        </div>
        <div className="compare-toolbar-end">
          <div className="compare-mode">
            {availableModes.map((availableMode) => (
              <button
                className={mode === availableMode.key ? "chip chip-active" : "chip"}
                onClick={() => onModeChange(availableMode.key)}
                type="button"
                key={availableMode.key}
              >
                {availableMode.label}
              </button>
            ))}
          </div>
          {!hideFullscreenButton && onOpenFullscreen ? (
            <button className="action-button action-button-ghost" onClick={onOpenFullscreen} type="button">
              Fullscreen
            </button>
          ) : null}
        </div>
      </div>
      <div className="compare-stage">
        <div className="compare-stage-badges">
          <span className="stage-badge">RGB frame</span>
          <span className="stage-badge stage-badge-accent">{activeMode}</span>
        </div>
        <img className="compare-base" src={resolveAssetUrl(sample.image)} alt={`${sample.id} source`} />
        <div className="compare-overlay" style={{ width: `${reveal}%` }}>
          <img src={resolveAssetUrl(overlayImage)} alt={`${sample.id} overlay`} />
        </div>
        <div className="compare-divider" style={{ left: `${reveal}%` }}>
          <span />
        </div>
      </div>
      <input
        className="compare-slider"
        type="range"
        min="0"
        max="100"
        value={reveal}
        onChange={(event) => onRevealChange(Number(event.target.value))}
      />
      {items.length ? (
        <div className="compare-meta">
          {items.map((item) => (
            <div key={item.label}>
              <p className="eyebrow">{item.label}</p>
              {item.href ? (
                <a href={resolveAssetUrl(item.href)} target="_blank" rel="noreferrer">
                  {item.value}
                </a>
              ) : (
                <strong>{item.value}</strong>
              )}
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default CompareSlider;
