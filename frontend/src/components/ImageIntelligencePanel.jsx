function percent(value) {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function ImageIntelligencePanel({ title, summary, qualityLabel, qualityScore, topClasses = [], suggestions = [], metaItems = [] }) {
  return (
    <section className="panel intelligence-panel reveal">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Image intelligence</p>
          <h3>{title}</h3>
        </div>
      </div>

      <div className="intelligence-summary">
        <p>{summary}</p>
        {qualityLabel ? (
          <div className="intelligence-quality">
            <span>{qualityLabel}</span>
            <strong>{qualityScore !== null && qualityScore !== undefined ? percent(qualityScore) : "N/A"}</strong>
          </div>
        ) : null}
      </div>

      {topClasses.length ? (
        <div className="intelligence-classes">
          {topClasses.map((entry) => (
            <div className="intelligence-class-card" key={entry.className}>
              <div className="bar-label">
                <span className="swatch" style={{ backgroundColor: entry.color }} />
                <span>{entry.className}</span>
              </div>
              <strong>{percent(entry.ratio)}</strong>
            </div>
          ))}
        </div>
      ) : null}

      {suggestions.length ? (
        <div className="intelligence-suggestions">
          <p className="eyebrow">Suggestions</p>
          <ul>
            {suggestions.map((suggestion) => (
              <li key={suggestion}>{suggestion}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {metaItems.length ? (
        <div className="intelligence-meta">
          {metaItems.map((item) => (
            <div key={item.label}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export default ImageIntelligencePanel;
