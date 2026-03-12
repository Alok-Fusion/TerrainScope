function buildPath(values, width, height, padding) {
  if (!values.length) {
    return "";
  }

  const max = Math.max(...values);
  const min = Math.min(...values);
  const xStep = values.length > 1 ? (width - padding * 2) / (values.length - 1) : 0;
  const yRange = max - min || 1;

  return values
    .map((value, index) => {
      const x = padding + index * xStep;
      const y = height - padding - ((value - min) / yRange) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function LineChart({ title, primary, secondary, primaryLabel, secondaryLabel }) {
  const width = 520;
  const height = 220;
  const padding = 24;

  return (
    <section className="panel chart-panel reveal">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Training curve</p>
          <h3>{title}</h3>
        </div>
        <div className="legend-inline">
          <span><i className="legend-dot legend-dot-primary" />{primaryLabel}</span>
          <span><i className="legend-dot legend-dot-secondary" />{secondaryLabel}</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img" aria-label={title}>
        <defs>
          <linearGradient id="chartGlow" x1="0" x2="1">
            <stop offset="0%" stopColor="#d76b30" />
            <stop offset="100%" stopColor="#1f8f65" />
          </linearGradient>
        </defs>
        <rect x="0" y="0" width={width} height={height} rx="24" className="chart-surface" />
        {[0.2, 0.4, 0.6, 0.8].map((tick) => (
          <line
            key={tick}
            x1={padding}
            x2={width - padding}
            y1={padding + (height - padding * 2) * tick}
            y2={padding + (height - padding * 2) * tick}
            className="chart-grid"
          />
        ))}
        <path d={buildPath(primary, width, height, padding)} className="chart-line chart-line-primary" />
        <path
          d={buildPath(secondary, width, height, padding)}
          className="chart-line chart-line-secondary"
        />
      </svg>
    </section>
  );
}

export default LineChart;
