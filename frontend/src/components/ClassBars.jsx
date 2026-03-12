function asPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function ClassBars({ title, values, classNames, colors }) {
  const ranked = classNames
    .map((name, index) => ({
      name,
      value: values[index] ?? 0,
      color: colors[index],
    }))
    .sort((left, right) => right.value - left.value);

  return (
    <section className="panel reveal">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Per-class view</p>
          <h3>{title}</h3>
        </div>
      </div>
      <div className="bars">
        {ranked.map((entry) => (
          <div className="bar-row" key={entry.name}>
            <div className="bar-label">
              <span className="swatch" style={{ backgroundColor: entry.color }} />
              <span>{entry.name}</span>
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{
                  width: `${Math.max(entry.value * 100, 1)}%`,
                  background: `linear-gradient(90deg, ${entry.color}, color-mix(in srgb, ${entry.color} 60%, white))`,
                }}
              />
            </div>
            <strong>{asPercent(entry.value)}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

export default ClassBars;
