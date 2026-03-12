function shortName(name) {
  return name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function ConfusionHeatmap({ matrix, classNames }) {
  const rowMaxima = matrix.map((row) => Math.max(...row, 1));

  return (
    <section className="panel reveal">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Error map</p>
          <h3>Confusion heatmap</h3>
        </div>
        <p className="panel-copy">Rows are ground truth, columns are predictions.</p>
      </div>
      <div className="heatmap-grid">
        <div className="heatmap-corner">GT / Pred</div>
        {classNames.map((name) => (
          <div className="heatmap-label heatmap-label-top" key={`col-${name}`} title={name}>
            {shortName(name)}
          </div>
        ))}
        {matrix.map((row, rowIndex) => (
          <div className="heatmap-row" key={classNames[rowIndex]}>
            <div className="heatmap-label heatmap-label-side" title={classNames[rowIndex]}>
              {shortName(classNames[rowIndex])}
            </div>
            {row.map((value, colIndex) => {
              const intensity = value / rowMaxima[rowIndex];
              const backgroundColor =
                rowIndex === colIndex
                  ? `rgba(31, 143, 101, ${0.12 + intensity * 0.82})`
                  : `rgba(215, 107, 48, ${0.08 + intensity * 0.55})`;

              return (
                <div
                  className="heatmap-cell"
                  key={`${rowIndex}-${colIndex}`}
                  style={{ backgroundColor }}
                  title={`${classNames[rowIndex]} -> ${classNames[colIndex]}: ${value.toLocaleString()}`}
                >
                  {value > 999 ? `${Math.round(value / 1000)}k` : value}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
  );
}

export default ConfusionHeatmap;
