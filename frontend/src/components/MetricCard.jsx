function MetricCard({ label, value, hint, accent, delay = 0 }) {
  return (
    <article
      className="metric-card reveal"
      style={{ "--accent": accent, animationDelay: `${delay}ms` }}
    >
      <p className="eyebrow">{label}</p>
      <h3>{value}</h3>
      <p className="metric-hint">{hint}</p>
    </article>
  );
}

export default MetricCard;
