import { resolveAssetUrl } from "../api";

function SampleRail({ samples, selectedSampleId, onSelect }) {
  return (
    <section className="panel reveal">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Scene library</p>
          <h3>Best and worst samples</h3>
        </div>
        <p className="panel-copy">Selected live from the active run.</p>
      </div>
      <div className="sample-rail">
        {samples.map((sample) => (
          <button
            className={sample.id === selectedSampleId ? "sample-card sample-card-active" : "sample-card"}
            key={sample.id}
            onClick={() => onSelect(sample.id)}
            type="button"
          >
            <img src={resolveAssetUrl(sample.image)} alt={sample.id} />
            <div className="sample-card-meta">
              <strong>{sample.id}</strong>
              <span>{(sample.iou * 100).toFixed(1)}% IoU</span>
              <em>
                {sample.bucket}
                {sample.imageInfo ? ` | ${sample.imageInfo.width}x${sample.imageInfo.height}` : ""}
              </em>
            </div>
          </button>
        ))}
      </div>
    </section>
  );
}

export default SampleRail;
