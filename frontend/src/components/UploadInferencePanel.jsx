import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../api";
import CompareSlider from "./CompareSlider";
import ImageIntelligencePanel from "./ImageIntelligencePanel";

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) {
    return "N/A";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

const EMPTY_ANALYSIS = {
  title: "Awaiting upload",
  summary:
    "Upload an RGB frame to generate a terrain summary, the dominant predicted classes, and handling suggestions based on the model's scene breakdown.",
  qualityLabel: "Expected output",
  qualityScore: null,
  topClasses: [],
  suggestions: [
    "Use a wide, front-facing frame so the model can separate horizon, vegetation, and ground surfaces cleanly.",
    "If bushes, rocks, or logs dominate the image, the model will emphasize local obstacle awareness over open-path confidence.",
    "Use the overlay mode after inference to inspect whether the predicted mask follows terrain boundaries instead of texture noise.",
  ],
  metaItems: [
    { label: "Accepted input", value: "PNG, JPG" },
    { label: "Best framing", value: "Forward-facing terrain" },
    { label: "Output", value: "Mask, overlay, suggestions" },
    { label: "Evaluation", value: "Scene composition analysis" },
  ],
};

function UploadInferencePanel({ runName, onOpenFullscreen }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [mode, setMode] = useState("prediction");
  const [reveal, setReveal] = useState(58);

  useEffect(() => {
    setResult(null);
    setFile(null);
    setError("");
    setMode("prediction");
    setReveal(58);
  }, [runName]);

  const compareSample = useMemo(() => {
    if (!result) {
      return null;
    }

    return {
      id: result.imageInfo.filename,
      image: result.images.original,
      prediction: result.images.prediction,
      overlay: result.images.overlay,
      comparison: result.images.comparison,
      bucket: result.imageInfo.dominantClass,
      iou: result.imageInfo.meanConfidence,
    };
  }, [result]);

  const compareMetaItems = useMemo(() => {
    if (!result) {
      return [];
    }

    return [
      { label: "Dimensions", value: `${result.imageInfo.width} x ${result.imageInfo.height}` },
      { label: "File size", value: formatBytes(result.imageInfo.fileSizeBytes) },
      { label: "Dominant class", value: result.imageInfo.dominantClass || "Unknown" },
      { label: "Mean confidence", value: percent(result.imageInfo.meanConfidence) },
      { label: "Inference time", value: `${Math.round(result.imageInfo.inferenceMs)} ms` },
      { label: "Model", value: result.modelInfo.modelName || result.modelInfo.backboneName },
    ];
  }, [result]);

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file || !runName) {
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setIsLoading(true);
    setError("");
    try {
      const payload = await apiFetch(`/api/inference?run_name=${encodeURIComponent(runName)}`, {
        method: "POST",
        body: formData,
      });
      setResult(payload);
      setMode("prediction");
      setReveal(58);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  const analysisPayload = result
    ? {
        title: result.imageInfo.filename,
        summary: result.analysis?.summary || "No analysis available for this upload.",
        qualityLabel: result.analysis?.qualityLabel,
        qualityScore: result.analysis?.qualityScore,
        topClasses: result.analysis?.topClasses || [],
        suggestions: result.analysis?.suggestions || [],
        metaItems: [
          { label: "Run", value: result.runName },
          { label: "Dimensions", value: `${result.imageInfo.width} x ${result.imageInfo.height}` },
          { label: "File size", value: formatBytes(result.imageInfo.fileSizeBytes) },
          { label: "Inference time", value: `${Math.round(result.imageInfo.inferenceMs)} ms` },
          { label: "Dominant class", value: result.imageInfo.dominantClass },
          { label: "Mean confidence", value: percent(result.imageInfo.meanConfidence) },
        ],
      }
    : EMPTY_ANALYSIS;

  return (
    <section className="panel upload-panel upload-panel-redesign reveal">
      <div className="upload-header">
        <div>
          <p className="eyebrow">Live inference</p>
          <h2>Field image analysis</h2>
        </div>
        <p className="panel-copy">
          Upload one off-road frame, inspect the prediction immediately, and read the model's scene summary and practical suggestions beside it.
        </p>
      </div>

      <div className="upload-workbench">
        <div className="upload-main">
          <form className="upload-form upload-form-redesign" onSubmit={handleSubmit}>
            <label className="upload-dropzone upload-dropzone-redesign">
              <input
                accept="image/*"
                className="upload-input"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                type="file"
              />
              <span>{file ? file.name : "Drop or choose an RGB terrain frame"}</span>
              <small>{file ? formatBytes(file.size) : "PNG and JPG are supported. The backend handles resize and inference."}</small>
            </label>
            <button className="action-button" disabled={!file || !runName || isLoading} type="submit">
              {isLoading ? "Running inference..." : "Run inference"}
            </button>
          </form>

          {error ? <p className="error-banner">{error}</p> : null}

          {result ? (
            <CompareSlider
              sample={compareSample}
              mode={mode}
              reveal={reveal}
              onRevealChange={setReveal}
              onModeChange={setMode}
              availableModes={[
                { key: "prediction", label: "Prediction mask" },
                { key: "overlay", label: "Overlay blend" },
              ]}
              metaItems={compareMetaItems}
              headerLabel="Uploaded image compare"
              onOpenFullscreen={() =>
                onOpenFullscreen({
                  title: `Upload: ${result.imageInfo.filename}`,
                  sample: compareSample,
                  initialMode: mode,
                  availableModes: [
                    { key: "prediction", label: "Prediction mask" },
                    { key: "overlay", label: "Overlay blend" },
                  ],
                  metaItems: compareMetaItems,
                })
              }
            />
          ) : (
            <section className="upload-preview-placeholder">
              <div className="upload-preview-frame">
                <div className="upload-preview-copy">
                  <p className="eyebrow">Preview lane</p>
                  <h3>Inference output will appear here</h3>
                  <p>
                    After upload, this area shows the original frame, mask view, overlay view, and fullscreen compare controls.
                  </p>
                </div>
              </div>
            </section>
          )}
        </div>

        <ImageIntelligencePanel
          title={analysisPayload.title}
          summary={analysisPayload.summary}
          qualityLabel={analysisPayload.qualityLabel}
          qualityScore={analysisPayload.qualityScore}
          topClasses={analysisPayload.topClasses}
          suggestions={analysisPayload.suggestions}
          metaItems={analysisPayload.metaItems}
        />
      </div>
    </section>
  );
}

export default UploadInferencePanel;
