import { useEffect, useMemo, useState } from "react";
import MetricCard from "./components/MetricCard";
import LineChart from "./components/LineChart";
import ClassBars from "./components/ClassBars";
import ConfusionHeatmap from "./components/ConfusionHeatmap";
import CompareSlider from "./components/CompareSlider";
import SampleRail from "./components/SampleRail";
import UploadInferencePanel from "./components/UploadInferencePanel";
import FullscreenCompareModal from "./components/FullscreenCompareModal";
import ImageIntelligencePanel from "./components/ImageIntelligencePanel";
import { apiFetch, resolveAssetUrl } from "./api";

function percent(value) {
  return `${((value ?? 0) * 100).toFixed(1)}%`;
}

function takeBest(values = []) {
  return Math.max(...values, 0);
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

function firstSplitKey(dashboard) {
  if (!dashboard || !dashboard.splits || Object.keys(dashboard.splits).length === 0) {
    return "val";
  }

  if (dashboard.splits.val) {
    return "val";
  }

  return Object.keys(dashboard.splits)[0];
}

function App() {
  const [runs, setRuns] = useState([]);
  const [activeRun, setActiveRun] = useState("");
  const [dashboard, setDashboard] = useState(null);
  const [activeSplit, setActiveSplit] = useState("val");
  const [overlayMode, setOverlayMode] = useState("prediction");
  const [reveal, setReveal] = useState(54);
  const [selectedSampleId, setSelectedSampleId] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [fullscreenState, setFullscreenState] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    async function loadRuns() {
      try {
        const payload = await apiFetch("/api/runs");
        setRuns(payload.runs || []);
        const defaultRun = payload.defaultRun || payload.runs?.[0]?.name || "";
        setActiveRun(defaultRun);
      } catch (requestError) {
        setError(
          `${requestError.message}. Start the API server with ".\\.venv\\Scripts\\uvicorn frontend.server.app:app --reload --port 8000".`
        );
        setIsLoading(false);
      }
    }

    loadRuns().catch((requestError) => {
      setError(String(requestError));
      setIsLoading(false);
    });
  }, []);

  useEffect(() => {
    if (!activeRun) {
      return;
    }

    async function loadDashboard(runName) {
      setIsLoading(true);
      setError("");
      try {
        const data = await apiFetch(`/api/dashboard?run_name=${encodeURIComponent(runName)}`);
        const splitKey = firstSplitKey(data);
        setDashboard(data);
        setActiveSplit(splitKey);
        setSelectedSampleId(data.splits[splitKey]?.samples?.[0]?.id ?? "");
        setFullscreenState(null);
      } catch (requestError) {
        setError(requestError.message);
      } finally {
        setIsLoading(false);
      }
    }

    loadDashboard(activeRun).catch((requestError) => {
      setError(String(requestError));
      setIsLoading(false);
    });
  }, [activeRun, refreshKey]);

  const splitEntries = useMemo(() => (dashboard ? Object.entries(dashboard.splits) : []), [dashboard]);
  const splitData = dashboard?.splits?.[activeSplit] ?? (splitEntries[0] ? splitEntries[0][1] : null);

  useEffect(() => {
    const firstSample = splitData?.samples?.[0];
    if (firstSample) {
      setSelectedSampleId(firstSample.id);
    }
    setReveal(54);
    setOverlayMode("prediction");
  }, [activeSplit, splitData]);

  const selectedSample = useMemo(
    () => splitData?.samples?.find((sample) => sample.id === selectedSampleId) ?? splitData?.samples?.[0],
    [selectedSampleId, splitData]
  );

  const selectedRunSummary = useMemo(
    () => runs.find((run) => run.name === activeRun) ?? runs[0] ?? null,
    [activeRun, runs]
  );

  const sampleMetaItems = useMemo(() => {
    if (!selectedSample) {
      return [];
    }

    return [
      { label: "Filename", value: selectedSample.imageInfo?.filename || selectedSample.id },
      selectedSample.imageInfo
        ? { label: "Image size", value: `${selectedSample.imageInfo.width} x ${selectedSample.imageInfo.height}` }
        : null,
      selectedSample.imageInfo ? { label: "File size", value: formatBytes(selectedSample.imageInfo.fileSizeBytes) } : null,
      { label: "Bucket", value: selectedSample.bucket },
    ].filter(Boolean);
  }, [selectedSample]);

  const metrics = splitData?.metrics;
  const peakTrainLoopIoU = takeBest(dashboard?.history?.val_iou);
  const validationBenchmark = selectedRunSummary?.valMeanIoU ?? dashboard?.splits?.val?.metrics?.mean_iou;
  const testBenchmark = selectedRunSummary?.testMeanIoU ?? dashboard?.splits?.test?.metrics?.mean_iou;
  const comparisonSignal =
    activeSplit === "val"
      ? {
          label: "Held-out test IoU",
          value: percent(testBenchmark),
          note: "Full 1002-image test evaluation",
        }
      : {
          label: "Validation IoU",
          value: percent(validationBenchmark),
          note: "Full 317-image validation benchmark",
        };

  const summarySignals = useMemo(
    () => [
      {
        label: "Current split IoU",
        value: percent(metrics?.mean_iou),
        note: `${splitData?.label || "Active"} performance`,
      },
      {
        label: "Pixel accuracy",
        value: percent(metrics?.pixel_accuracy),
        note: "Global segmentation hit rate",
      },
      {
        label: "Tracked epochs",
        value: String(selectedRunSummary?.epochs ?? dashboard?.history?.val_iou?.length ?? 0),
        note: "Saved training history",
      },
      comparisonSignal,
    ],
    [
      comparisonSignal,
      dashboard?.history?.val_iou?.length,
      metrics?.mean_iou,
      metrics?.pixel_accuracy,
      selectedRunSummary?.epochs,
      splitData?.label,
    ]
  );

  if (isLoading && !dashboard) {
    return (
      <main className="page">
        <section className="loading-shell">
          <p className="eyebrow">TerrainScope</p>
          <h1>Loading API and run data...</h1>
        </section>
      </main>
    );
  }

  if (error && !dashboard) {
    return (
      <main className="page">
        <section className="loading-shell">
          <p className="eyebrow">TerrainScope</p>
          <h1>API connection failed</h1>
          <p className="panel-copy">{error}</p>
        </section>
      </main>
    );
  }

  if (dashboard && !splitData) {
    return (
      <main className="page">
        <section className="loading-shell">
          <p className="eyebrow">TerrainScope</p>
          <h1>No evaluation splits found for this run</h1>
          <p className="panel-copy">
            This run has training history, but the API could not find usable evaluation artifacts. Pick another run or rerun
            `test.py` for validation/test outputs.
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="page page-redesign">
      <section className="top-grid">
        <section className="masthead panel reveal">
          <div className="masthead-copy">
            <p className="eyebrow">Off-road segmentation review</p>
            <h1>TerrainScope</h1>
            <p className="hero-text">
              A clean review surface for model runs, validation outputs, and live frame inference. Performance stays visible and image-specific guidance stays next to the scene it describes.
            </p>
            <div className="hero-pills">
              <span className="pill">Run: {dashboard.runName}</span>
              <span className="pill">Model: {dashboard.config.modelName || dashboard.config.backboneName}</span>
              <span className="pill">{splitData?.label} split</span>
            </div>
          </div>

          <div className="summary-strip">
            {summarySignals.map((signal) => (
              <article className="summary-tile" key={signal.label}>
                <span>{signal.label}</span>
                <strong>{signal.value}</strong>
                <p>{signal.note}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="control-panel panel reveal">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Run controls</p>
              <h3>Session selection</h3>
            </div>
          </div>

          <section className="control-card control-card-light">
            <div className="control-card-row">
              <div className="control-label-group">
                <p className="eyebrow">Run</p>
                <div className="run-picker-row">
                  <select value={activeRun} onChange={(event) => setActiveRun(event.target.value)}>
                    {runs.map((run) => (
                      <option value={run.name} key={run.name}>
                        {run.name}
                      </option>
                    ))}
                  </select>
                  <button className="action-button action-button-ghost" onClick={() => setRefreshKey((value) => value + 1)} type="button">
                    Refresh
                  </button>
                </div>
              </div>
            </div>

            <div className="control-card-row">
              <div className="control-label-group">
                <p className="eyebrow">Split</p>
                <div className="split-toggle split-toggle-compact">
                  {splitEntries.map(([key, split]) => (
                    <button
                      className={key === activeSplit ? "split-button split-button-active" : "split-button"}
                      key={key}
                      onClick={() => setActiveSplit(key)}
                      type="button"
                    >
                      <span>{split.label}</span>
                      <strong>{percent(split.metrics.mean_iou)} IoU</strong>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="run-summary-grid run-summary-grid-compact">
              <div>
                <span>Epochs</span>
                <strong>{selectedRunSummary?.epochs ?? "N/A"}</strong>
              </div>
              <div>
                <span>Val IoU</span>
                <strong>{percent(selectedRunSummary?.valMeanIoU)}</strong>
              </div>
              <div>
                <span>Test IoU</span>
                <strong>{percent(selectedRunSummary?.testMeanIoU)}</strong>
              </div>
              <div>
                <span>Train-loop peak</span>
                <strong>{percent(peakTrainLoopIoU)}</strong>
              </div>
            </div>
          </section>
        </section>
      </section>

      <section className="training-panel panel reveal">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Training trace</p>
            <h3>Run progression</h3>
          </div>
          <div className="plot-badge">History synced from saved artifacts</div>
        </div>
        <img src={resolveAssetUrl(dashboard.trainingPlot)} alt="Training metrics plot" />
      </section>

      {error ? <p className="error-banner error-inline">{error}</p> : null}

      <section className="metric-ribbon">
        <MetricCard
          label="Mean IoU"
          value={percent(metrics.mean_iou)}
          hint={`${metrics.num_images} images evaluated`}
          accent="#d76b30"
          delay={50}
        />
        <MetricCard
          label="Mean Dice"
          value={percent(metrics.mean_dice)}
          hint="Region overlap stability"
          accent="#1f8f65"
          delay={120}
        />
        <MetricCard
          label="Pixel Accuracy"
          value={percent(metrics.pixel_accuracy)}
          hint="Global pixel hit rate"
          accent="#2f6fdd"
          delay={190}
        />
        <MetricCard
          label="Average Loss"
          value={metrics.avg_loss?.toFixed(3) ?? "N/A"}
          hint={`${splitData?.label} split`}
          accent="#7a4ad8"
          delay={260}
        />
      </section>

      <UploadInferencePanel runName={activeRun} onOpenFullscreen={(state) => setFullscreenState(state)} />

      <section className="review-section">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Validation review</p>
            <h2>Selected scene breakdown</h2>
          </div>
          <p className="panel-copy">
            Review a real validation or test sample, compare masks, then use the side analysis panel for scene summary and handling suggestions.
          </p>
        </div>

        <section className="focus-layout focus-layout-redesign">
          <CompareSlider
            sample={selectedSample}
            mode={overlayMode}
            reveal={reveal}
            onRevealChange={setReveal}
            onModeChange={setOverlayMode}
            availableModes={[
              { key: "prediction", label: "Prediction" },
              { key: "groundTruth", label: "Ground Truth" },
            ]}
            metaItems={[
              { label: "Sample IoU", value: percent(selectedSample?.iou) },
              { label: "Bucket", value: selectedSample?.bucket || "N/A" },
              selectedSample?.comparison ? { label: "Artifacts", value: "Open composite", href: selectedSample.comparison } : null,
            ].filter(Boolean)}
            headerLabel={`${splitData?.label} sample compare`}
            onOpenFullscreen={() =>
              setFullscreenState({
                title: `${splitData?.label} sample: ${selectedSample?.id}`,
                sample: selectedSample,
                initialMode: overlayMode,
                availableModes: [
                  { key: "prediction", label: "Prediction" },
                  { key: "groundTruth", label: "Ground Truth" },
                ],
                metaItems: sampleMetaItems,
              })
            }
          />

          <div className="focus-side-column">
            <ImageIntelligencePanel
              title={selectedSample?.id || "Selected sample"}
              summary={selectedSample?.analysis?.summary || "No analysis available for this sample."}
              qualityLabel={selectedSample?.analysis?.qualityLabel}
              qualityScore={selectedSample?.analysis?.qualityScore}
              topClasses={selectedSample?.analysis?.topClasses || []}
              suggestions={selectedSample?.analysis?.suggestions || []}
              metaItems={sampleMetaItems}
            />
            <SampleRail
              samples={splitData?.samples || []}
              selectedSampleId={selectedSample?.id}
              onSelect={setSelectedSampleId}
            />
          </div>
        </section>
      </section>

      <section className="analytics-stack">
        <details className="compact-details analytics-card" open>
          <summary>Model Analytics</summary>
          <div className="details-body">
            <section className="two-up">
              <LineChart
                title="Loss decay"
                primary={dashboard.history.train_loss}
                secondary={dashboard.history.val_loss}
                primaryLabel="Train loss"
                secondaryLabel="Val loss"
              />
              <LineChart
                title="IoU lift"
                primary={dashboard.history.train_iou}
                secondary={dashboard.history.val_iou}
                primaryLabel="Train IoU"
                secondaryLabel="Val IoU"
              />
            </section>
            <section className="two-up">
              <ClassBars
                title={`${splitData.label} class IoU`}
                values={metrics.per_class_iou}
                classNames={dashboard.classNames}
                colors={dashboard.classColors}
              />
              <ClassBars
                title={`${splitData.label} class Dice`}
                values={metrics.per_class_dice}
                classNames={dashboard.classNames}
                colors={dashboard.classColors}
              />
            </section>
          </div>
        </details>

        <details className="compact-details analytics-card">
          <summary>Confusion Matrix</summary>
          <div className="details-body">
            <ConfusionHeatmap matrix={metrics.confusion_matrix} classNames={dashboard.classNames} />
          </div>
        </details>
      </section>

      <FullscreenCompareModal
        open={Boolean(fullscreenState)}
        title={fullscreenState?.title}
        sample={fullscreenState?.sample}
        initialMode={fullscreenState?.initialMode}
        availableModes={fullscreenState?.availableModes || []}
        metaItems={fullscreenState?.metaItems || []}
        onClose={() => setFullscreenState(null)}
      />
    </main>
  );
}

export default App;
