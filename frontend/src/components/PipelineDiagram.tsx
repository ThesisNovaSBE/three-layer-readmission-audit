// Current (rerun) metrics, hand-copied from the committed result JSONs (the
// frontend bundle cannot import the gitignored models/ files). Source file and
// key given per value; regenerate numbers via docs/THESIS_NUMBERS.md
// (scripts/export_thesis_numbers.py). Last synced 2026-11.
const STAGE1 = {
  cohort: 521_191,            // study cohort (README)
  trainPartition: 416_949,    // cohort minus held-out test 104,242
  auroc: 0.7215,              // xgboost_metrics.json auroc
  recall: 0.352,              // xgboost_metrics.json operating_point.recall (capacity-15%, primary)
  precision: 0.431,           // xgboost_metrics.json operating_point.precision
  threshold: 0.3235,          // xgboost_metrics.json operating_point.threshold
  flaggedNoted: 9_899,        // flagged + noted group (stage2_evaluation.json overall.n)
};

const STAGE2 = {
  auroc: 0.622,               // stage2_evaluation.json overall.auroc (flagged group)
  recall: 0.987,              // stage2_evaluation.json overall.recall
  precision: 0.434,           // stage2_evaluation.json overall.precision
  f2: 0.786,                  // stage2_evaluation.json overall.f2
  confirmed: 9_618,           // stage2_evaluation.json overall tp+fp (4171+5447)
};

// Pipeline vs. matched control at equal alert volume (pipeline_evaluation.json).
const PIPELINE = {
  pipePrecision: 0.439,       // pipeline.full_cohort.precision
  pipeRecall: 0.324,          // pipeline.full_cohort.recall
  ctrlPrecision: 0.444,       // pipeline.control_arm_stage1_matched.precision
  ctrlRecall: 0.328,          // pipeline.control_arm_stage1_matched.recall
};

function StageBox({
  number,
  title,
  subtitle,
  model,
  metrics,
  color,
}: {
  number: string;
  title: string;
  subtitle: string;
  model: string;
  metrics: { label: string; value: string }[];
  color: string;
}) {
  return (
    <div className={`flex-1 rounded-xl border-2 ${color} bg-white p-5 shadow-sm`}>
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs font-bold text-white px-2 py-0.5 rounded-full bg-slate-600">
          Stage {number}
        </span>
        <span className="text-xs text-slate-400">{model}</span>
      </div>
      <h3 className="text-base font-bold text-slate-800 mb-0.5">{title}</h3>
      <p className="text-xs text-slate-500 mb-4">{subtitle}</p>
      <div className="space-y-1.5">
        {metrics.map((m) => (
          <div key={m.label} className="flex justify-between text-xs">
            <span className="text-slate-500">{m.label}</span>
            <span className="font-semibold text-slate-700">{m.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center w-8 shrink-0">
      <div className="text-2xl text-slate-300 font-light">→</div>
    </div>
  );
}

export default function PipelineDiagram() {
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
      <h2 className="text-lg font-bold text-slate-800 mb-1">Screen → Independent Read → Audit Pipeline</h2>
      <p className="text-sm text-slate-500 mb-5">
        Three-stage pipeline for predicting unplanned 30-day hospital readmissions from MIMIC-IV.
      </p>

      <div className="flex gap-2 items-stretch">
        <StageBox
          number="1"
          title="Screen"
          subtitle="Structured EHR data → capacity-constrained flag"
          model="XGBoost"
          color="border-blue-200"
          metrics={[
            { label: "Cohort",            value: `${STAGE1.cohort.toLocaleString()} admissions` },
            { label: "AUROC",             value: STAGE1.auroc.toFixed(4) },
            { label: "Recall @ cap-15%",  value: STAGE1.recall.toFixed(3) },
            { label: "Precision",         value: STAGE1.precision.toFixed(3) },
            { label: "Flagged + noted",   value: STAGE1.flaggedNoted.toLocaleString() },
          ]}
        />
        <Arrow />
        <StageBox
          number="2"
          title="Independent Read"
          subtitle="Discharge notes → independent risk estimate"
          model="Clinical-Longformer"
          color="border-violet-200"
          metrics={[
            { label: "Flagged group",   value: STAGE1.flaggedNoted.toLocaleString() },
            { label: "AUROC",           value: STAGE2.auroc.toFixed(3) },
            { label: "Recall",          value: STAGE2.recall.toFixed(3) },
            { label: "Precision",       value: STAGE2.precision.toFixed(3) },
            { label: "F2",              value: STAGE2.f2.toFixed(3) },
            { label: "Cascade-confirmed", value: `${STAGE2.confirmed.toLocaleString()} (${((STAGE2.confirmed / STAGE1.flaggedNoted) * 100).toFixed(0)}%)` },
          ]}
        />
        <Arrow />
        <StageBox
          number="3"
          title="Audit"
          subtitle="Both scores + note → uphold/override decision"
          model="MedGemma-27B (HF transformers)"
          color="border-emerald-200"
          metrics={[
            { label: "Audited",     value: `${STAGE1.flaggedNoted.toLocaleString()} flagged + noted` },
            { label: "Input",       value: "SHAP + both scores + note" },
            { label: "Output",      value: "Decision + quoted justification" },
            { label: "Pipeline vs control", value: `P ${PIPELINE.pipePrecision.toFixed(3)} vs ${PIPELINE.ctrlPrecision.toFixed(3)}` },
            { label: "(same alert volume)",  value: `R ${PIPELINE.pipeRecall.toFixed(3)} vs ${PIPELINE.ctrlRecall.toFixed(3)}` },
          ]}
        />
      </div>

      <div className="mt-5 grid grid-cols-3 gap-3">
        <div className="bg-blue-50 rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-blue-700">521k</div>
          <div className="text-xs text-blue-500">MIMIC-IV admissions</div>
        </div>
        <div className="bg-violet-50 rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-violet-700">
            {STAGE2.confirmed.toLocaleString()}
          </div>
          <div className="text-xs text-violet-500">Cascade-confirmed (secondary)</div>
        </div>
        <div className="bg-emerald-50 rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-emerald-700">{STAGE2.auroc.toFixed(3)}</div>
          <div className="text-xs text-emerald-500">Stage 2 AUROC</div>
        </div>
      </div>
    </div>
  );
}
