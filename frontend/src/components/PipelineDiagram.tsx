// Real metrics from training runs — Stage 1 from full MIMIC-IV, Stage 2 from KISSKI A100 job.
const STAGE1 = {
  nPatients: 521_191,
  auroc: 0.706,
  recall: 0.848,
  precision: 0.256,
  threshold: 0.354,
  notesCohort: 43_776,
};

const STAGE2 = {
  auroc: 0.676,
  recall: 0.908,
  precision: 0.289,
  f2: 0.635,
  confirmed: 35_187,
  pruned: 8_589,
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
      <h2 className="text-lg font-bold text-slate-800 mb-1">Triage-and-Verify Pipeline</h2>
      <p className="text-sm text-slate-500 mb-5">
        Three-stage pipeline for predicting unplanned 30-day hospital readmissions from MIMIC-IV.
      </p>

      <div className="flex gap-2 items-stretch">
        <StageBox
          number="1"
          title="Triage"
          subtitle="Structured EHR data → high-recall flag"
          model="XGBoost"
          color="border-blue-200"
          metrics={[
            { label: "Dataset",          value: `${STAGE1.nPatients.toLocaleString()} admissions` },
            { label: "AUROC",            value: STAGE1.auroc.toFixed(3) },
            { label: "Recall @ thr",     value: STAGE1.recall.toFixed(3) },
            { label: "Precision",        value: STAGE1.precision.toFixed(3) },
            { label: "Notes cohort",     value: STAGE1.notesCohort.toLocaleString() },
          ]}
        />
        <Arrow />
        <StageBox
          number="2"
          title="Verify"
          subtitle="Discharge notes → prune false positives"
          model="Clinical-Longformer"
          color="border-violet-200"
          metrics={[
            { label: "With notes",   value: STAGE1.notesCohort.toLocaleString() },
            { label: "AUROC",        value: STAGE2.auroc.toFixed(3) },
            { label: "Recall",       value: STAGE2.recall.toFixed(3) },
            { label: "Precision",    value: STAGE2.precision.toFixed(3) },
            { label: "F2",           value: STAGE2.f2.toFixed(3) },
            { label: "Confirmed",    value: `${STAGE2.confirmed.toLocaleString()} (${((STAGE2.confirmed / STAGE1.notesCohort) * 100).toFixed(0)}%)` },
          ]}
        />
        <Arrow />
        <StageBox
          number="3"
          title="Explain"
          subtitle="On-demand clinical narrative for one patient"
          model="phi4-mini (Ollama)"
          color="border-emerald-200"
          metrics={[
            { label: "Trigger",     value: "Clinician request" },
            { label: "Input",       value: "SHAP + attention + delta" },
            { label: "Output",      value: "2–3 sentence narrative" },
            { label: "Deployment",  value: "Local (on-device)" },
            { label: "Training",    value: "None — inference only" },
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
          <div className="text-xs text-violet-500">Stage 2 confirmed</div>
        </div>
        <div className="bg-emerald-50 rounded-lg p-3 text-center">
          <div className="text-xl font-bold text-emerald-700">{STAGE2.auroc.toFixed(3)}</div>
          <div className="text-xs text-emerald-500">Stage 2 AUROC</div>
        </div>
      </div>
    </div>
  );
}
