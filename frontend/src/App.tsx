import { useState } from "react";
import PipelineDiagram from "./components/PipelineDiagram";
import PatientTable from "./components/PatientTable";
import PatientModal from "./components/PatientModal";
import { MOCK_PATIENTS } from "./data/mockPatients";
import type { Patient } from "./types";
import "./index.css";

type Tab = "pipeline" | "patients";

export default function App() {
  const [tab, setTab] = useState<Tab>("pipeline");
  const [selected, setSelected] = useState<Patient | null>(null);

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Nav */}
      <header className="bg-white border-b border-slate-100 sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-6 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-base font-bold text-slate-800 leading-none">
              Readmission Prediction
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">
              LLM-Augmented Triage · Nova SBE M.Sc. Thesis
            </p>
          </div>
          <nav className="flex gap-1">
            {(["pipeline", "patients"] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  tab === t
                    ? "bg-blue-600 text-white"
                    : "text-slate-500 hover:bg-slate-100"
                }`}
              >
                {t === "pipeline" ? "Pipeline" : "Patients"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6">
        {tab === "pipeline" && (
          <>
            <PipelineDiagram />

            {/* How it works */}
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
              <h2 className="text-base font-bold text-slate-800 mb-4">How it works</h2>
              <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                <div className="flex gap-3">
                  <span className="shrink-0 w-6 h-6 rounded-full bg-blue-100 text-blue-600 font-bold text-xs flex items-center justify-center">1</span>
                  <div>
                    <strong className="text-slate-800">Stage 1 — Triage.</strong> An XGBoost model trained on 521,191 MIMIC-IV admissions scores every patient at discharge using structured EHR features (labs, vitals, comorbidities, LOS). Patients above the flag threshold (0.354) are forwarded to Stage 2. The model is tuned for high recall (84.8%) — it is better to over-flag and let Stage 2 prune than to miss a true readmission.
                  </div>
                </div>
                <div className="flex gap-3">
                  <span className="shrink-0 w-6 h-6 rounded-full bg-violet-100 text-violet-600 font-bold text-xs flex items-center justify-center">2</span>
                  <div>
                    <strong className="text-slate-800">Stage 2 — Verify.</strong> A fine-tuned Clinical-Longformer (4096-token context, 2048 used) reads the discharge note of each flagged patient. Notes contain richer clinical context than structured data: symptom progression, clinician judgment, and discharge plan — signals that predict readmission but are absent from labs and vitals. Training uses focal loss and per-age-group loss weights to improve fairness across age bands. After training, Platt scaling calibrates probabilities per age group, and a recall-floor threshold (≥ 0.65 recall) is selected per group. Stage 2 improves precision by +21% (0.256 → 0.309) while retaining 70.9% of true positives in the notes cohort.
                  </div>
                </div>
                <div className="flex gap-3">
                  <span className="shrink-0 w-6 h-6 rounded-full bg-emerald-100 text-emerald-600 font-bold text-xs flex items-center justify-center">3</span>
                  <div>
                    <strong className="text-slate-800">Stage 3 — Explain.</strong> For each Stage 2–confirmed patient, a local generative model (phi4-mini via Ollama) writes a 3–5 sentence plain-language explanation of why the patient is at elevated risk. This explanation is sent to the responsible nurse or physician. Stage 3 performs no training — it is pure inference, triggered per patient.
                  </div>
                </div>
              </div>
            </div>

            {/* Key metrics */}
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: "Stage 1 AUROC", value: "0.706", sub: "XGBoost" },
                { label: "Stage 1 Recall", value: "84.8%", sub: "@ thr = 0.354" },
                { label: "Precision gain", value: "+21%", sub: "Stage 1 → Stage 2" },
                { label: "Stage 2 F2", value: "0.563", sub: "notes cohort" },
              ].map((m) => (
                <div key={m.label} className="bg-white rounded-xl border border-slate-100 p-4 text-center shadow-sm">
                  <div className="text-2xl font-bold text-blue-600">{m.value}</div>
                  <div className="text-xs font-medium text-slate-600 mt-1">{m.label}</div>
                  <div className="text-xs text-slate-400">{m.sub}</div>
                </div>
              ))}
            </div>
          </>
        )}

        {tab === "patients" && (
          <>
            <div className="mb-4">
              <h2 className="text-base font-bold text-slate-800">Patient Dashboard</h2>
              <p className="text-sm text-slate-400 mt-0.5">
                Synthetic representative patients · click any row for Stage 3 explanation
              </p>
            </div>
            <PatientTable patients={MOCK_PATIENTS} onSelect={setSelected} />
          </>
        )}
      </main>

      {selected && (
        <PatientModal patient={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
