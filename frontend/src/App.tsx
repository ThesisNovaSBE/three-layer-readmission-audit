import { useEffect, useRef, useState } from "react";
import { fetchHealth, fetchPatients } from "./api";
import PatientModal from "./components/PatientModal";
import PatientTable from "./components/PatientTable";
import PipelineDiagram from "./components/PipelineDiagram";
import type { HealthStatus, Patient } from "./types";
import "./index.css";

type Tab = "pipeline" | "patients";

const PAGE_SIZE = 50;

export default function App() {
  const [tab, setTab] = useState<Tab>("pipeline");
  const [patients, setPatients] = useState<Patient[]>([]);
  const [total, setTotal] = useState(0);
  const [nextOffset, setNextOffset] = useState(PAGE_SIZE);
  const [search, setSearch] = useState("");
  const [loadingMore, setLoadingMore] = useState(false);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Patient | null>(null);
  const searchMounted = useRef(false);

  // Initial load
  useEffect(() => {
    Promise.all([fetchHealth(), fetchPatients({ limit: PAGE_SIZE, offset: 0 })])
      .then(([h, page]) => {
        setHealth(h);
        setPatients(page.patients);
        setTotal(page.total);
        setNextOffset(PAGE_SIZE);
      })
      .catch(() =>
        setError(
          "Cannot reach the API. Make sure the backend is running:\n  uvicorn api:app --reload --port 8000"
        )
      )
      .finally(() => setLoading(false));
  }, []);

  // Re-fetch when search changes (debounced, skip initial mount)
  useEffect(() => {
    if (!searchMounted.current) {
      searchMounted.current = true;
      return;
    }
    const id = setTimeout(() => {
      fetchPatients({ limit: PAGE_SIZE, offset: 0, q: search })
        .then((page) => {
          setPatients(page.patients);
          setTotal(page.total);
          setNextOffset(PAGE_SIZE);
        })
        .catch(() => {});
    }, 250);
    return () => clearTimeout(id);
  }, [search]);

  function handleLoadMore() {
    setLoadingMore(true);
    fetchPatients({ limit: PAGE_SIZE, offset: nextOffset, q: search })
      .then((page) => {
        setPatients((prev) => [...prev, ...page.patients]);
        setNextOffset((prev) => prev + PAGE_SIZE);
      })
      .finally(() => setLoadingMore(false));
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center">
        <p className="text-slate-400 text-sm">Connecting to API…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-8">
        <div className="bg-white rounded-2xl border border-red-100 shadow-sm p-8 max-w-md w-full text-center">
          <div className="text-2xl mb-3">⚠️</div>
          <h2 className="text-base font-bold text-slate-800 mb-2">API unavailable</h2>
          <pre className="text-xs text-slate-500 bg-slate-50 rounded-lg p-4 text-left whitespace-pre-wrap">
            {error}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {health?.demo_mode && (
        <div className="bg-amber-50 border-b border-amber-200 px-6 py-2 text-xs text-amber-700 text-center">
          Demo mode — no Stage 2 results found. Showing Stage 1 predictions on synthetic data.
        </div>
      )}

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
                {t === "pipeline" ? "Pipeline" : `Patients (${patients.length})`}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6">
        {tab === "pipeline" && (
          <>
            <PipelineDiagram />
            <div className="bg-white rounded-2xl shadow-sm border border-slate-100 p-6 mb-6">
              <h2 className="text-base font-bold text-slate-800 mb-4">How it works</h2>
              <div className="space-y-4 text-sm text-slate-600 leading-relaxed">
                {[
                  {
                    n: "1",
                    color: "bg-blue-100 text-blue-600",
                    title: "Stage 1 — Triage.",
                    body: "An XGBoost model trained on 521,191 MIMIC-IV admissions scores every patient at discharge using structured EHR features (labs, vitals, comorbidities, LOS). Patients above the flag threshold (0.354) are forwarded to Stage 2. The model is tuned for high recall (84.8%) — it is better to over-flag and let Stage 2 prune than to miss a true readmission.",
                  },
                  {
                    n: "2",
                    color: "bg-violet-100 text-violet-600",
                    title: "Stage 2 — Verify.",
                    body: "A fine-tuned Clinical-Longformer (2048-token context) reads the discharge note of each flagged patient. Notes contain richer context than structured data: symptom progression, clinician judgment, discharge plan — signals that predict readmission but are absent from labs and vitals. Training uses focal loss and per-age-group loss weights. After training, Platt scaling calibrates probabilities per age group. Stage 2 achieves AUROC 0.676, recall 90.8%, confirming 35,187 of 43,776 flagged patients.",
                  },
                  {
                    n: "3",
                    color: "bg-emerald-100 text-emerald-600",
                    title: "Stage 3 — Explain.",
                    body: "For any Stage 2-confirmed patient, a clinician can request an on-demand explanation. A local phi4-mini model (via Ollama) synthesises the top SHAP features from Stage 1, the Longformer attention spans from Stage 2, and the score delta to produce a 2–3 sentence clinical narrative explaining why the note agreed or disagreed with the structured risk signal.",
                  },
                ].map(({ n, color, title, body }) => (
                  <div key={n} className="flex gap-3">
                    <span
                      className={`shrink-0 w-6 h-6 rounded-full ${color} font-bold text-xs flex items-center justify-center`}
                    >
                      {n}
                    </span>
                    <div>
                      <strong className="text-slate-800">{title}</strong> {body}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-4 gap-3">
              {[
                { label: "Stage 1 AUROC", value: "0.706", sub: "XGBoost" },
                { label: "Stage 1 Recall", value: "84.8%", sub: "@ thr = 0.354" },
                { label: "Stage 2 AUROC", value: "0.676", sub: "Clinical-Longformer" },
                { label: "Stage 2 Recall", value: "90.8%", sub: "notes cohort" },
              ].map((m) => (
                <div
                  key={m.label}
                  className="bg-white rounded-xl border border-slate-100 p-4 text-center shadow-sm"
                >
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
                {health?.demo_mode
                  ? "Synthetic patients — Stage 1 predictions only · click any row for details"
                  : "Stage 2 confirmed · calibrated per-age-group thresholds · click any row for details"}
              </p>
            </div>
            <PatientTable
              patients={patients}
              total={total}
              search={search}
              onSearchChange={setSearch}
              onLoadMore={handleLoadMore}
              loadingMore={loadingMore}
              onSelect={setSelected}
            />
          </>
        )}
      </main>

      {selected && (
        <PatientModal
          patient={selected}
          artifactLoaded={health?.artifact_loaded ?? false}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
