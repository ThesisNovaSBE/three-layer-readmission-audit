import { useState } from "react";
import { explainPatient } from "../api";
import type { ExplanationResult, Patient } from "../types";

// ── Small layout helpers ──────────────────────────────────────────────────────

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between py-2 border-b border-slate-50 last:border-0">
      <span className="text-xs text-slate-400">{label}</span>
      <span className="text-xs font-medium text-slate-700 text-right max-w-[60%]">{value}</span>
    </div>
  );
}

function ScoreGauge({ label, score, color }: { label: string; score: number; color: string }) {
  return (
    <div className="text-center">
      <div className={`text-2xl font-bold ${color}`}>{score.toFixed(3)}</div>
      <div className="text-xs text-slate-400 mt-0.5">{label}</div>
      <div className="mt-2 w-full bg-slate-100 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${color.replace("text-", "bg-")}`}
          style={{ width: `${score * 100}%` }}
        />
      </div>
    </div>
  );
}

// ── Percentile-rank displacement visualisation ──────────────────────────────
// displacement = r2 (Stage 2's percentile rank) − r1 (Stage 1's), range
// [-100, +100]. Not a raw score subtraction — see docs/ARCHITECTURE.md §2.

const DISPLACEMENT_BANDS = [
  { max: -20, label: "Note mitigates risk (Stage 2 ranks it lower)", color: "text-green-700", barColor: "bg-green-400" },
  { max: 20, label: "Concordant", color: "text-slate-500", barColor: "bg-slate-300" },
  { max: Infinity, label: "Note amplifies risk (Stage 2 ranks it higher)", color: "text-red-600", barColor: "bg-red-400" },
] as const;

function displacementInfo(displacement: number) {
  return DISPLACEMENT_BANDS.find((b) => displacement < b.max) ?? DISPLACEMENT_BANDS[DISPLACEMENT_BANDS.length - 1];
}

function DisplacementBar({ displacement }: { displacement: number }) {
  const info = displacementInfo(displacement);
  const clamped = Math.max(-100, Math.min(100, displacement));
  const pct = Math.abs(clamped) / 2;
  const isNeg = clamped < 0;
  return (
    <div className="space-y-1.5">
      <div className="relative w-full h-2 bg-slate-100 rounded-full overflow-hidden">
        {/* centre line */}
        <div className="absolute top-0 bottom-0 w-px bg-slate-300" style={{ left: "50%" }} />
        <div
          className={`absolute h-full ${info.barColor}`}
          style={{ left: isNeg ? `${50 - pct}%` : "50%", width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-400">
        <span>← mitigates</span>
        <span className={`font-semibold ${info.color}`}>{info.label}</span>
        <span>amplifies →</span>
      </div>
    </div>
  );
}

// ── Discordance mode + decision badges ──────────────────────────────────────

const MODE_STYLES: Record<string, string> = {
  CONCORDANT:     "bg-slate-100 text-slate-600",
  NOTE_MITIGATES: "bg-green-100 text-green-700",
  NOTE_AMPLIFIES: "bg-orange-100 text-orange-700",
};

function ModeBadge({ mode }: { mode: string | null }) {
  if (!mode) return null;
  const style = MODE_STYLES[mode] ?? "bg-slate-100 text-slate-500";
  const label = mode.replace("_", " ").replace("NOTE ", "Note ");
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${style}`}>{label}</span>
  );
}

const DECISION_STYLES: Record<string, string> = {
  uphold: "bg-blue-100 text-blue-700",
  override: "bg-red-100 text-red-700",
};

function DecisionBadge({ decision }: { decision: string | null }) {
  if (!decision) return null;
  const style = DECISION_STYLES[decision] ?? "bg-slate-100 text-slate-500";
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full uppercase tracking-wide ${style}`}>
      {decision}
    </span>
  );
}

// ── Explanation panel ─────────────────────────────────────────────────────────

function ExplanationPanel({ result }: { result: ExplanationResult }) {
  return (
    <div className="space-y-4">
      {result.annotation_failed && (
        <div className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          phi4-mini returned an unparseable response — structured fields unavailable. Raw inputs
          shown as-is.
        </div>
      )}

      {/* Percentile-rank displacement */}
      <div className="bg-slate-50 rounded-xl p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Displacement (Stage 2 rank − Stage 1 rank)
          </span>
          <span className={`text-sm font-bold ${displacementInfo(result.displacement).color}`}>
            {result.displacement >= 0 ? "+" : ""}{result.displacement.toFixed(1)}pp
          </span>
        </div>
        <DisplacementBar displacement={result.displacement} />
        <div className="flex justify-between text-xs text-slate-400">
          <span>Stage 1 percentile: {result.r1.toFixed(0)}</span>
          <span>Stage 2 percentile: {result.r2.toFixed(0)}</span>
        </div>
      </div>

      {/* Decision + mode + domain */}
      <div className="flex items-center gap-2 flex-wrap">
        <DecisionBadge decision={result.decision} />
        <ModeBadge mode={result.discordance_mode} />
        {result.primary_clinical_domain && (
          <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full">
            {result.primary_clinical_domain.replace(/_/g, " ")}
          </span>
        )}
      </div>

      {/* SHAP features */}
      {result.top_shap_features.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Top structured risk factors (Stage 1 SHAP)
          </h4>
          <ul className="space-y-1">
            {result.top_shap_features.map((f, i) => (
              <li key={i} className="text-xs text-slate-600 flex gap-2">
                <span className="text-slate-300 shrink-0">•</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Attention sentences */}
      {result.attention_sentences.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
            Key note content (Longformer attention)
          </h4>
          <ol className="space-y-1">
            {result.attention_sentences.map((s, i) => (
              <li key={i} className="text-xs text-slate-600 flex gap-2">
                <span className="text-slate-400 shrink-0 font-mono">[{i + 1}]</span>
                <span>{s}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Supporting quote */}
      {result.supporting_quote && (
        <div>
          <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
            Supporting quote
            {result.quote_verified === true && (
              <span className="text-emerald-600 normal-case font-normal text-xs">✓ verified in note</span>
            )}
            {result.quote_verified === false && (
              <span className="text-amber-600 normal-case font-normal text-xs">⚠ not found verbatim in note</span>
            )}
          </h4>
          <blockquote className="border-l-2 border-slate-200 pl-3 text-xs text-slate-600 italic">
            “{result.supporting_quote}”
          </blockquote>
        </div>
      )}

      {/* Planned return */}
      {result.planned_return && (
        <InfoRow
          label="Planned return (per note)"
          value={result.planned_return.replace(/_/g, " ")}
        />
      )}

      {/* Clinical justification */}
      <div>
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Clinical Justification
        </h4>
        <div className="bg-emerald-50 border border-emerald-100 rounded-xl px-4 py-3 text-sm text-slate-700 leading-relaxed">
          {result.clinical_justification || "(no justification generated)"}
        </div>
      </div>

      <p className="text-xs text-slate-400">
        Generated by a local LLM. For clinical review only — not a diagnosis or recommendation.
      </p>
    </div>
  );
}

// ── Stage 3 section (button → loading → result) ───────────────────────────────

function Stage3Section({
  hadmId,
  artifactLoaded,
}: {
  hadmId: number;
  artifactLoaded: boolean;
}) {
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleExplain() {
    setLoading(true);
    setError(null);
    try {
      const res = await explainPatient(hadmId);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Explanation failed.");
    } finally {
      setLoading(false);
    }
  }

  if (result) return <ExplanationPanel result={result} />;

  if (loading) {
    return (
      <div className="bg-slate-50 rounded-xl p-6 flex flex-col items-center gap-3 text-center">
        <div className="w-6 h-6 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-slate-500">phi4-mini is reasoning…</p>
        <p className="text-xs text-slate-400">First call may take 15–30 s while the model loads.</p>
      </div>
    );
  }

  return (
    <div className="bg-slate-50 rounded-xl p-5 space-y-3">
      {!artifactLoaded && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          Stage 1 artifact not loaded — SHAP features will be unavailable.
        </p>
      )}
      <p className="text-sm text-slate-600">
        Request an on-demand audit from phi4-mini. The model reasons over Stage 1's SHAP
        features, Stage 2's independent note-based score, and the discharge note itself, then
        returns its own uphold/override decision with a quoted, verified justification.
      </p>
      {error && (
        <p className="text-xs text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
          {error}
        </p>
      )}
      <button
        onClick={handleExplain}
        className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold transition-colors"
      >
        Request Explanation
      </button>
      <p className="text-xs text-slate-400 text-center">
        Requires Ollama running locally with <code>phi4-mini</code> pulled.
      </p>
    </div>
  );
}

// ── Modal ─────────────────────────────────────────────────────────────────────

const fmt = (v: number | null, digits = 0, suffix = "") =>
  v != null ? `${digits ? v.toFixed(digits) : Math.round(v)}${suffix}` : "N/A";

export default function PatientModal({
  patient,
  artifactLoaded,
  onClose,
}: {
  patient: Patient;
  artifactLoaded: boolean;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
          <div>
            <h2 className="text-base font-bold text-slate-800">
              Patient #{patient.hadm_id}
            </h2>
            <p className="text-xs text-slate-400">
              {patient.age != null ? `${Math.round(patient.age)}y` : "Age N/A"}{" "}
              · {patient.gender ?? "—"}{" "}
              · {patient.admission_type ?? "—"} admission
            </p>
          </div>
          <div className="flex items-center gap-2">
            {patient.readmitted === true && (
              <span className="text-xs font-semibold text-red-600 bg-red-50 px-2 py-1 rounded-full">
                Readmitted within 30d
              </span>
            )}
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-600 text-lg leading-none px-2"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="p-6 space-y-5">
          {/* Score gauges */}
          <div className="grid grid-cols-2 gap-4 bg-slate-50 rounded-xl p-4">
            <ScoreGauge
              label={`Stage 1 · XGBoost · thr=${patient.stage1_threshold.toFixed(3)}`}
              score={patient.stage1_score}
              color="text-blue-600"
            />
            <ScoreGauge
              label={`Stage 2 · Longformer · thr=${patient.stage2_threshold.toFixed(3)} (${patient.age_band})`}
              score={patient.stage2_score}
              color="text-violet-600"
            />
          </div>

          {/* Patient profile */}
          <div>
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Patient Profile
            </h3>
            <div className="bg-slate-50 rounded-xl px-4">
              <InfoRow label="Age group"                 value={patient.age_band} />
              <InfoRow label="Length of stay"           value={fmt(patient.los_days, 1, " days")} />
              <InfoRow label="Charlson Comorbidity Index"
                value={`${fmt(patient.charlson_index)} (${fmt(patient.n_comorbidities)} conditions)`} />
              <InfoRow label="Prior admissions"          value={fmt(patient.n_prior_admissions)} />
              <InfoRow label="Days since last discharge" value={fmt(patient.days_since_last_discharge, 0, " d")} />
              <InfoRow label="Came via ED"               value={patient.came_via_ed == null ? "N/A" : patient.came_via_ed ? "Yes" : "No"} />
              <InfoRow label="Discharged to facility"   value={patient.discharged_to_facility == null ? "N/A" : patient.discharged_to_facility ? "Yes (SNF/rehab)" : "Home"} />
              <InfoRow label="Last creatinine"           value={patient.creatinine_last != null ? `${patient.creatinine_last.toFixed(2)} mg/dL` : "N/A"} />
              <InfoRow label="Last WBC"                  value={patient.white_blood_cells_last != null ? `${patient.white_blood_cells_last.toFixed(1)} ×10³/μL` : "N/A"} />
            </div>
          </div>

          {/* Stage 3 */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Stage 3 — Clinical Audit
              </h3>
              <span className="text-xs text-emerald-500 bg-emerald-50 px-2 py-0.5 rounded-full">
                phi4-mini
              </span>
            </div>
            <Stage3Section hadmId={patient.hadm_id} artifactLoaded={artifactLoaded} />
          </div>
        </div>
      </div>
    </div>
  );
}
