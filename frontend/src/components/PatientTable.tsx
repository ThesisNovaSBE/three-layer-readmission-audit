import { useState } from "react";
import type { Patient } from "../types";

function RiskBadge({ score }: { score: number }) {
  const color =
    score >= 0.7
      ? "bg-red-100 text-red-700"
      : score >= 0.5
      ? "bg-orange-100 text-orange-700"
      : "bg-yellow-100 text-yellow-700";
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold ${color}`}>
      {score.toFixed(3)}
    </span>
  );
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 bg-slate-100 rounded-full h-1.5">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="text-xs text-slate-500">{value.toFixed(2)}</span>
    </div>
  );
}

type SortKey = "stage1_score" | "stage2_score";

export default function PatientTable({
  patients,
  onSelect,
}: {
  patients: Patient[];
  onSelect: (p: Patient) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("stage1_score");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");

  const sorted = [...patients].sort((a, b) => {
    const diff = a[sortKey] - b[sortKey];
    return sortDir === "desc" ? -diff : diff;
  });

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSortKey(key); setSortDir("desc"); }
  }

  function SortBtn({ col, label }: { col: SortKey; label: string }) {
    const active = sortKey === col;
    return (
      <button
        onClick={() => toggleSort(col)}
        className={`text-xs font-semibold flex items-center gap-1 ${
          active ? "text-blue-600" : "text-slate-500"
        }`}
      >
        {label} {active ? (sortDir === "desc" ? "↓" : "↑") : "↕"}
      </button>
    );
  }

  const fmt = (v: number | null, digits = 0) =>
    v != null ? (digits ? v.toFixed(digits) : String(Math.round(v))) : "—";

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-100 overflow-hidden">
      <div className="px-6 py-4 border-b border-slate-100 flex items-center justify-between">
        <div>
          <h2 className="text-base font-bold text-slate-800">Confirmed High-Risk Patients</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Stage 2 confirmed · calibrated per-age-group thresholds · Click a row for details
          </p>
        </div>
        <span className="text-sm text-slate-400">{patients.length} shown</span>
      </div>

      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
          <tr>
            <th className="px-4 py-3 text-left">Admission ID</th>
            <th className="px-4 py-3 text-left">Age / Band</th>
            <th className="px-4 py-3 text-left">Gender</th>
            <th className="px-4 py-3 text-left">LOS (d)</th>
            <th className="px-4 py-3 text-left">Charlson</th>
            <th className="px-4 py-3 text-left">
              <SortBtn col="stage1_score" label="Stage 1" />
            </th>
            <th className="px-4 py-3 text-left">
              <SortBtn col="stage2_score" label="Stage 2" />
            </th>
            <th className="px-4 py-3 text-left">Readmitted</th>
            <th className="px-4 py-3 text-left">Detail</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-50">
          {sorted.map((p) => (
            <tr
              key={p.hadm_id}
              className="hover:bg-blue-50 cursor-pointer transition-colors"
              onClick={() => onSelect(p)}
            >
              <td className="px-4 py-3 font-mono text-slate-600 text-xs">{p.hadm_id}</td>
              <td className="px-4 py-3 text-slate-700">
                {p.age != null ? `${Math.round(p.age)}y` : "—"}
                <span className="ml-1 text-xs text-slate-400">({p.age_band})</span>
              </td>
              <td className="px-4 py-3 text-slate-600">{p.gender ?? "—"}</td>
              <td className="px-4 py-3 text-slate-600">{fmt(p.los_days, 1)}</td>
              <td className="px-4 py-3 text-slate-600">{fmt(p.charlson_index)}</td>
              <td className="px-4 py-3">
                <RiskBadge score={p.stage1_score} />
              </td>
              <td className="px-4 py-3">
                <ScoreBar value={p.stage2_score} color="bg-violet-400" />
              </td>
              <td className="px-4 py-3">
                {p.readmitted === true ? (
                  <span className="text-xs font-semibold text-red-600 bg-red-50 px-2 py-0.5 rounded-full">
                    Yes
                  </span>
                ) : p.readmitted === false ? (
                  <span className="text-xs font-semibold text-slate-400 bg-slate-100 px-2 py-0.5 rounded-full">
                    No
                  </span>
                ) : (
                  <span className="text-xs text-slate-300">—</span>
                )}
              </td>
              <td className="px-4 py-3">
                <button className="text-xs text-blue-500 hover:underline">View →</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
