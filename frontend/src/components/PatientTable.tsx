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

type SortKey = "hadm_id" | "age" | "los_days" | "charlson_index" | "stage1_score" | "stage2_score";

export default function PatientTable({
  patients,
  total,
  search,
  onSearchChange,
  onLoadMore,
  loadingMore,
  onSelect,
}: {
  patients: Patient[];
  total: number;
  search: string;
  onSearchChange: (q: string) => void;
  onLoadMore: () => void;
  loadingMore: boolean;
  onSelect: (p: Patient) => void;
}) {
  const [sortKey, setSortKey] = useState<SortKey>("stage1_score");
  const [sortDir, setSortDir] = useState<"desc" | "asc">("desc");

  const sorted = [...patients].sort((a, b) => {
    const av = (a[sortKey] as number | null) ?? -Infinity;
    const bv = (b[sortKey] as number | null) ?? -Infinity;
    return sortDir === "desc" ? bv - av : av - bv;
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
      <div className="px-6 py-4 border-b border-slate-100 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-slate-800">Confirmed High-Risk Patients</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Stage 2 confirmed · calibrated per-age-group thresholds · Click a row for details
            </p>
          </div>
          <span className="text-sm text-slate-400 shrink-0 ml-4">
            {patients.length.toLocaleString()} of {total.toLocaleString()} shown
          </span>
        </div>
        <div className="relative">
          <svg
            className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-400"
            fill="none" stroke="currentColor" strokeWidth={2}
            viewBox="0 0 24 24"
          >
            <circle cx={11} cy={11} r={8} />
            <path d="M21 21l-4.35-4.35" strokeLinecap="round" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search by admission ID, age group (e.g. 70+), or gender…"
            className="w-full pl-8 pr-4 py-1.5 text-sm border border-slate-200 rounded-lg bg-slate-50 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-200 focus:border-blue-300"
          />
          {search && (
            <button
              onClick={() => onSearchChange("")}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 text-xs leading-none"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <table className="w-full text-sm">
        <thead className="bg-slate-50 text-xs text-slate-500 uppercase tracking-wide">
          <tr>
            <th className="px-4 py-3 text-left">
              <SortBtn col="hadm_id" label="Admission ID" />
            </th>
            <th className="px-4 py-3 text-left">
              <SortBtn col="age" label="Age / Band" />
            </th>
            <th className="px-4 py-3 text-left">Gender</th>
            <th className="px-4 py-3 text-left">
              <SortBtn col="los_days" label="LOS (d)" />
            </th>
            <th className="px-4 py-3 text-left">
              <SortBtn col="charlson_index" label="Charlson" />
            </th>
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

      {patients.length < total && (
        <div className="px-6 py-4 border-t border-slate-100 text-center">
          <button
            onClick={onLoadMore}
            disabled={loadingMore}
            className="px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400 text-white text-sm font-medium transition-colors"
          >
            {loadingMore
              ? "Loading…"
              : `Load 50 more  (${patients.length.toLocaleString()} of ${total.toLocaleString()})`}
          </button>
        </div>
      )}
      {patients.length > 0 && patients.length === total && (
        <div className="px-6 py-3 border-t border-slate-100 text-center text-xs text-slate-400">
          All {total.toLocaleString()} patients shown
        </div>
      )}
    </div>
  );
}
