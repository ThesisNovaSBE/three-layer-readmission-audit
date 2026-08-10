import type { ExplanationResult, HealthStatus, PatientsPage } from "./types";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail ?? res.statusText);
  }
  return res.json() as Promise<T>;
}

export function fetchHealth(): Promise<HealthStatus> {
  return request<HealthStatus>("/health");
}

export function fetchPatients(params: {
  confirmedOnly?: boolean;
  limit?: number;
  offset?: number;
  q?: string;
} = {}): Promise<PatientsPage> {
  const { confirmedOnly = true, limit = 50, offset = 0, q = "" } = params;
  const qs = new URLSearchParams({
    confirmed_only: String(confirmedOnly),
    limit: String(limit),
    offset: String(offset),
  });
  if (q) qs.set("q", q);
  return request<PatientsPage>(`/patients?${qs}`);
}

export function explainPatient(hadmId: number): Promise<ExplanationResult> {
  return request<ExplanationResult>(`/patients/${hadmId}/explain`, { method: "POST" });
}
