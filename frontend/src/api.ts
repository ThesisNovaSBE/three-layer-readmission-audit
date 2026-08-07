import type { ExplanationResult, HealthStatus, Patient } from "./types";

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

export function fetchPatients(confirmedOnly = true): Promise<Patient[]> {
  return request<Patient[]>(`/patients?confirmed_only=${confirmedOnly}`);
}

export function explainPatient(hadmId: number): Promise<ExplanationResult> {
  return request<ExplanationResult>(`/patients/${hadmId}/explain`, { method: "POST" });
}
