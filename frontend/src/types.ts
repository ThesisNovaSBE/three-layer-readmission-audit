export interface Patient {
  hadm_id: number;
  age_band: string;
  stage1_score: number;
  stage1_threshold: number;
  stage2_score: number;
  stage2_threshold: number;
  stage2_confirmed: boolean;
  readmitted: boolean | null;
  age: number | null;
  gender: string | null;
  los_days: number | null;
  charlson_index: number | null;
  n_comorbidities: number | null;
  n_prior_admissions: number | null;
  days_since_last_discharge: number | null;
  admission_type: string | null;
  came_via_ed: boolean | null;
  discharged_to_facility: boolean | null;
  creatinine_last: number | null;
  white_blood_cells_last: number | null;
}

export interface ExplanationResult {
  hadm_id: number;
  stage1_score: number;
  stage1_threshold: number;
  stage2_score: number;
  stage2_confirmed: boolean;
  r1: number;
  r2: number;
  displacement: number;
  discordance_mode: string | null;
  top_shap_features: string[];
  attention_sentences: string[];
  decision: string | null;
  primary_clinical_domain: string | null;
  supporting_quote: string;
  quote_verified: boolean | null;
  planned_return: string | null;
  clinical_justification: string;
  annotation_failed: boolean;
}

export interface PatientsPage {
  patients: Patient[];
  total: number;
}

export interface HealthStatus {
  status: string;
  patients_loaded: number;
  artifact_loaded: boolean;
  demo_mode: boolean;
}
