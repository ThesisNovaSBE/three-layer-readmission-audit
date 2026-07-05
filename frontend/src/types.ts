export interface Patient {
  hadm_id: number;
  age: number;
  age_band: string;
  gender: string;
  los_days: number;
  charlson_index: number;
  n_comorbidities: number;
  n_prior_admissions: number;
  days_since_last_discharge: number;
  admission_type: string;
  came_via_ed: boolean;
  discharged_to_facility: boolean;
  creatinine: number | null;
  wbc: number | null;
  stage1_score: number;
  stage1_threshold: number;
  stage2_score: number;
  stage2_threshold: number;
  stage2_confirmed: boolean;
  readmitted: boolean;
  explanation: string;
}
