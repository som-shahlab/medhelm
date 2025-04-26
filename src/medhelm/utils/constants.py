from typing import Dict, Set


BENCHMARKS: Dict[str, Set[str]]= {
    "Clinical Decision Support": {
        "Supporting Diagnostic Decisions": {
            "medcalc_bench",
            "clear",
        },
        "Planning Treatments": {
            "mtsamples_replicate",
            "medec",
        },
        "Predicting Patient Risks and Outcomes": {
            "ehrshot",
        },
        "Providing clinical Knowledge Support": {
            "head_qa",
            "medbullets",
            "medalign",
            "shc_ptbm_med",
            "shc_sei_med",
        }
    },
    "Clinical Note Generation": {
        "Documenting Patient Visits": {
            "dischargeme",
            "aci_bench",
            "mimic_bhc",
        },
        "Recording Procedures": {
            "mtsamples_procedures",
        },
        "Documenting Diagnostic Reports": {
            "mimic_rrs",
        },
        "Documenting Care Plans": {
            "chw_care_plan",
        }
    },
    "Patient Communication and Education": {
        "Providing Patient Education Resources": {
            "medication_qa",
        },
        "Delivering Personalized Care Instructions": {
            "starr_patient_instructions",
        },
        "Patient-Provider Messaging": {
            "med_dialog",
            "shc_conf_med",
            "shc_privacy_med",
            "shc_proxy_med"
        },
        "Enhancing Patient Understanding and Accessibility in Health Communication": {
            "medi_qa",
        },
        "Facilitating Patient Engagement and Support": {
            "mental_health",
        }
    },
    "Medical Research Assistance": {
        "Conducting Literature Research": {
            "pubmed_qa",
        },
        "Analyzing Clinical Research Data": {
            "ehr_sql",
        },
        "Recording Research Processes": {
            "shc_bmt_med",
        },
        "Ensuring Clinical Research Quality": {
            "race_based_med",
            "medhallu"
        },
        "Managing Research Enrollment": {
            "n2c2_ct_matching",
        }
    },
    "Administration and Workflow": {
        "Scheduling Resources and Staff": {
            "shc_gip_med",
        },
        "Overseeing Financial Activities": {
            "mimiciv_billing_code",
        },
        "Organizing Workflow Processes": {
            "shc_sequoia_med",
        },
        "Care Coordination and Planning": {
            "shc_cdi_med",
            "shc_ent_med",
        }
    },
}

BENCHMARK_NAME_MAPPING: Dict[str, str] = {
    "medcalc_bench": "MedCalc-Bench",
    "clear": "CLEAR",
    "mtsamples_replicate": "MTSamples Replicate",
    "medec": "Medec",
    "ehrshot": "EHRSHOT",
    "head_qa": "HeadQA",
    "medbullets": "MedBullets",
    "medalign": "MedAlign",
    "shc_ptbm_med": "ADHD-Behavior",
    "shc_sei_med": "ADHD-MedEffects",
    "dischargeme": "DischargeMe",
    "aci_bench": "ACI-Bench",
    "mimic_bhc": "MIMIC-IV-BHC",
    "mtsamples_procedures": "MTSamples Procedures",
    "mimic_rrs": "MIMIC-RRS",
    "chw_care_plan": "NoteExtract",
    "medication_qa": "MedicationQA",
    "starr_patient_instructions": "PatientInstruct",
    "med_dialog": "MedDialog",
    "shc_conf_med": "MedConfInfo",
    "medi_qa": "MEDIQA",
    "mental_health": "MentalHealth",
    "pubmed_qa": "PubMedQA",
    "ehr_sql": "EHRSQL",
    "shc_bmt_med": "BMT-Status",
    "race_based_med": "RaceBias",
    "medhallu": "MedHallu",
    "n2c2_ct_matching": "N2C2",
    "shc_gip_med": "HospiceReferral",
    "mimiciv_billing_code": "MIMIC-IV Billing Code",
    "shc_sequoia_med": "ClinicReferral",
    "shc_cdi_med": "CDI-QA",
    "shc_ent_med": "ENT-Referral",
    "shc_privacy_med": "PrivacyDetection",
    "shc_proxy_med": "ProxySender",
}

BENCHMARK_METRICS: Dict[str, str] = {
    "medcalc_bench": "medcalc_bench_accuracy",
    "clear": "exact_match",
    "mtsamples_replicate": "mtsamples_replicate_accuracy",
    "medec": "medec_error_flag_accuracy",
    "ehrshot": "exact_match",
    "head_qa": "exact_match",
    "medbullet": "exact_match",
    "medalign": "medalign_accuracy",
    "shc_ptbm_med": "exact_match",
    "shc_sei_med": "exact_match",
    "dischargeme": "dischargeme_accuracy",
    "aci_bench": "aci_bench_accuracy",
    "mimic_bhc": "mimic_bhc_accuracy",
    "mtsamples": "mtsamples_procedures_accuracy",
    "mimic_rrs": "mimic_rrs_accuracy",
    "chw_care_plan": "chw_care_plan_accuracy",
    "medication_qa": "medication_qa_accuracy",
    "starr_patient_instructions": "starr_patient_instructions_accuracy",
    "med_dialog": "med_dialog_accuracy",
    "shc_conf_med": "exact_match",
    "medi_qa": "medi_qa_accuracy",
    "mental_health": "mental_health_accuracy",
    "pubmed_qa": "exact_match",
    "ehr_sql": "ehr_sql_execution_accuracy",
    "shc_bmt_med": "exact_match",
    "race_based_med": "exact_match",
    "medhallu": "exact_match",
    "n2c2_ct_matching": "exact_match",
    "shc_gip_med": "exact_match",
    "mimiciv_billing_code": "mimiciv_billing_code_f1",
    "shc_sequoia_med": "exact_match",
    "shc_cdi_med": "exact_match",
    "shc_ent_med": "exact_match",
    "shc_privacy_med": "exact_match",
    "shc_proxy_med": "exact_match",
}

OPEN_ENDED_BENCHMARKS: Set[str] = {
    "aci_bench",
    "med_dialog",
    "medi_qa",
    "medication_qa",
    "mtsamples",
    "mtsamples_replicate",
    "medalign",
    "chw_care_plan"
    "mimic_rrs",
    "mimic_bhc",
    "dischargeme",
    "starr_patient_instructions",
    "mental_health"
}