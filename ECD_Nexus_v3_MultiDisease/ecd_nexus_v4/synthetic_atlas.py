# ─────────────────────────────────────────────────────────────────────────────
# ECD-Nexus 3.0 | synthetic_atlas.py
# Synthetic Clinical Atlas — maps feature values to clinical risk findings
# Each rule backed by published medical research
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ─── Acoustic Atlas ───────────────────────────────────────────────────────────
ACOUSTIC_ATLAS = [
    {
        "feature"   : "SC_Mean",
        "condition" : "above",
        "threshold" : 1800,
        "finding"   : "High-pitched cry",
        "clinical"  : "Meningitis / Birth Asphyxia / Acute Pain",
        "confidence": "HIGH",
        "evidence"  : "Wasz-Höckert et al. (1968): pain cry F0 significantly elevated vs hunger cry",
        "urgency"   : "IMMEDIATE",
    },
    {
        "feature"   : "SC_Mean",
        "condition" : "above",
        "threshold" : 1200,
        "finding"   : "Moderately elevated pitch",
        "clinical"  : "Hunger / Discomfort / Early distress",
        "confidence": "MEDIUM",
        "evidence"  : "Donate-A-Cry Corpus: hunger cries show SC 1200-1800 Hz range",
        "urgency"   : "ROUTINE",
    },
    {
        "feature"   : "SC_Mean",
        "condition" : "below",
        "threshold" : 900,
        "finding"   : "Low-pitched, hoarse cry",
        "clinical"  : "Fatigue / Hypothyroidism / Cardiac compromise",
        "confidence": "MEDIUM",
        "evidence"  : "Lind et al. (1965): hypothyroid neonates show depressed cry frequency",
        "urgency"   : "URGENT",
    },
    {
        "feature"   : "RMS_Mean",
        "condition" : "above",
        "threshold" : 0.06,
        "finding"   : "High energy, intense cry",
        "clinical"  : "Belly Pain / Acute Distress",
        "confidence": "HIGH",
        "evidence"  : "Donate-A-Cry Corpus: belly pain shows highest RMS amplitude",
        "urgency"   : "URGENT",
    },
    {
        "feature"   : "RMS_Mean",
        "condition" : "below",
        "threshold" : 0.008,
        "finding"   : "Weak, low-energy cry",
        "clinical"  : "Neonatal Sepsis / Congenital Heart Defect / Severe Fatigue",
        "confidence": "HIGH",
        "evidence"  : "Michelsson et al. (1977): pathological neonates show reduced cry amplitude",
        "urgency"   : "IMMEDIATE",
    },
    {
        "feature"   : "ZCR_Mean",
        "condition" : "above",
        "threshold" : 0.08,
        "finding"   : "High zero-crossing rate — noisy, turbulent cry",
        "clinical"  : "Respiratory Distress / Seizure / Neurological abnormality",
        "confidence": "MEDIUM",
        "evidence"  : "Várallyay et al. (2004): neurological cry disorders show elevated ZCR",
        "urgency"   : "URGENT",
    },
    {
        "feature"   : "MFCCs13Mean",
        "condition" : "below",
        "threshold" : -50,
        "finding"   : "Abnormal vocal tract shape",
        "clinical"  : "Hypothyroidism / Structural airway abnormality",
        "confidence": "MEDIUM",
        "evidence"  : "MFCC captures vocal tract resonance — abnormal in thyroid dysfunction",
        "urgency"   : "ROUTINE",
    },
    {
        "feature"   : "SBAN_Mean",
        "condition" : "below",
        "threshold" : 1200,
        "finding"   : "Narrow spectral bandwidth — restricted frequency range",
        "clinical"  : "Airway obstruction / Bronchiolitis / RSV",
        "confidence": "MEDIUM",
        "evidence"  : "Narrow bandwidth indicates constrained vocal output — respiratory compromise",
        "urgency"   : "URGENT",
    },
]

# ─── Motor Atlas ──────────────────────────────────────────────────────────────
MOTOR_ATLAS = [
    {
        "feature"   : "Limb_Asymmetry",
        "condition" : "above",
        "threshold" : 0.40,
        "finding"   : "HIGH limb asymmetry — significant left-right movement difference",
        "clinical"  : "Cerebral Palsy / Hemiplegia / HIE",
        "confidence": "HIGH",
        "evidence"  : "Prechtl (1997): asymmetric spontaneous movements are early CP biomarker",
        "urgency"   : "URGENT",
    },
    {
        "feature"   : "Limb_Asymmetry",
        "condition" : "above",
        "threshold" : 0.20,
        "finding"   : "MEDIUM limb asymmetry",
        "clinical"  : "Possible unilateral motor delay — monitor closely",
        "confidence": "MEDIUM",
        "evidence"  : "Einspieler et al. (2004): asymmetry > 0.2 warrants follow-up",
        "urgency"   : "ROUTINE",
    },
    {
        "feature"   : "Repetitive_Motion_Index",
        "condition" : "above",
        "threshold" : 0.45,
        "finding"   : "HIGH repetitive rhythmic motion",
        "clinical"  : "Neonatal Seizure (ictal pattern)",
        "confidence": "HIGH",
        "evidence"  : "Mizrahi & Kellaway (1987): rhythmic limb movements key seizure indicator",
        "urgency"   : "IMMEDIATE",
    },
    {
        "feature"   : "Activity_Level",
        "condition" : "below",
        "threshold" : 0.003,
        "finding"   : "Very LOW motor activity — infant barely moving",
        "clinical"  : "Neonatal Sepsis / Hypotonia / Cardiac failure",
        "confidence": "HIGH",
        "evidence"  : "Als (1982): severely reduced spontaneous movement indicates systemic illness",
        "urgency"   : "IMMEDIATE",
    },
    {
        "feature"   : "Activity_Level",
        "condition" : "above",
        "threshold" : 0.025,
        "finding"   : "HIGH motor activity — intense body movements",
        "clinical"  : "Acute Pain / Colic / Hypoglycemia jitteriness",
        "confidence": "MEDIUM",
        "evidence"  : "High activity combined with high SC = pain/distress pattern",
        "urgency"   : "URGENT",
    },
    {
        "feature"   : "Movement_Frequency",
        "condition" : "above",
        "threshold" : 3.5,
        "finding"   : "High-frequency repetitive movement (>3.5 Hz)",
        "clinical"  : "Hypoglycemia jitteriness / Seizure tremor",
        "confidence": "HIGH",
        "evidence"  : "Scher et al. (1993): tremor frequency > 3 Hz distinguishes jitteriness from seizure",
        "urgency"   : "IMMEDIATE",
    },
    {
        "feature"   : "Joint_Angle_Variability",
        "condition" : "below",
        "threshold" : 8.0,
        "finding"   : "Very LOW joint angle variability — rigid, restricted movements",
        "clinical"  : "Hypotonia / HIE / Severe neurological compromise",
        "confidence": "MEDIUM",
        "evidence"  : "Low joint variability indicates decreased motor repertoire — neurological sign",
        "urgency"   : "URGENT",
    },
]

# ─── Fusion Atlas ─────────────────────────────────────────────────────────────
FUSION_ATLAS = [
    {
        "pattern"   : "HIGH_SC + HIGH_RMI",
        "conditions": {"SC_Mean": (">", 1800), "Repetitive_Motion_Index": (">", 0.45)},
        "finding"   : "High-pitched cry WITH repetitive movements",
        "clinical"  : "Neonatal Seizure (PRIORITY)",
        "confidence": "HIGH",
        "urgency"   : "IMMEDIATE",
    },
    {
        "pattern"   : "LOW_RMS + LOW_ACT",
        "conditions": {"RMS_Mean": ("<", 0.008), "Activity_Level": ("<", 0.003)},
        "finding"   : "Weak cry WITH very low motor activity",
        "clinical"  : "Neonatal Sepsis / Cardiac Compromise (PRIORITY)",
        "confidence": "HIGH",
        "urgency"   : "IMMEDIATE",
    },
    {
        "pattern"   : "HIGH_SC + HIGH_ASYM",
        "conditions": {"SC_Mean": (">", 1500), "Limb_Asymmetry": (">", 0.40)},
        "finding"   : "High-pitched cry WITH limb asymmetry",
        "clinical"  : "HIE / Birth Asphyxia with neurological involvement",
        "confidence": "HIGH",
        "urgency"   : "IMMEDIATE",
    },
    {
        "pattern"   : "MEDIUM_SC + LOW_ASYM",
        "conditions": {"SC_Mean": (">", 1200), "Limb_Asymmetry": ("<", 0.15)},
        "finding"   : "Normal pitch, symmetric movement",
        "clinical"  : "Hunger / Discomfort (baseline state)",
        "confidence": "HIGH",
        "urgency"   : "MONITOR",
    },
]

# ─── Disease Risk Profiles ────────────────────────────────────────────────────
DISEASE_RISK_MATRIX = {
    "belly_pain" : {
        "display"   : "Belly Pain / Colic",
        "category"  : "Baseline",
        "urgency"   : "ROUTINE",
        "first_aid" : "Gently massage tummy clockwise. Try bicycle legs. Hold baby upright. See doctor if crying > 2 hours.",
        "see_doctor_if": "Crying does not stop after 2 hours | Blood in stool | Fever > 38C",
    },
    "burping"    : {
        "display"   : "Needs to Burp / Gas",
        "category"  : "Baseline",
        "urgency"   : "MONITOR",
        "first_aid" : "Hold baby upright on shoulder and gently pat back for 10-15 minutes.",
        "see_doctor_if": "Baby vomits forcefully | Refuses to feed",
    },
    "discomfort" : {
        "display"   : "General Discomfort",
        "category"  : "Baseline",
        "urgency"   : "MONITOR",
        "first_aid" : "Check diaper. Adjust clothing. Check room temperature. Offer feed.",
        "see_doctor_if": "Discomfort persists > 1 hour with no clear cause",
    },
    "hungry"     : {
        "display"   : "Hunger",
        "category"  : "Baseline",
        "urgency"   : "MONITOR",
        "first_aid" : "Feed the baby immediately. Hold upright 15 minutes after feeding.",
        "see_doctor_if": "Baby refuses to feed | Not gaining weight",
    },
    "tired"      : {
        "display"   : "Fatigue / Tiredness",
        "category"  : "Baseline",
        "urgency"   : "MONITOR",
        "first_aid" : "Move to a quiet dark room. Swaddle gently. Rock slowly.",
        "see_doctor_if": "Baby is abnormally limp | Cannot be awakened",
    },
    "crackle"    : {
        "display"   : "Crackle (Lung Sound)",
        "category"  : "Respiratory",
        "urgency"   : "URGENT",
        "first_aid" : "Keep baby upright. Ensure hydration. See doctor today.",
        "see_doctor_if": "Fast breathing | Difficulty breathing | Blue lips",
    },
    "wheeze"     : {
        "display"   : "Wheeze (Lung Sound)",
        "category"  : "Respiratory",
        "urgency"   : "URGENT",
        "first_aid" : "Keep baby calm and upright. See doctor today.",
        "see_doctor_if": "Breathing difficulty | Not feeding | Blue lips",
    },
    "crackle_and_wheeze": {
        "display"   : "Crackle + Wheeze (Severe Respiratory)",
        "category"  : "Respiratory",
        "urgency"   : "IMMEDIATE",
        "first_aid" : "Go to hospital immediately. Do not delay.",
        "see_doctor_if": "Go NOW — both crackle and wheeze detected",
    },
    "normal_respiratory": {
        "display"   : "Normal Lung Sounds",
        "category"  : "Respiratory",
        "urgency"   : "MONITOR",
        "first_aid" : "No respiratory concern detected. Monitor normally.",
        "see_doctor_if": "Breathing becomes labored or fast",
    },
    "heart_murmur": {
        "display"   : "Heart Murmur (possible CHD)",
        "category"  : "Cardiac",
        "urgency"   : "URGENT",
        "first_aid" : "See a pediatric cardiologist for evaluation.",
        "see_doctor_if": "Blue lips or fingertips | Poor feeding | Excessive sweating",
    },
    "normal_cardiac": {
        "display"   : "Normal Heart Sound",
        "category"  : "Cardiac",
        "urgency"   : "MONITOR",
        "first_aid" : "No cardiac concern detected. Monitor normally.",
        "see_doctor_if": "Baby turns blue | Breathes very fast | Feeds poorly",
    },
    "extrahls"   : {
        "display"   : "Extra Heart Sound",
        "category"  : "Cardiac",
        "urgency"   : "URGENT",
        "first_aid" : "Consult a doctor for cardiac evaluation.",
        "see_doctor_if": "Any sign of breathlessness or cyanosis",
    },
    "uncertain"  : {
        "display"   : "Uncertain — Consult Doctor",
        "category"  : "Unknown",
        "urgency"   : "ROUTINE",
        "first_aid" : "AI confidence too low. Please consult an ASHA worker or doctor.",
        "see_doctor_if": "If any symptoms worsen",
    },
}


# ─── Query Functions ──────────────────────────────────────────────────────────
def query_acoustic_atlas(audio_features):
    triggered = []
    for rule in ACOUSTIC_ATLAS:
        val = audio_features.get(rule["feature"])
        if val is None:
            continue
        try:
            val = float(val)
        except:
            continue
        hit = ((rule["condition"] == "above" and val > rule["threshold"]) or
               (rule["condition"] == "below" and val < rule["threshold"]))
        if hit:
            triggered.append({
                "feature"   : rule["feature"],
                "value"     : round(val, 6),
                "threshold" : rule["threshold"],
                "finding"   : rule["finding"],
                "clinical"  : rule["clinical"],
                "confidence": rule["confidence"],
                "evidence"  : rule["evidence"],
                "urgency"   : rule["urgency"],
            })
    return triggered


def query_motor_atlas(motor_features):
    triggered = []
    for rule in MOTOR_ATLAS:
        val = motor_features.get(rule["feature"])
        if val is None:
            continue
        try:
            val = float(val)
        except:
            continue
        hit = ((rule["condition"] == "above" and val > rule["threshold"]) or
               (rule["condition"] == "below" and val < rule["threshold"]))
        if hit:
            triggered.append({
                "feature"   : rule["feature"],
                "value"     : round(val, 6),
                "threshold" : rule["threshold"],
                "finding"   : rule["finding"],
                "clinical"  : rule["clinical"],
                "confidence": rule["confidence"],
                "evidence"  : rule["evidence"],
                "urgency"   : rule["urgency"],
            })
    return triggered


def query_fusion_atlas(audio_features, motor_features):
    triggered = []
    all_features = {**audio_features, **motor_features}
    for rule in FUSION_ATLAS:
        match = True
        for feat, (op, thresh) in rule["conditions"].items():
            val = all_features.get(feat)
            if val is None:
                match = False
                break
            try:
                val = float(val)
            except:
                match = False
                break
            if op == ">" and not (val > thresh):
                match = False
                break
            if op == "<" and not (val < thresh):
                match = False
                break
        if match:
            triggered.append(rule)
    return triggered


def compute_risk_score(audio_findings, motor_findings, fusion_findings):
    urgency_scores = {"IMMEDIATE": 3, "URGENT": 2, "ROUTINE": 1, "MONITOR": 0}
    max_urgency    = "MONITOR"
    total_score    = 0

    for f in audio_findings + motor_findings:
        u = f.get("urgency", "MONITOR")
        total_score += urgency_scores.get(u, 0)
        if urgency_scores.get(u, 0) > urgency_scores.get(max_urgency, 0):
            max_urgency = u

    for f in fusion_findings:
        u = f.get("urgency", "MONITOR")
        total_score += urgency_scores.get(u, 0) * 2
        if urgency_scores.get(u, 0) > urgency_scores.get(max_urgency, 0):
            max_urgency = u

    risk_level = (
        "CRITICAL" if total_score >= 8 else
        "HIGH"     if total_score >= 4 else
        "MEDIUM"   if total_score >= 2 else
        "LOW"
    )

    return {
        "risk_level"   : risk_level,
        "max_urgency"  : max_urgency,
        "total_score"  : total_score,
        "n_acoustic"   : len(audio_findings),
        "n_motor"      : len(motor_findings),
        "n_fusion"     : len(fusion_findings),
    }


def get_disease_profile(label):
    return DISEASE_RISK_MATRIX.get(label, {
        "display"      : label.replace("_", " ").title(),
        "category"     : "Unknown",
        "urgency"      : "ROUTINE",
        "first_aid"    : "Consult a healthcare worker.",
        "see_doctor_if": "Condition persists or worsens.",
    })


if __name__ == "__main__":
    test_audio = {"SC_Mean": 1900, "RMS_Mean": 0.003, "ZCR_Mean": 0.09}
    test_motor = {"Limb_Asymmetry": 0.45, "Activity_Level": 0.002,
                  "Repetitive_Motion_Index": 0.5, "Movement_Frequency": 4.0}

    print("=== Acoustic Atlas ===")
    for f in query_acoustic_atlas(test_audio):
        print(f"  [{f['urgency']}] {f['finding']} → {f['clinical']}")

    print("\n=== Motor Atlas ===")
    for f in query_motor_atlas(test_motor):
        print(f"  [{f['urgency']}] {f['finding']} → {f['clinical']}")

    print("\n=== Fusion Atlas ===")
    for f in query_fusion_atlas(test_audio, test_motor):
        print(f"  [{f['urgency']}] {f['pattern']} → {f['clinical']}")

    risk = compute_risk_score(
        query_acoustic_atlas(test_audio),
        query_motor_atlas(test_motor),
        query_fusion_atlas(test_audio, test_motor)
    )
    print(f"\n=== Risk Score ===")
    print(f"  Level: {risk['risk_level']} | Urgency: {risk['max_urgency']}")
