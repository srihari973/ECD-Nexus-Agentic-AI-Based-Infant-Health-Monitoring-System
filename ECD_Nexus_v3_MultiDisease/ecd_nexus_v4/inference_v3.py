# ─────────────────────────────────────────────────────────────────────────────
# ECD-Nexus 3.0 | inference_v3.py
# Multi-model inference: Cry + Respiratory + Cardiac + Motor
# All 3 models run on uploaded audio → unified disease report
# ─────────────────────────────────────────────────────────────────────────────

import pickle
import numpy as np
import librosa
import warnings
import time
import os
warnings.filterwarnings('ignore')

from groq import Groq

CONFIDENCE_THRESHOLD = 55.0

# ─── Disease display names ────────────────────────────────────────────────────
DISEASE_DISPLAY = {
    # Cry states
    'belly_pain'            : 'Belly Pain / Colic',
    'burping'               : 'Needs to Burp',
    'discomfort'            : 'General Discomfort',
    'hungry'                : 'Hunger',
    'tired'                 : 'Fatigue / Tiredness',
    # Respiratory
    'healthy_respiratory'   : 'Healthy Lungs',
    'pneumonia'             : 'Pneumonia',
    'bronchiolitis'         : 'Bronchiolitis',
    'respiratory_infection' : 'Respiratory Infection (URTI/LRTI)',
    'copd'                  : 'Chronic Respiratory Disease',
    'bronchiectasis'        : 'Bronchiectasis',
    'asthma'                : 'Asthma',
    # Cardiac
    'normal_cardiac'        : 'Normal Heart Sound',
    'heart_murmur'          : 'Heart Murmur (possible CHD)',
    'extrasystole'          : 'Extra Heartbeat (Extrasystole)',
    'extra_heart_sound'     : 'Extra Heart Sound',
}

# First aid per condition
FIRST_AID = {
    'belly_pain'            : 'Gently massage tummy clockwise. Try bicycle legs. Hold upright. See doctor if crying > 2 hours.',
    'burping'               : 'Hold baby upright on shoulder, gently pat back for 10-15 minutes.',
    'discomfort'            : 'Check diaper, clothing, room temperature. Offer feed.',
    'hungry'                : 'Feed the baby immediately. Hold upright 15 minutes after feeding.',
    'tired'                 : 'Move to quiet dark room. Swaddle gently. Rock slowly.',
    'pneumonia'             : 'This is a medical emergency. Take baby to doctor/hospital immediately.',
    'bronchiolitis'         : 'Keep baby upright. Give small frequent feeds. See doctor today.',
    'respiratory_infection' : 'Keep baby comfortable. Ensure hydration. See doctor today.',
    'heart_murmur'          : 'This needs medical evaluation. See a pediatric cardiologist.',
    'extrasystole'          : 'Monitor baby closely. See a doctor today.',
}

SEE_DOCTOR_IF = {
    'belly_pain'            : 'Crying does not stop after 2 hours | Blood in stool | Fever > 38°C',
    'burping'               : 'Baby vomits forcefully | Refuses all feeds',
    'hungry'                : 'Baby refuses to feed | Not gaining weight',
    'tired'                 : 'Baby is abnormally limp | Cannot be awakened',
    'pneumonia'             : 'ANY breathing difficulty — go immediately',
    'bronchiolitis'         : 'Fast breathing | Blue lips | Refusing feeds',
    'heart_murmur'          : 'Blue lips/fingertips | Poor feeding | Excessive sweating',
}


# ─── Load a saved model ───────────────────────────────────────────────────────
def load_model(model_path, scaler_path, encoder_path, features_path):
    with open(model_path,   'rb') as f: model   = pickle.load(f)
    with open(scaler_path,  'rb') as f: scaler  = pickle.load(f)
    with open(encoder_path, 'rb') as f: le      = pickle.load(f)
    with open(features_path,'rb') as f: feats   = pickle.load(f)
    return model, scaler, le, feats


def load_all_models():
    """Load all available trained models."""
    models = {}

    model_configs = {
        'cry': ('model_cry.pkl', 'scaler_cry.pkl',
                'encoder_cry.pkl', 'features_cry.pkl'),
        'respiratory': ('model_respiratory.pkl', 'scaler_respiratory.pkl',
                        'encoder_respiratory.pkl', 'features_respiratory.pkl'),
        'cardiac': ('model_cardiac.pkl', 'scaler_cardiac.pkl',
                    'encoder_cardiac.pkl', 'features_cardiac.pkl'),
    }

    for name, (mp, sp, ep, fp) in model_configs.items():
        if all(os.path.exists(p) for p in [mp, sp, ep, fp]):
            try:
                models[name] = load_model(mp, sp, ep, fp)
                print(f"  ✅ Loaded {name} model | Classes: {models[name][2].classes_.tolist()}")
            except Exception as e:
                print(f"  ⚠️  Could not load {name} model: {e}")
        else:
            print(f"  ⚠️  {name} model not found — run train_all_models.py")

    return models


# ─── Extract features from audio ─────────────────────────────────────────────
def extract_features(audio_path, feature_cols):
    y, sr = librosa.load(audio_path, sr=22050, duration=10)

    raw = {}
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr)
    raw['SC_Mean']  = float(np.mean(sc))
    raw['SC_Std']   = float(np.std(sc))

    rms = librosa.feature.rms(y=y)
    raw['RMS_Mean'] = float(np.mean(rms))
    raw['RMS_Std']  = float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y)
    raw['ZCR_Mean'] = float(np.mean(zcr))

    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    raw['SBAN_Mean'] = float(np.mean(sb))

    sc2 = librosa.feature.spectral_contrast(y=y, sr=sr)
    raw['SCON_Mean'] = float(np.mean(sc2))

    ro  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    raw['Rolloff_Mean'] = float(np.mean(ro))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    raw['MFCCs13Mean'] = float(np.mean(mfccs))
    for i in range(13):
        raw[f'MFCC_{i+1}'] = float(np.mean(mfccs[i]))
        raw[f'MFCCs{i+1}'] = float(np.mean(mfccs[i]))  # both naming conventions

    hop = 512
    ae  = np.array([max(abs(y[i:i+1024])) for i in range(0, len(y), hop)])
    raw['AE_Mean']                   = float(np.mean(ae))
    raw['Amplitude_Envelope_Mean']   = float(np.mean(ae))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    raw['Chroma_Mean'] = float(np.mean(chroma))

    # Align to feature cols
    aligned = {col: raw.get(col, 0.0) for col in feature_cols}
    return raw, aligned


# ─── Run prediction from one model ───────────────────────────────────────────
def predict_one(model, scaler, le, feature_cols, audio_features):
    vector = np.array([float(audio_features.get(col, 0.0))
                       for col in feature_cols])
    scaled     = scaler.transform(vector.reshape(1, -1))
    pred_idx   = model.predict(scaled)[0]
    pred_proba = model.predict_proba(scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    conf_pct   = round(float(max(pred_proba)) * 100, 1)

    class_probs = {
        le.classes_[i]: round(float(p)*100, 1)
        for i, p in enumerate(pred_proba)
    }

    is_uncertain = conf_pct < CONFIDENCE_THRESHOLD

    return {
        'label'        : pred_label,
        'display'      : DISEASE_DISPLAY.get(pred_label, pred_label.replace('_',' ').title()),
        'confidence'   : conf_pct,
        'is_uncertain' : is_uncertain,
        'class_probs'  : class_probs,
    }


# ─── Synthetic motor features ─────────────────────────────────────────────────
def get_motor_features(video_path=None):
    if video_path and video_path != 'none':
        try:
            import cv2
            import mediapipe as mp
            mp_pose = mp.solutions.pose
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return _synthetic_motor()
            lw, rw = [], []
            count = 0
            with mp_pose.Pose(min_detection_confidence=0.5,
                              model_complexity=0) as pose:
                while cap.isOpened() and count < 150:
                    ret, frame = cap.read()
                    if not ret: break
                    count += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        lw.append((lm[15].x, lm[15].y))
                        rw.append((lm[16].x, lm[16].y))
            cap.release()
            if len(lw) < 5:
                return _synthetic_motor()
            def vel(pos):
                p = np.array(pos)
                return float(np.mean(np.sqrt(np.sum(np.diff(p,axis=0)**2,axis=1))))
            lw_v, rw_v = vel(lw), vel(rw)
            asym = abs(lw_v-rw_v)/(max(lw_v,rw_v)+1e-8)
            return {
                'Limb_Asymmetry'          : round(asym, 4),
                'Activity_Level'          : round((lw_v+rw_v)/2, 6),
                'Repetitive_Motion_Index' : round(min(asym*2, 1.0), 4),
                'Movement_Frequency'      : 1.5,
                'source'                  : 'video',
            }
        except:
            return _synthetic_motor()
    return _synthetic_motor()


def _synthetic_motor():
    rng = np.random.default_rng()
    return {
        'Limb_Asymmetry'          : round(float(rng.uniform(0.05, 0.25)), 4),
        'Activity_Level'          : round(float(rng.uniform(0.005, 0.02)), 6),
        'Repetitive_Motion_Index' : round(float(rng.uniform(0.05, 0.25)), 4),
        'Movement_Frequency'      : round(float(rng.uniform(0.5, 2.5)), 4),
        'source'                  : 'synthetic',
    }


# ─── Generate report via Groq ─────────────────────────────────────────────────
def generate_report(cry_result, resp_result, cardiac_result,
                    motor_features, audio_features, groq_api_key):

    cry_text  = f"{cry_result['display']} ({cry_result['confidence']}%)" if cry_result else "Not available"
    resp_text = f"{resp_result['display']} ({resp_result['confidence']}%)" if resp_result else "Not available"
    card_text = f"{cardiac_result['display']} ({cardiac_result['confidence']}%)" if cardiac_result else "Not available"

    cry_uncertain  = cry_result['is_uncertain'] if cry_result else True
    resp_uncertain = resp_result['is_uncertain'] if resp_result else True
    card_uncertain = cardiac_result['is_uncertain'] if cardiac_result else True

    prompt = f"""You are ECD-Nexus 3.0, a pediatric AI health assistant.

Three specialized AI models analyzed a baby's audio:

CRY ANALYSIS MODEL:
  Result: {cry_text}
  Uncertain: {cry_uncertain}
  Probabilities: {cry_result['class_probs'] if cry_result else 'N/A'}

RESPIRATORY MODEL (lung sounds):
  Result: {resp_text}
  Uncertain: {resp_uncertain}

CARDIAC MODEL (heart sounds):
  Result: {card_text}
  Uncertain: {card_uncertain}

MOTOR ANALYSIS (MediaPipe):
  Limb Asymmetry: {motor_features.get('Limb_Asymmetry', 0):.4f}
  Activity Level: {motor_features.get('Activity_Level', 0):.6f}
  Source: {motor_features.get('source', 'synthetic')}

AUDIO BIOMARKERS:
  SC_Mean (cry pitch): {audio_features.get('SC_Mean', 0):.2f} Hz
  RMS_Mean (loudness): {audio_features.get('RMS_Mean', 0):.6f}

STRICT RULES YOU MUST FOLLOW:
1. If a model says uncertain — clearly state that result is not reliable
2. Never claim a disease is confirmed — say "possible" or "risk of"
3. If all models are uncertain — say "INCONCLUSIVE — please see a doctor"
4. Be warm and simple — parents are reading this

Write a report with these sections:

## 🔍 What the AI Found
Simple 2-3 sentence summary of all 3 model results.

## 🎵 Cry Analysis
What the cry sound suggests.

## 🫁 Respiratory Analysis
What the lung sound pattern suggests.

## ❤️ Cardiac Analysis
What the heart sound pattern suggests.

## 🎥 Movement Analysis
What the body movement suggests.

## ✅ What To Do Right Now
Clear first aid steps based on the most likely finding.

## 🏥 When To See A Doctor
Specific warning signs.

## ⚠️ Important
Always end with: "This AI tool screens for possible risks only. Always consult a qualified doctor or ASHA worker for diagnosis."

Keep under 500 words. Be honest about uncertainty. Be warm and reassuring.
"""

    client = Groq(api_key=groq_api_key)
    candidate_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    last_error = None
    for attempt in range(3):
        for model_name in candidate_models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.3,
                    max_tokens=1200,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if '429' in str(e) or 'rate' in str(e).lower():
                    time.sleep(25)
                    break
                if '404' in str(e) or 'model_not_found' in str(e).lower() or 'decommissioned' in str(e).lower():
                    continue
                if '401' in str(e) or 'invalid_api_key' in str(e).lower():
                    break
                continue

    if last_error is not None:
        return (
            "## 🔍 Local Clinical Summary\n\n"
            "The Groq report service is currently unavailable for this account, so this summary was generated from the model outputs locally.\n\n"
            f"- Cry analysis: {cry_text}\n"
            f"- Respiratory analysis: {resp_text}\n"
            f"- Cardiac analysis: {card_text}\n\n"
            "The AI output should be treated as a screening tool only. Please consult a qualified doctor or ASHA worker for diagnosis and follow-up.\n\n"
            "This AI tool screens for possible risks only. Always consult a qualified doctor or ASHA worker for diagnosis."
        )

    return "Report generation failed. Please try again."


# ─── Full inference pipeline ──────────────────────────────────────────────────
def run_full_inference(audio_path, video_path, models, groq_api_key):
    """
    Run all 3 models on the uploaded audio file.
    Returns complete result dict.
    """
    print("🎵 Extracting audio features...")

    # Use cry model features as base
    if 'cry' in models:
        _, _, _, cry_feat_cols = models['cry']
        audio_raw, _ = extract_features(audio_path, cry_feat_cols)
    else:
        _, _, _, resp_feat_cols = list(models.values())[0]
        audio_raw, _ = extract_features(audio_path, resp_feat_cols)

    results = {}

    # Run each available model
    for model_name, (model, scaler, le, feat_cols) in models.items():
        print(f"🤖 Running {model_name} classifier...")
        _, aligned = extract_features(audio_path, feat_cols)
        results[model_name] = predict_one(model, scaler, le, feat_cols, aligned)

    # Motor features
    print("🎥 Extracting motor features...")
    motor = get_motor_features(video_path)

    # Generate report
    print("📝 Generating clinical report...")
    report = generate_report(
        cry_result     = results.get('cry'),
        resp_result    = results.get('respiratory'),
        cardiac_result = results.get('cardiac'),
        motor_features = motor,
        audio_features = audio_raw,
        groq_api_key   = groq_api_key,
    )

    # Overall risk — highest urgency across models
    any_uncertain = any(r['is_uncertain'] for r in results.values())
    high_conf_results = {k: v for k, v in results.items() if not v['is_uncertain']}

    return {
        'results'      : results,
        'motor'        : motor,
        'audio_raw'    : audio_raw,
        'report'       : report,
        'any_uncertain': any_uncertain,
        'high_conf'    : high_conf_results,
    }
