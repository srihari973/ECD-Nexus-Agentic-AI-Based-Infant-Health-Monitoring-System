# ─────────────────────────────────────────────────────────────────────────────
# ECD-Nexus 3.0 | train_all_models.py
# Trains 3 specialized XGBoost models:
#   Model 1: Cry Classifier (Donate-A-Cry)
#   Model 2: Respiratory Classifier (ICBHI)
#   Model 3: Cardiac Classifier (Pascal)
# Run AFTER extract_features.py
# Run: python train_all_models.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Paths to pre-extracted CSVs (from extract_features.py)
DONATE_CRY_CSV  = "donateacry-corpus_features_final.csv"
CRY_FEATURES    = "cry_features.csv"
RESP_FEATURES   = "respiratory_features.csv"
CARDIAC_FEATURES= "cardiac_features.csv"

# Feature columns used for training
AUDIO_FEATURE_COLS = [
    'SC_Mean', 'SC_Std', 'RMS_Mean', 'RMS_Std', 'ZCR_Mean',
    'SBAN_Mean', 'SCON_Mean', 'Rolloff_Mean', 'MFCCs13Mean',
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5',
    'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10',
    'MFCC_11', 'MFCC_12', 'MFCC_13', 'AE_Mean', 'Chroma_Mean',
]

# Fallback cols for donate-a-cry original CSV
DONATE_COLS = [
    'SC_Mean', 'RMS_Mean', 'ZCR_Mean', 'SBAN_Mean',
    'SCON_Mean', 'MFCCs13Mean', 'Amplitude_Envelope_Mean',
    'MFCCs1','MFCCs2','MFCCs3','MFCCs4','MFCCs5',
    'MFCCs6','MFCCs7','MFCCs8','MFCCs9','MFCCs10',
    'MFCCs11','MFCCs12','MFCCs13',
]


def clean_df(df, label_col='label'):
    df = df.dropna(how='all')
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col] != 'nan']
    df = df[df[label_col] != '']
    df = df.reset_index(drop=True)
    return df


def get_feature_cols(df, preferred_cols):
    available = [c for c in preferred_cols if c in df.columns]
    if len(available) < 5:
        # Try any numeric column except label/filename/dataset
        skip = {'label','filename','dataset','Cry_Reason'}
        available = [c for c in df.columns
                     if c not in skip and df[c].dtype in ['float64','int64','float32']]
    return available


def augment(X, y, target=100):
    """Gaussian noise augmentation for minority classes."""
    X_aug, y_aug = list(X), list(y)
    for label in np.unique(y):
        idx    = np.where(np.array(y) == label)[0]
        needed = max(0, target - len(idx))
        if needed > 0:
            base = X[idx]
            for _ in range(needed):
                s = base[np.random.randint(len(base))].copy()
                n = np.random.normal(0, 0.03 * np.std(base, axis=0) + 1e-8)
                X_aug.append(s + n)
                y_aug.append(label)
    return np.array(X_aug), np.array(y_aug)


def train_model(df, feature_cols, label_col, model_name,
                model_out, scaler_out, encoder_out, features_out,
                augment_target=100):

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    df = clean_df(df, label_col)
    available = get_feature_cols(df, feature_cols)
    print(f"  Features  : {len(available)}")
    print(f"  Samples   : {len(df)}")
    print(f"  Labels    : {df[label_col].value_counts().to_dict()}")

    # Fill nulls
    X_df = df[available].copy()
    for col in available:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values
    y_labels = df[label_col].values

    le = LabelEncoder()
    y  = le.fit_transform(y_labels)
    print(f"  Classes   : {le.classes_.tolist()}")

    # Augment minority classes
    X_aug, y_aug = augment(X, y, target=augment_target)
    print(f"  After aug : {len(X_aug)} samples")

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_aug)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_aug, test_size=0.2, random_state=42, stratify=y_aug)

    # Class weights
    cw = compute_class_weight('balanced', classes=np.unique(y_aug), y=y_aug)
    sw = np.array([cw[yi] for yi in y_train])

    # Train XGBoost
    print(f"\n  Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              sample_weight=sw,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # Evaluate
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc*100:.1f}%")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, zero_division=0))

    # Cross validation
    cv = cross_val_score(model, X_scaled, y_aug, cv=5, scoring='accuracy')
    print(f"  5-Fold CV : {[round(s*100,1) for s in cv]}")
    print(f"  CV Mean   : {cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(le.classes_)*1.2),
                                    max(5, len(le.classes_))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f'{model_name} — Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    cm_path = f"{model_name.lower().replace(' ','_')}_confusion.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  ✅ Saved: {cm_path}")

    # Save model
    with open(model_out,   'wb') as f: pickle.dump(model, f)
    with open(scaler_out,  'wb') as f: pickle.dump(scaler, f)
    with open(encoder_out, 'wb') as f: pickle.dump(le, f)
    with open(features_out,'wb') as f: pickle.dump(available, f)

    print(f"  ✅ Model saved: {model_out}")

    return {
        'model_name' : model_name,
        'accuracy'   : round(acc * 100, 1),
        'cv_mean'    : round(cv.mean() * 100, 1),
        'cv_std'     : round(cv.std() * 100, 1),
        'classes'    : le.classes_.tolist(),
        'n_samples'  : len(X_aug),
        'n_features' : len(available),
    }


def load_cry_data():
    """Load cry data — try extracted CSV first, fall back to original CSV."""
    if os.path.exists(CRY_FEATURES):
        print(f"Loading cry features: {CRY_FEATURES}")
        df = pd.read_csv(CRY_FEATURES)
        return df, 'label', AUDIO_FEATURE_COLS
    elif os.path.exists(DONATE_CRY_CSV):
        print(f"Loading Donate-A-Cry CSV: {DONATE_CRY_CSV}")
        df = pd.read_csv(DONATE_CRY_CSV)
        # Map numeric labels
        LABEL_MAP = {'0':'belly_pain','1':'burping','2':'discomfort','3':'hungry','4':'tired'}
        df['label'] = df['Cry_Reason'].astype(str).map(LABEL_MAP).fillna(df['Cry_Reason'].astype(str))
        return df, 'label', DONATE_COLS
    else:
        raise FileNotFoundError("No cry dataset found. Run extract_features.py first.")


import os

def main():
    print("\n" + "="*60)
    print("  ECD-Nexus 3.0 — Multi-Model Training Pipeline")
    print("="*60)

    results = []

    # ── MODEL 1: Cry Classifier ───────────────────────────────────────────────
    try:
        df_cry, label_col, feat_cols = load_cry_data()
        r = train_model(
            df=df_cry, feature_cols=feat_cols, label_col=label_col,
            model_name="Cry Classifier",
            model_out="model_cry.pkl",
            scaler_out="scaler_cry.pkl",
            encoder_out="encoder_cry.pkl",
            features_out="features_cry.pkl",
            augment_target=100,
        )
        results.append(r)
    except Exception as e:
        print(f"\n❌ Cry model failed: {e}")
        import traceback; traceback.print_exc()

    # ── MODEL 2: Respiratory Classifier ──────────────────────────────────────
    if os.path.exists(RESP_FEATURES):
        try:
            df_resp = pd.read_csv(RESP_FEATURES)
            r = train_model(
                df=df_resp, feature_cols=AUDIO_FEATURE_COLS, label_col='label',
                model_name="Respiratory Classifier",
                model_out="model_respiratory.pkl",
                scaler_out="scaler_respiratory.pkl",
                encoder_out="encoder_respiratory.pkl",
                features_out="features_respiratory.pkl",
                augment_target=120,
            )
            results.append(r)
        except Exception as e:
            print(f"\n❌ Respiratory model failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n⚠️  {RESP_FEATURES} not found. Run extract_features.py first.")

    # ── MODEL 3: Cardiac Classifier ───────────────────────────────────────────
    if os.path.exists(CARDIAC_FEATURES):
        try:
            df_cardiac = pd.read_csv(CARDIAC_FEATURES)
            r = train_model(
                df=df_cardiac, feature_cols=AUDIO_FEATURE_COLS, label_col='label',
                model_name="Cardiac Classifier",
                model_out="model_cardiac.pkl",
                scaler_out="scaler_cardiac.pkl",
                encoder_out="encoder_cardiac.pkl",
                features_out="features_cardiac.pkl",
                augment_target=120,
            )
            results.append(r)
        except Exception as e:
            print(f"\n❌ Cardiac model failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n⚠️  {CARDIAC_FEATURES} not found. Run extract_features.py first.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING SUMMARY")
    print("="*60)
    for r in results:
        print(f"\n  {r['model_name']}")
        print(f"    Accuracy  : {r['accuracy']}%")
        print(f"    CV Mean   : {r['cv_mean']}% ± {r['cv_std']}%")
        print(f"    Classes   : {r['classes']}")
        print(f"    Samples   : {r['n_samples']}")

    print("\n✅ All models trained!")
    print("   Next step: streamlit run app.py")


if __name__ == '__main__':
    main()
