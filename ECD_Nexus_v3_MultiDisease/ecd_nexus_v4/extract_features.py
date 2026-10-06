# ─────────────────────────────────────────────────────────────────────────────
# ECD-Nexus 3.0 | extract_features.py (FIXED)
# Extracts Librosa audio features from all 3 datasets:
#   1. Donate-A-Cry (cry states)
#   2. ICBHI 2017 (respiratory) - reads crackle/wheeze from .txt files
#   3. Pascal Heart Sound (cardiac)
# Run: python extract_features.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

# ─── UPDATE THESE PATHS ───────────────────────────────────────────────────────
DONATE_CRY_WAV_DIR = r"C:\Users\namth\Downloads\ECD_Nexus_v3\ECD_Nexus_v3_MultiDisease\ecd_nexus_v4\donate cry\donateacry_corpus"
ICBHI_DIR          = r"C:\Users\namth\Downloads\ECD_Nexus_v3\ECD_Nexus_v3_MultiDisease\ecd_nexus_v4\icbhi\ICBHI_final_database"
PASCAL_DIR         = r"C:\Users\namth\Downloads\ECD_Nexus_v3\ECD_Nexus_v3_MultiDisease\ecd_nexus_v4\pascal_heart"
DONATE_CRY_CSV     = r"C:\Users\namth\Downloads\ECD_Nexus_v3\ECD_Nexus_v3_MultiDisease\ecd_nexus_v4\donateacry-corpus_features_final.csv"

CRY_OUTPUT      = "cry_features.csv"
RESP_OUTPUT     = "respiratory_features.csv"
CARDIAC_OUTPUT  = "cardiac_features.csv"

# ─── Core feature extractor ───────────────────────────────────────────────────
def extract_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=22050, duration=10)
        if len(y) < 1000:
            return None
        feats = {}
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats['SC_Mean']  = float(np.mean(sc))
        feats['SC_Std']   = float(np.std(sc))
        rms = librosa.feature.rms(y=y)
        feats['RMS_Mean'] = float(np.mean(rms))
        feats['RMS_Std']  = float(np.std(rms))
        zcr = librosa.feature.zero_crossing_rate(y)
        feats['ZCR_Mean'] = float(np.mean(zcr))
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        feats['SBAN_Mean'] = float(np.mean(sb))
        sc2 = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats['SCON_Mean'] = float(np.mean(sc2))
        ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        feats['Rolloff_Mean'] = float(np.mean(ro))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feats['MFCCs13Mean'] = float(np.mean(mfccs))
        for i in range(13):
            feats[f'MFCC_{i+1}'] = float(np.mean(mfccs[i]))
        hop = 512
        ae = np.array([max(abs(y[i:i+1024])) for i in range(0, len(y), hop)])
        feats['AE_Mean'] = float(np.mean(ae))
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats['Chroma_Mean'] = float(np.mean(chroma))
        return feats
    except Exception as e:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1: Donate-A-Cry
# ══════════════════════════════════════════════════════════════════════════════

DONATE_LABEL_MAP = {
    'belly_pain' : 'belly_pain',
    'burping'    : 'burping',
    'discomfort' : 'discomfort',
    'hungry'     : 'hungry',
    'tired'      : 'tired',
    'belly pain' : 'belly_pain',
    'hunger'     : 'hungry',
    'pain'       : 'belly_pain',
}

def process_donate_cry(wav_dir):
    print(f"\n{'='*60}")
    print(f"Processing Donate-A-Cry: {wav_dir}")
    print(f"{'='*60}")
    records = []
    for folder in os.listdir(wav_dir):
        folder_path = os.path.join(wav_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label_key = folder.lower().strip()
        label = DONATE_LABEL_MAP.get(label_key, label_key)
        wav_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.wav','.mp3','.caf','.ogg'))]
        print(f"  Folder: {folder} → Label: {label} ({len(wav_files)} files)")
        for i, fname in enumerate(wav_files):
            fpath = os.path.join(folder_path, fname)
            feats = extract_features(fpath)
            if feats:
                feats['label']   = label
                feats['filename']= fname
                feats['dataset'] = 'donate_cry'
                records.append(feats)
            if (i+1) % 50 == 0:
                print(f"    Processed {i+1}/{len(wav_files)}...")
    df = pd.DataFrame(records)
    print(f"\n  ✅ Extracted {len(df)} samples")
    if len(df) > 0:
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2: ICBHI 2017 Respiratory
# ══════════════════════════════════════════════════════════════════════════════

def get_icbhi_label(txt_path):
    """
    Read ICBHI annotation .txt file and return respiratory label.
    Each line: start_time  end_time  crackle(0/1)  wheeze(0/1)
    """
    has_crackle = False
    has_wheeze  = False
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        crackle = int(parts[2])
                        wheeze  = int(parts[3])
                        if crackle == 1:
                            has_crackle = True
                        if wheeze == 1:
                            has_wheeze = True
                    except:
                        continue
    except:
        return None

    if has_crackle and has_wheeze:
        return 'crackle_and_wheeze'
    elif has_crackle:
        return 'crackle'
    elif has_wheeze:
        return 'wheeze'
    else:
        return 'normal_respiratory'


def process_icbhi(icbhi_dir):
    print(f"\n{'='*60}")
    print(f"Processing ICBHI 2017: {icbhi_dir}")
    print(f"{'='*60}")

    wav_files = []
    for root, dirs, files in os.walk(icbhi_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))

    print(f"  Found {len(wav_files)} WAV files")

    records  = []
    no_label = 0

    for i, fpath in enumerate(wav_files):
        fname    = os.path.basename(fpath)
        txt_path = fpath.replace('.wav', '.txt')

        if not os.path.exists(txt_path):
            no_label += 1
            continue

        label = get_icbhi_label(txt_path)
        if not label:
            no_label += 1
            continue

        feats = extract_features(fpath)
        if feats:
            feats['label']   = label
            feats['filename']= fname
            feats['dataset'] = 'icbhi'
            records.append(feats)

        if (i+1) % 100 == 0:
            print(f"  Processed {i+1}/{len(wav_files)}...")

    df = pd.DataFrame(records)
    print(f"\n  ✅ Extracted {len(df)} samples ({no_label} skipped)")
    if len(df) > 0:
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 3: Pascal Heart Sound
# ══════════════════════════════════════════════════════════════════════════════

PASCAL_LABEL_NORM = {
    'normal'       : 'normal_cardiac',
    'murmur'       : 'heart_murmur',
    'extrastole'   : 'extrasystole',
    'extrasystole' : 'extrasystole',
    'extrahs'      : 'extra_heart_sound',
    'artifact'     : None,
    'unlabelled'   : None,
}

def load_pascal_labels(csv_path):
    label_map = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # fname is like "set_a/artifact__201012172012.wav"
            # we just need the filename part
            fname = os.path.basename(str(row['fname']).strip())
            label = str(row['label']).strip().lower()
            label_map[fname] = label
        print(f"  Loaded {len(label_map)} labels from CSV")
    except Exception as e:
        print(f"  CSV load error: {e}")
    return label_map


def process_pascal(pascal_dir):
    print(f"\n{'='*60}")
    print(f"Processing Pascal Heart Sound: {pascal_dir}")
    print(f"{'='*60}")
    records = []
    for set_name in ['set_a', 'set_b']:
        set_dir = os.path.join(pascal_dir, set_name)
        if not os.path.exists(set_dir):
            print(f"  Skipping {set_name} — not found")
            continue
        csv_path  = os.path.join(pascal_dir, f'{set_name}.csv')
        label_map = load_pascal_labels(csv_path) if os.path.exists(csv_path) else {}
        wav_files = []
        for root, dirs, files in os.walk(set_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, f))
        print(f"\n  {set_name}: {len(wav_files)} WAV files")
        for i, fpath in enumerate(wav_files):
            fname     = os.path.basename(fpath)
            raw_label = label_map.get(fname, '').lower()
            if not raw_label:
                parent = os.path.basename(os.path.dirname(fpath)).lower()
                for key in PASCAL_LABEL_NORM:
                    if key in parent:
                        raw_label = key
                        break
            label = PASCAL_LABEL_NORM.get(raw_label, raw_label)
            if not label:
                continue
            feats = extract_features(fpath)
            if feats:
                feats['label']   = label
                feats['filename']= fname
                feats['dataset'] = f'pascal_{set_name}'
                records.append(feats)
            if (i+1) % 100 == 0:
                print(f"    Processed {i+1}/{len(wav_files)}...")
    df = pd.DataFrame(records)
    print(f"\n  ✅ Extracted {len(df)} cardiac samples")
    if len(df) > 0:
        print(f"  Labels: {df['label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  ECD-Nexus 3.0 — Multi-Dataset Feature Extraction")
    print("="*60)

    # 1. Donate-A-Cry
    if os.path.exists(DONATE_CRY_WAV_DIR):
        df_cry = process_donate_cry(DONATE_CRY_WAV_DIR)
        if len(df_cry) > 0:
            df_cry.to_csv(CRY_OUTPUT, index=False)
            print(f"\n✅ Saved: {CRY_OUTPUT} ({len(df_cry)} samples)")
    else:
        print(f"\n⚠️  Donate-A-Cry not found: {DONATE_CRY_WAV_DIR}")

    # 2. ICBHI Respiratory
    if os.path.exists(ICBHI_DIR):
        df_resp = process_icbhi(ICBHI_DIR)
        if len(df_resp) > 0:
            df_resp.to_csv(RESP_OUTPUT, index=False)
            print(f"\n✅ Saved: {RESP_OUTPUT} ({len(df_resp)} samples)")
    else:
        print(f"\n⚠️  ICBHI not found: {ICBHI_DIR}")

    # 3. Pascal Cardiac
    if os.path.exists(PASCAL_DIR):
        df_cardiac = process_pascal(PASCAL_DIR)
        if len(df_cardiac) > 0:
            df_cardiac.to_csv(CARDIAC_OUTPUT, index=False)
            print(f"\n✅ Saved: {CARDIAC_OUTPUT} ({len(df_cardiac)} samples)")
    else:
        print(f"\n⚠️  Pascal not found: {PASCAL_DIR}")

    print("\n" + "="*60)
    print("  Feature extraction complete!")
    print("  Next step: python train_all_models.py")
    print("="*60)


if __name__ == '__main__':
    main()
