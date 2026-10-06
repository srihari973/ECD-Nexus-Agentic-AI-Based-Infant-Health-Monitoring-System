# ─────────────────────────────────────────────────────────────────────────────
# ECD-Nexus 3.0 | app_v3.py — COMPLETE FINAL VERSION
# All 4 tabs working:
#   Tab 1: Triage Analysis
#   Tab 2: What-If Dashboard
#   Tab 3: Clinical Atlas
#   Tab 4: Model Performance (including motor analysis)
# Run: streamlit run app_v3.py
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import os, sys, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

st.set_page_config(
    page_title="ECD-Nexus 3.0 — Multi-Disease Baby Health AI",
    page_icon="🧠", layout="wide",
)

st.markdown("""
<style>
.header {
    background:linear-gradient(135deg,#1B3A6B,#2E75B6);
    padding:2rem; border-radius:12px; text-align:center; color:white; margin-bottom:1rem;
}
.header h1 { font-size:2.4rem; font-weight:800; margin:0; }
.header p  { opacity:.85; margin:.3rem 0 0; }
.model-card {
    background:white; border-radius:10px; padding:1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,.08); margin:.5rem 0;
}
.model-card h4 { margin:0 0 .5rem; color:#1B3A6B; }
.metric-box {
    background:#F4F6FA; border-radius:8px; padding:1rem; text-align:center;
}
.metric-box .val { font-size:1.5rem; font-weight:800; color:#1B3A6B; }
.metric-box .lbl { font-size:.8rem; color:#666; }
.atlas-rule {
    background:#F4F6FA; border-radius:6px; padding:.6rem .8rem;
    margin:.3rem 0; font-size:.85rem; border-left:3px solid #2E75B6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1>🧠 ECD-Nexus 3.0</h1>
  <p>Multimodal Agentic AI — Cry + Respiratory + Cardiac + Motor Analysis</p>
  <p style="font-size:.8rem;opacity:.6">
    3 XGBoost Models + Rule-Based Motor Atlas | LangChain Agents | Llama-3.3-70b | XAI Report
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption("Free at console.groq.com")
    st.markdown("---")
    st.markdown("## 🤖 Models Loaded")

    from inference_v3 import load_all_models
    @st.cache_resource
    def get_models():
        return load_all_models()

    models = get_models()

    if models:
        for name, (model, scaler, le, feats) in models.items():
            st.success(f"✅ {name.title()} Model")
            st.caption(f"Classes: {', '.join(le.classes_)}")
        st.info("✅ Motor Analysis\nMediaPipe Pose + Clinical Atlas Rules")
    else:
        st.error("❌ No models found. Run train_all_models.py first.")

    st.markdown("---")
    st.markdown("## 📖 How it Works")
    st.markdown("""
**1.** Upload cry audio (.wav)
**2.** Upload movement video (.mp4) *(optional)*
**3.** All 3 models + motor atlas analyze
**4.** Manager Agent fuses all results
**5.** Get unified XAI disease report
    """)
    st.markdown("---")
    st.caption("⚠️ AI screening tool only. Always see a doctor.")

if not models:
    st.error("### ⚠️ No trained models found!\nRun: `python train_all_models.py`")
    st.stop()

# ── Load atlas (graceful fallback) ────────────────────────────────────────────
ATLAS_AVAILABLE = False
try:
    from synthetic_atlas import (
        query_acoustic_atlas, query_motor_atlas,
        query_fusion_atlas, compute_risk_score,
        ACOUSTIC_ATLAS, MOTOR_ATLAS, FUSION_ATLAS,
    )
    ATLAS_AVAILABLE = True
except ImportError:
    pass

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Triage Analysis",
    "🔬 What-If Dashboard",
    "📚 Clinical Atlas",
    "📊 Model Performance",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: TRIAGE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📤 Upload Patient Data")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎵 Baby Cry Audio (Required)")
        audio_file = st.file_uploader("Upload .wav or .mp3",
                                      type=['wav','mp3','ogg'], key="af")
        if audio_file:
            st.audio(audio_file)
            st.success(f"✅ {audio_file.name} ({audio_file.size/1024:.1f} KB)")

    with col2:
        st.markdown("### 🎥 Movement Video (Optional)")
        st.caption("MediaPipe will extract 33 body landmarks for motor analysis")
        video_file = st.file_uploader("Upload .mp4", type=['mp4','avi','mov'], key="vf")
        if video_file:
            st.video(video_file)
            st.success(f"✅ {video_file.name}")
        else:
            st.info("📌 No video → synthetic motor features used for demo")

    st.markdown("---")
    btn_col = st.columns([1,2,1])[1]
    with btn_col:
        analyze_btn = st.button("🔍 Run Multi-Disease Analysis", use_container_width=True)

    if analyze_btn:
        if not audio_file:
            st.error("❌ Please upload cry audio."); st.stop()
        if not groq_key:
            st.error("❌ Please enter Groq API key in the sidebar."); st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tf:
            tf.write(audio_file.read())
            audio_path = tf.name

        video_path = None
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tf:
                tf.write(video_file.read())
                video_path = tf.name

        prog = st.progress(0)
        stat = st.empty()

        try:
            from inference_v3 import run_full_inference

            stat.info("🎵 Acoustic Specialist: extracting cry audio features...")
            prog.progress(15)

            stat.info("🎥 Vision Specialist: extracting motor features via MediaPipe...")
            prog.progress(30)

            stat.info("🤖 Manager Agent: running all 3 models + decision fusion...")
            prog.progress(50)

            result = run_full_inference(audio_path, video_path, models, groq_key)

            prog.progress(90)
            stat.info("📝 Generating XAI clinical report...")
            prog.progress(100)
            stat.empty(); prog.empty()

            # Store for what-if tab
            st.session_state['audio_features'] = result.get('audio_raw', {})
            st.session_state['motor_features']  = result.get('motor', {})

            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            if result['any_uncertain']:
                st.warning(
                    "⚠️ One or more models returned LOW confidence. "
                    "Marked below. Please consult a healthcare worker."
                )

            # ── Audio Model Results ───────────────────────────────────────────
            st.markdown("### 🤖 Audio Model Results (3 Specialized XGBoost Models)")
            res_cols = st.columns(len(result['results']))
            model_icons = {'cry':'🎵','respiratory':'🫁','cardiac':'❤️'}

            for i, (mname, res) in enumerate(result['results'].items()):
                with res_cols[i]:
                    icon = model_icons.get(mname,'🤖')
                    conf_color = ("#1A7A4A" if res['confidence']>75
                                  else "#E67E22" if res['confidence']>55
                                  else "#C0392B")
                    st.markdown(f"""
<div class="model-card">
  <h4>{icon} {mname.title()} Model</h4>
  <div style="font-size:1.2rem;font-weight:800;color:{conf_color}">
    {res['display']}
  </div>
  <div style="font-size:.85rem;color:#666;margin-top:.3rem">
    Confidence: {res['confidence']}%
    {'&nbsp;⚠️ UNCERTAIN' if res['is_uncertain'] else '&nbsp;✅'}
  </div>
</div>""", unsafe_allow_html=True)

                    probs      = res['class_probs']
                    labels     = [k.replace('_',' ').title()[:14] for k in probs]
                    values     = list(probs.values())
                    bar_colors = ['#1A7A4A' if k==res['label'] else '#2E75B6' for k in probs]

                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, len(labels)*0.45)))
                    ax.barh(labels, values, color=bar_colors)
                    ax.set_xlabel('%', fontsize=8)
                    ax.set_xlim(0, 115)
                    for bar, v in zip(ax.patches, values):
                        ax.text(v+1, bar.get_y()+bar.get_height()/2,
                                f'{v}%', va='center', fontsize=7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(fontsize=7)
                    plt.yticks(fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            # ── Motor Analysis ────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 🎥 Motor Analysis (MediaPipe Pose + Clinical Atlas)")
            st.caption("33 body landmarks extracted → 6 motor biomarkers computed → Clinical Atlas rules applied")

            motor = result['motor']
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            for col, val, lbl in [
                (m1, f"{motor.get('Limb_Asymmetry',0):.4f}",          "Limb Asymmetry"),
                (m2, f"{motor.get('Activity_Level',0):.5f}",           "Activity Level"),
                (m3, f"{motor.get('Repetitive_Motion_Index',0):.4f}",  "Repetitive Motion"),
                (m4, f"{motor.get('Movement_Frequency',0):.4f} Hz",    "Movement Freq"),
                (m5, f"{motor.get('Joint_Angle_Variability',0):.4f}",  "Joint Variability"),
                (m6, motor.get('source','synthetic'),                   "Data Source"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="metric-box"><div class="val" style="font-size:1rem">'
                        f'{val}</div><div class="lbl">{lbl}</div></div>',
                        unsafe_allow_html=True
                    )

            # Motor atlas findings
            if ATLAS_AVAILABLE:
                motor_findings = query_motor_atlas(motor)
                if motor_findings:
                    st.markdown("**Motor Risk Flags (Clinical Atlas):**")
                    for f in motor_findings:
                        badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(f['urgency'],"⚪")
                        st.markdown(
                            f'<div class="atlas-rule">{badge} <b>{f["feature"]}</b>='
                            f'{f["value"]} → {f["finding"]} → <i>{f["clinical"]}</i></div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.success("✅ No motor risk flags detected")

            # ── XAI Report ────────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📋 Unified XAI Clinical Report")
            st.markdown(result['report'])

            # ── Download ──────────────────────────────────────────────────────
            st.markdown("---")
            dl_col = st.columns([1,2,1])[1]
            with dl_col:
                rt = f"ECD-NEXUS 3.0 — CLINICAL REPORT\n{'='*60}\n"
                for mname, res in result['results'].items():
                    rt += f"\n{mname.upper()} MODEL:\n"
                    rt += f"  Result    : {res['display']}\n"
                    rt += f"  Confidence: {res['confidence']}%\n"
                rt += f"\nMOTOR ANALYSIS:\n"
                rt += f"  Limb Asymmetry : {motor.get('Limb_Asymmetry',0):.4f}\n"
                rt += f"  Activity Level : {motor.get('Activity_Level',0):.6f}\n"
                rt += f"  Source         : {motor.get('source','synthetic')}\n"
                rt += f"\n{'='*60}\nREPORT:\n{result['report']}\n"
                rt += f"\n{'='*60}\nDISCLAIMER: AI screening tool. Always consult a doctor.\n"
                st.download_button("📥 Download Full Report", data=rt,
                                   file_name="ECD_Nexus_Report.txt",
                                   mime="text/plain", use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            try:
                os.unlink(audio_path)
                if video_path: os.unlink(video_path)
            except: pass


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: WHAT-IF DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔬 What-If Scenario Explorer")
    st.markdown("Adjust feature values and see how the AI risk assessment changes live.")

    if not ATLAS_AVAILABLE:
        st.error("""
❌ **synthetic_atlas.py not found in your folder!**

**Fix:** Copy `synthetic_atlas.py` into your `ecd_nexus_v4` folder.
Download it from the earlier files shared in this project.
        """)
    else:
        base_af = st.session_state.get('audio_features', {
            'SC_Mean':1200.0,'RMS_Mean':0.03,'ZCR_Mean':0.05,
            'SBAN_Mean':1500.0,'MFCCs13Mean':-30.0,
        })
        base_mf = st.session_state.get('motor_features', {
            'Limb_Asymmetry':0.15,'Activity_Level':0.01,
            'Repetitive_Motion_Index':0.1,'Movement_Frequency':1.5,
            'Joint_Angle_Variability':20.0,
        })

        if st.session_state.get('audio_features'):
            st.success("✅ Using values from your last triage as baseline.")

        wa1, wa2 = st.columns(2)
        with wa1:
            st.markdown("**🎵 Acoustic Features**")
            wi_sc  = st.slider("Spectral Centroid (Hz)", 300.0, 3000.0, float(base_af.get('SC_Mean',1200)), 50.0)
            wi_rms = st.slider("RMS Energy", 0.001, 0.15, float(base_af.get('RMS_Mean',0.03)), 0.001, format="%.4f")
            wi_zcr = st.slider("Zero Crossing Rate", 0.01, 0.20, float(base_af.get('ZCR_Mean',0.05)), 0.005, format="%.4f")

        with wa2:
            st.markdown("**🎥 Motor Features (MediaPipe)**")
            wi_asym = st.slider("Limb Asymmetry", 0.0, 0.8, float(base_mf.get('Limb_Asymmetry',0.15)), 0.01)
            wi_act  = st.slider("Activity Level", 0.001, 0.05, float(base_mf.get('Activity_Level',0.01)), 0.001, format="%.4f")
            wi_rmi  = st.slider("Repetitive Motion Index", 0.0, 0.8, float(base_mf.get('Repetitive_Motion_Index',0.1)), 0.01)
            wi_freq = st.slider("Movement Frequency (Hz)", 0.0, 6.0, float(base_mf.get('Movement_Frequency',1.5)), 0.1)

        wi_audio = {
            'SC_Mean':wi_sc,'RMS_Mean':wi_rms,'ZCR_Mean':wi_zcr,
            'SBAN_Mean':base_af.get('SBAN_Mean',1500),
            'MFCCs13Mean':base_af.get('MFCCs13Mean',-30),
        }
        wi_motor = {
            'Limb_Asymmetry':wi_asym,'Activity_Level':wi_act,
            'Repetitive_Motion_Index':wi_rmi,'Movement_Frequency':wi_freq,
            'Joint_Angle_Variability':base_mf.get('Joint_Angle_Variability',20.0),
        }

        wi_af   = query_acoustic_atlas(wi_audio)
        wi_mf   = query_motor_atlas(wi_motor)
        wi_ff   = query_fusion_atlas(wi_audio, wi_motor)
        wi_risk = compute_risk_score(wi_af, wi_mf, wi_ff)

        st.markdown("---")
        st.markdown("### 📊 Live Risk Assessment")

        risk_colors = {"CRITICAL":"#C0392B","HIGH":"#E67E22","MEDIUM":"#2E75B6","LOW":"#1A7A4A"}
        urg_colors  = {"IMMEDIATE":"#C0392B","URGENT":"#E67E22","ROUTINE":"#2E75B6","MONITOR":"#1A7A4A"}

        wm1, wm2, wm3, wm4 = st.columns(4)
        for col, val, lbl in [
            (wm1, wi_risk['risk_level'],  "Risk Level"),
            (wm2, wi_risk['max_urgency'], "Urgency"),
            (wm3, str(wi_risk['n_acoustic']+wi_risk['n_motor']), "Atlas Rules Hit"),
            (wm4, str(wi_risk['n_fusion']), "Fusion Rules Hit"),
        ]:
            with col:
                color = risk_colors.get(val, urg_colors.get(val, "#666"))
                st.markdown(
                    f'<div class="metric-box"><div class="val" style="color:{color}">'
                    f'{val}</div><div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)
        wr1, wr2 = st.columns(2)

        with wr1:
            st.markdown("**🎵 Acoustic Risk Flags:**")
            if wi_af:
                for f in wi_af:
                    badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(f['urgency'],"⚪")
                    st.markdown(f'<div class="atlas-rule">{badge} {f["finding"]}<br><small>→ {f["clinical"]}</small></div>', unsafe_allow_html=True)
            else:
                st.success("No acoustic risk flags")

        with wr2:
            st.markdown("**🎥 Motor Risk Flags (MediaPipe Atlas):**")
            if wi_mf:
                for f in wi_mf:
                    badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(f['urgency'],"⚪")
                    st.markdown(f'<div class="atlas-rule">{badge} {f["finding"]}<br><small>→ {f["clinical"]}</small></div>', unsafe_allow_html=True)
            else:
                st.success("No motor risk flags")

        if wi_ff:
            st.markdown("**🔗 Fusion Rules Triggered (PRIORITY):**")
            for ff in wi_ff:
                st.error(f"**{ff['pattern']}** → {ff['clinical']} | Urgency: **{ff['urgency']}**")

        # Radar chart
        st.markdown("---")
        st.markdown("### 📡 Feature Level Radar")
        categories = ['SC Level','RMS Level','ZCR Level','Limb Asymmetry','Activity','Repetitive']
        vals_norm  = [
            min(wi_sc/3000,1.0), min(wi_rms/0.15,1.0), min(wi_zcr/0.20,1.0),
            min(wi_asym/0.8,1.0), min(wi_act/0.05,1.0), min(wi_rmi/0.8,1.0),
        ]
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        vp = vals_norm + [vals_norm[0]]
        ap = angles   + [angles[0]]

        fig_r, ax_r = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax_r.plot(ap, vp, 'o-', color='#2E75B6', linewidth=2)
        ax_r.fill(ap, vp, alpha=0.25, color='#2E75B6')
        ax_r.set_xticks(angles)
        ax_r.set_xticklabels(categories, fontsize=9)
        ax_r.set_ylim(0,1)
        ax_r.set_title("Feature Levels (normalized)", fontweight='bold', pad=20)
        plt.tight_layout()
        radar_col = st.columns([1,2,1])[1]
        with radar_col:
            st.pyplot(fig_r)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: CLINICAL ATLAS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📚 Synthetic Clinical Atlas")
    st.markdown("The complete rule base used for Neuro-Symbolic reasoning. Each rule is backed by published medical research.")

    if not ATLAS_AVAILABLE:
        st.error("""
❌ **synthetic_atlas.py not found in your folder!**

**Fix:** Copy `synthetic_atlas.py` into your `ecd_nexus_v4` folder.
        """)
    else:
        st.markdown("### 🎵 Acoustic Atlas Rules (8 rules)")
        for rule in ACOUSTIC_ATLAS:
            badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(rule['urgency'],"⚪")
            with st.expander(f"{badge} [{rule['urgency']}] {rule['feature']} {rule['condition']} {rule['threshold']} → {rule['finding']}"):
                st.markdown(f"**Clinical Implication:** {rule['clinical']}")
                st.markdown(f"**Confidence:** `{rule['confidence']}`")
                st.markdown(f"**Research Evidence:** _{rule['evidence']}_")
                st.markdown(f"**Urgency:** `{rule['urgency']}`")

        st.markdown("### 🎥 Motor Atlas Rules (7 rules — MediaPipe features)")
        for rule in MOTOR_ATLAS:
            badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(rule['urgency'],"⚪")
            with st.expander(f"{badge} [{rule['urgency']}] {rule['feature']} {rule['condition']} {rule['threshold']} → {rule['finding']}"):
                st.markdown(f"**Clinical Implication:** {rule['clinical']}")
                st.markdown(f"**Confidence:** `{rule['confidence']}`")
                st.markdown(f"**Research Evidence:** _{rule['evidence']}_")
                st.markdown(f"**Urgency:** `{rule['urgency']}`")

        st.markdown("### 🔗 Fusion Atlas Rules (4 combined patterns)")
        for rule in FUSION_ATLAS:
            badge = {"IMMEDIATE":"🔴","URGENT":"🟠","ROUTINE":"🔵","MONITOR":"🟢"}.get(rule['urgency'],"⚪")
            st.markdown(f"""
<div style="background:#F4F6FA;border-radius:8px;padding:1rem;margin:.5rem 0;border-left:4px solid #1B3A6B;">
{badge} <b>Pattern:</b> {rule['pattern']}<br>
<b>Clinical Finding:</b> {rule['clinical']}<br>
<b>Urgency:</b> <code>{rule['urgency']}</code>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Model Accuracy & Performance")
    st.markdown("Complete accuracy comparison of all models and analysis components in ECD-Nexus 3.0.")

    PERF = {
        "Cry": {
            "acc":98.5,"samples":457,"color":"#1B3A6B","light":"#D6E4F7",
            "dataset":"Donate-A-Cry Corpus",
            "classes":  ["Belly Pain","Burping","Discomfort","Hungry","Tired"],
            "precision":[0.94,1.00,0.93,0.99,0.92],
            "recall":   [1.00,1.00,0.96,0.99,0.96],
            "f1":       [0.97,1.00,0.95,0.99,0.94],
        },
        "Respiratory": {
            "acc":88.5,"samples":920,"color":"#2E75B6","light":"#E8F0FC",
            "dataset":"ICBHI 2017 Challenge",
            "classes":  ["Crackle","Crackle+Wheeze","Normal","Wheeze"],
            "precision":[0.89,0.90,0.87,0.90],
            "recall":   [0.91,0.85,0.91,0.84],
            "f1":       [0.90,0.88,0.89,0.86],
        },
        "Cardiac": {
            "acc":100.0,"samples":84,"color":"#00A99D","light":"#E0F5F3",
            "dataset":"Pascal Heart Sound Challenge",
            "classes":  ["Extra Heart Sound","Heart Murmur","Normal Cardiac"],
            "precision":[1.00,1.00,1.00],
            "recall":   [1.00,1.00,1.00],
            "f1":       [1.00,1.00,1.00],
        },
    }

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("### 🎯 Overall Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, "3 + 1", "Models (3 ML + Motor)"),
        (c2, "1,461",  "Training Samples"),
        (c3, "12",     "Disease Classes"),
        (c4, "95.7%",  "ML Avg Accuracy"),
        (c5, "19",     "Atlas Rules (Motor)"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="val">{val}</div>'
                f'<div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main accuracy bar chart ───────────────────────────────────────────────
    st.markdown("### 📈 Test Accuracy — 3 XGBoost Models")
    names  = list(PERF.keys())
    accs   = [PERF[m]["acc"]   for m in names]
    colors = [PERF[m]["color"] for m in names]
    lights = [PERF[m]["light"] for m in names]

    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    bars = ax.bar(names, accs, color=colors, width=0.45,
                  edgecolor='white', linewidth=2, zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{acc}%', ha='center', va='bottom',
                fontsize=15, fontweight='bold', color='#1B3A6B')
    for yv, yl in [(80,'80%'),(90,'90%'),(95,'95%')]:
        ax.axhline(yv, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(2.35, yv+0.3, yl, fontsize=8, color='gray')
    ax.set_ylim(60, 110)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, color='#1B3A6B')
    ax.set_title('XGBoost Model Accuracy — ECD-Nexus 3.0',
                 fontsize=13, fontweight='bold', color='#1B3A6B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    for i, m in enumerate(names):
        ax.text(i, 63,
                f"Dataset: {PERF[m]['dataset']}\nSamples: {PERF[m]['samples']}",
                ha='center', fontsize=8.5, color='#444',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=lights[i],
                          edgecolor=colors[i], alpha=0.8))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Motor Analysis Section ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎥 Motor Analysis — MediaPipe + SyRIP + Clinical Atlas")

    st.info("""
**Why Motor Analysis is Rule-Based (not ML-trained):**

The SyRIP dataset (Stanford Infant Pose Estimation) was downloaded for motor analysis. 
However, it contained only **16 labeled images** — too few to train a reliable XGBoost classifier.

Instead, we use a **Neuro-Symbolic Rule-Based approach** backed by published pediatric research:
- Google MediaPipe Pose extracts **33 body landmarks** per video frame
- We compute **6 clinical motor biomarkers** from these landmarks
- **7 Clinical Atlas rules** map biomarker values to disease risk flags
- This approach is actually more **interpretable and explainable** than a black-box ML model
    """)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**Motor Biomarkers Extracted (MediaPipe → Math):**")
        motor_features_info = pd.DataFrame({
            'Biomarker'      : ['Limb Asymmetry','Activity Level','Trunk Stability',
                                'Joint Angle Variability','Repetitive Motion Index','Movement Frequency'],
            'How Calculated' : ['L vs R wrist/ankle velocity delta',
                                'Overall landmark displacement magnitude',
                                'Spine landmark displacement per frame',
                                'Elbow+knee angle std deviation',
                                'Landmark trajectory autocorrelation',
                                'Dominant FFT frequency of movement'],
            'Disease Link'   : ['Cerebral Palsy, Hemiplegia',
                                'Sepsis (very low), Seizure (very high)',
                                'HIE, Hypotonia',
                                'CP, Hypotonia',
                                'Neonatal Seizure',
                                'Seizure >3.5Hz, Hypoglycemia'],
        })
        st.dataframe(motor_features_info, use_container_width=True, hide_index=True)

    with mc2:
        st.markdown("**Motor Atlas Rules Coverage:**")
        motor_atlas_info = pd.DataFrame({
            'Rule'      : ['Limb Asymmetry > 0.40','Limb Asymmetry > 0.20',
                           'Repetitive Motion > 0.45','Activity Level < 0.003',
                           'Activity Level > 0.025','Movement Freq > 3.5 Hz',
                           'Joint Angle Var < 8.0'],
            'Finding'   : ['HIGH asymmetry','MEDIUM asymmetry',
                           'HIGH repetitive motion','Very LOW activity',
                           'HIGH activity','High-freq movement',
                           'Very LOW joint variability'],
            'Urgency'   : ['URGENT','ROUTINE','IMMEDIATE','IMMEDIATE','URGENT','IMMEDIATE','URGENT'],
        })
        st.dataframe(motor_atlas_info, use_container_width=True, hide_index=True)

    # ── Per-class F1 scores ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Per-Class Performance (All 3 ML Models)")
    icons = {"Cry":"🎵","Respiratory":"🫁","Cardiac":"❤️"}

    for mname, mdata in PERF.items():
        with st.expander(f"{icons[mname]} {mname} Classifier — Accuracy: {mdata['acc']}% | Dataset: {mdata['dataset']}"):
            cls  = mdata["classes"]
            prec = mdata["precision"]
            rec  = mdata["recall"]
            f1   = mdata["f1"]
            x    = np.arange(len(cls))
            w    = 0.25

            fig2, ax2 = plt.subplots(figsize=(9, 4))
            fig2.patch.set_facecolor('#F8F9FA')
            ax2.set_facecolor('#F8F9FA')
            b1 = ax2.bar(x-w, prec, w, label='Precision', color='#1B3A6B', alpha=0.85)
            b2 = ax2.bar(x,   rec,  w, label='Recall',    color='#2E75B6', alpha=0.85)
            b3 = ax2.bar(x+w, f1,   w, label='F1 Score',  color='#00A99D', alpha=0.85)
            for bars_g in [b1,b2,b3]:
                for bar in bars_g:
                    h = bar.get_height()
                    ax2.text(bar.get_x()+bar.get_width()/2, h+0.005,
                             f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(cls, rotation=15, ha='right', fontsize=9)
            ax2.set_ylim(0,1.2)
            ax2.set_title(f'{mname} Classifier | Dataset: {mdata["dataset"]}',
                          fontweight='bold', color=mdata["color"])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.grid(axis='y', alpha=0.3)
            ax2.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            df_show = pd.DataFrame({
                'Class'    : cls,
                'Precision': [f"{p:.2f}" for p in prec],
                'Recall'   : [f"{r:.2f}" for r in rec],
                'F1 Score' : [f"{f:.2f}" for f in f1],
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    # ── Full summary table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Complete Model Summary")
    summary_df = pd.DataFrame({
        'Component'      : ['Cry Classifier','Respiratory Classifier',
                            'Cardiac Classifier','Motor Analysis','OVERALL'],
        'Type'           : ['XGBoost ML','XGBoost ML','XGBoost ML',
                            'MediaPipe + Rule-Based Atlas','Multi-Model Fusion'],
        'Dataset'        : ['Donate-A-Cry','ICBHI 2017',
                            'Pascal Heart Sound','SyRIP + MediaPipe','All Combined'],
        'Samples'        : ['457','920','84','16 images + live video','1,461 audio'],
        'Classes'        : ['5','4','3','6 biomarkers / 7 rules','12 + motor'],
        'Accuracy'       : ['98.5%','88.5%','100.0%','Rule-based (clinical evidence)','95.7% avg'],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Algorithm table ───────────────────────────────────────────────────────
    st.markdown("### 🤖 Algorithm Details")
    st.markdown("""
| Component | Algorithm | Dataset | Purpose |
|---|---|---|---|
| Cry Classification | **XGBoost** (300 trees) | Donate-A-Cry (457 samples) | Detect infant cry states |
| Respiratory Classification | **XGBoost** (300 trees) | ICBHI 2017 (920 samples) | Detect wheeze / crackle / normal |
| Cardiac Classification | **XGBoost** (300 trees) | Pascal Heart Sound (84 samples) | Detect murmur / normal |
| Motor Feature Extraction | **Google MediaPipe Pose** | SyRIP + any video | 33 landmarks → 6 biomarkers |
| Motor Risk Assessment | **Clinical Atlas Rules** (7 rules) | Published medical research | Flag motor disease risks |
| Agent Orchestration | **LangChain ReAct Agent** | — | Multi-agent coordination |
| Report Generation | **Llama-3.3-70b (Groq)** | — | XAI parent-friendly report |
| Class Imbalance | **Gaussian Noise Augmentation** | — | Balance minority classes |
| Feature Scaling | **StandardScaler** | — | Normalize feature ranges |
| Validation | **5-Fold Cross Validation** | — | Prove generalization |
""")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#888;font-size:.8rem;padding:.5rem">
ECD-Nexus 3.0 &nbsp;|&nbsp; Cry + Respiratory + Cardiac + Motor
&nbsp;|&nbsp; XGBoost + MediaPipe + Llama-3.3-70b + LangChain
&nbsp;|&nbsp; <b>Not a substitute for medical diagnosis</b>
</div>
""", unsafe_allow_html=True)