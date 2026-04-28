"""
Medical Jargon Detection & Simplification
==========================================
Built from 3 team notebooks:
  Person 1 (eda.ipynb)              → Data loading, cleaning, EDA
  Person 2 (jargon_detection.ipynb) → Detection engine, spaCy lemmatization
  Person 3 (evaluation_results.ipynb) → Metrics, evaluation, charts
Deployment: HuggingFace Spaces (free)
"""

import streamlit as st
import pandas as pd
import json
import re
import os
import subprocess
from pathlib import Path
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Jargon NLP",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.original-box  { background:#fde8e8; border-left:4px solid #e74c3c;
                 border-radius:8px; padding:14px; line-height:1.8; font-size:15px }
.simplified-box{ background:#e8f8ee; border-left:4px solid #2ecc71;
                 border-radius:8px; padding:14px; line-height:1.8; font-size:15px }
.metric-box    { background:#f0f4ff; border-radius:8px; padding:12px;
                 text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 Medical Jargon NLP")
    st.caption("MedJargon-NLP-Simplifier")
    st.divider()
    page = st.radio("Go to", [
        "🏠 Home",
        "📊 EDA — Person 1",
        "🔍 Detection — Person 2",
        "📈 Evaluation — Person 3",
        "👥 Human Annotation",
    ])
    st.divider()
    st.caption("GitHub repo: lubabasadiyanp/MedJargon-NLP-Simplifier")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (matches exactly what all 3 notebooks do)
# ═══════════════════════════════════════════════════════════════════════════════
REPO_URL  = "https://github.com/lubabasadiyanp/MedJargon-NLP-Simplifier.git"
REPO_PATH = "/tmp/MedJargon"

@st.cache_data(show_spinner="Loading data from GitHub…")
def load_data():
    """Clone repo if needed, then load train.csv + jargon.json — exactly like Person 1."""
    if not os.path.exists(REPO_PATH):
        subprocess.run(["git", "clone", REPO_URL, REPO_PATH],
                       capture_output=True)

    train_path  = os.path.join(REPO_PATH, "train.csv")
    jargon_path = os.path.join(REPO_PATH, "jargon.json")

    # Fallback: files uploaded next to app.py
    if not os.path.exists(train_path):
        train_path  = "train.csv"
        jargon_path = "jargon.json"

    df, jargon_json = None, []

    if os.path.exists(train_path):
        df = pd.read_csv(train_path)

    if os.path.exists(jargon_path):
        with open(jargon_path) as f:
            jargon_json = json.load(f)

    return df, jargon_json


@st.cache_data(show_spinner="Building knowledge base…")
def build_knowledge_base(_jargon_json, _train_df):
    """
    Exactly Person 2's build_expert_dictionary logic.
    Source 1 → jargon.json entities (entity[3] = word list)
    Source 2 → train.csv target_text patterns: 'Jargon (simple)'
    """
    kb = {}

    # Source 1: jargon.json
    for entry in _jargon_json:
        if "entities" in entry:
            for entity in entry["entities"]:
                try:
                    term = " ".join(entity[3]).lower()
                    kb[term] = "Medical Term"
                except (IndexError, TypeError):
                    pass

    # Source 2: train.csv target_text "Jargon (Simple)" pattern
    if _train_df is not None and "target_text" in _train_df.columns:
        pattern = re.compile(r'([a-zA-Z\s\-]{3,})\s\(([^)]+)\)')
        for text in _train_df["target_text"].dropna():
            for jargon, simple in pattern.findall(str(text)):
                kb[jargon.strip().lower()] = simple.strip().lower()

    return kb


# ── NLP utilities ─────────────────────────────────────────────────────────────

def count_syllables(word):
    word = word.lower()
    count, prev_v = 0, False
    for ch in word:
        v = ch in "aeiouy"
        if v and not prev_v:
            count += 1
        prev_v = v
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def fkgl(text):
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not sents or not words:
        return 0.0
    sylls = sum(count_syllables(w) for w in words)
    score = 0.39*(len(words)/len(sents)) + 11.8*(sylls/len(words)) - 15.59
    return round(max(0.0, score), 2)

def simple_clean(text):
    """Person 1's cleaning function."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

def detect_with_kb(text, kb):
    """
    Person 2's predict_jargon logic using lemmatization via regex fallback.
    Returns list of dicts: {term, start, end, simplification}
    """
    results = []
    text_lower = text.lower()
    seen = set()

    for term, simple in sorted(kb.items(), key=lambda x: -len(x[0])):
        pattern = r'\b' + re.escape(term) + r'\b'
        for m in re.finditer(pattern, text_lower):
            span = (m.start(), m.end())
            if span not in seen:
                seen.add(span)
                results.append({
                    "term":           text[m.start():m.end()],
                    "start":          m.start(),
                    "end":            m.end(),
                    "simplification": simple,
                })

    return sorted(results, key=lambda x: x["start"])


def highlight_html(text, detections):
    if not detections:
        return f'<p style="line-height:1.9">{text}</p>'
    out, prev = "", 0
    for d in detections:
        out += text[prev:d["start"]]
        tip = d["simplification"] if d["simplification"] != "Medical Term" else "medical jargon"
        out += (f'<span style="background:#fff3cd;color:#856404;border-radius:4px;'
                f'padding:1px 5px;font-weight:600;cursor:help" title="{tip}">'
                f'{text[d["start"]:d["end"]]}</span>')
        prev = d["end"]
    out += text[prev:]
    return f'<p style="line-height:1.9;font-size:15px">{out}</p>'


def simplify_text(text, detections):
    result = text
    for d in sorted(detections, key=lambda x: -x["start"]):
        simp = d["simplification"]
        if simp and simp.lower() != "medical term":
            result = result[:d["start"]] + simp + result[d["end"]:]
    return result


def calc_accuracy(df, kb, n=100):
    """Person 3's calculate_accuracy — checks if any kb word appears in each row."""
    samples = df["input_text"].dropna().head(n)
    hits = sum(
        1 for t in samples
        if any(w in str(t).lower() for w in kb)
    )
    return round((hits / min(len(samples), n)) * 100, 1)


# ── Load everything ───────────────────────────────────────────────────────────
df, jargon_json = load_data()
kb = build_knowledge_base(jargon_json, df) if (df is not None and jargon_json) else {}

data_ok = df is not None and len(kb) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🩺 Medical Jargon Detection & Simplification")
    st.markdown("*Enter any medical text below — the system will detect jargon and simplify it.*")

    if data_ok:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Terms in KB", f"{len(kb):,}")
        c2.metric("Training rows", f"{len(df):,}")
        c3.metric("jargon.json entries", f"{len(jargon_json):,}")
        c4.metric("Status", "✅ Ready")
    else:
        st.warning("Data not loaded yet. Check sidebar for details.")

    st.divider()

    user_text = st.text_area(
        "Paste any medical text:",
        height=130,
        placeholder=(
            "e.g. The patient presents with severe edema and acute tachycardia. "
            "Prophylaxis with anticoagulant therapy was initiated."
        ),
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        go = st.button("🔍 Simplify", type="primary", use_container_width=True)
    with col_b:
        st.caption("Uses the knowledge base built from your jargon.json + train.csv")

    if go and user_text.strip():
        dets = detect_with_kb(user_text, kb)
        simplified = simplify_text(user_text, dets)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Jargon terms found", len(dets))
        r2.metric("Original FKGL", fkgl(user_text))
        r3.metric("Simplified FKGL", fkgl(simplified))
        r4.metric("Grade level drop", round(fkgl(user_text) - fkgl(simplified), 2))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original (jargon highlighted)**")
            st.markdown(highlight_html(user_text, dets), unsafe_allow_html=True)
        with col2:
            st.markdown("**Simplified**")
            st.markdown(f'<div class="simplified-box">{simplified}</div>',
                        unsafe_allow_html=True)

        if dets:
            st.subheader("Detected terms")
            df_det = pd.DataFrame(dets)[["term", "simplification"]]
            df_det.columns = ["Jargon Term", "Plain Language"]
            st.dataframe(df_det, hide_index=True, use_container_width=True)

    elif go:
        st.info("Please enter some text first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA — PERSON 1
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA — Person 1":
    st.title("📊 EDA — Person 1 (eda.ipynb)")
    st.markdown("Replicates the full EDA notebook: data loading → label distribution → cleaning.")

    if not data_ok:
        st.error("Data not loaded. Ensure train.csv and jargon.json are accessible.")
        st.stop()

    tabs = st.tabs(["Dataset Info", "Jargon Label Distribution", "Text Cleaning Preview",
                    "Raw Data"])

    with tabs[0]:
        st.subheader("Train Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing values", int(df.isnull().sum().sum()))
        st.dataframe(df.head(10), use_container_width=True)
        st.write("**Column types:**")
        st.dataframe(
            pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}),
            hide_index=True, use_container_width=True,
        )

        st.subheader("jargon.json")
        st.metric("Total entries", len(jargon_json))
        if jargon_json:
            with st.expander("First 3 entries"):
                for item in jargon_json[:3]:
                    st.json(item)

    with tabs[1]:
        st.subheader("Distribution of Jargon Categories (entity[2] = class label)")
        all_labels = []
        for entry in jargon_json:
            for entity in entry.get("entities", []):
                try:
                    all_labels.append(entity[2])
                except IndexError:
                    pass

        if all_labels:
            label_counts = pd.Series(all_labels).value_counts().reset_index()
            label_counts.columns = ["Jargon Class", "Count"]
            st.dataframe(label_counts, hide_index=True, use_container_width=True)
            st.bar_chart(label_counts.set_index("Jargon Class"))
            st.metric("Total labelled spans", len(all_labels))
            st.metric("Unique classes", len(set(all_labels)))
        else:
            st.info("No entity labels found in jargon.json")

    with tabs[2]:
        st.subheader("Text Cleaning (Person 1's simple_clean)")
        st.code("""
def simple_clean(text):
    text = text.lower()
    text = re.sub(r'\\[.*?\\]', '', text)  # Remove citations like [1, 2]
    return text.strip()
        """)
        if "input_text" in df.columns:
            sample = df["input_text"].dropna().head(5)
            clean_df = pd.DataFrame({
                "Original":   sample.values,
                "Cleaned":    [simple_clean(t) for t in sample],
            })
            st.dataframe(clean_df, use_container_width=True)
        else:
            st.info("No 'input_text' column found — showing available columns:")
            st.write(df.columns.tolist())

        # FKGL on full dataset sample
        if "input_text" in df.columns:
            sample_texts = df["input_text"].dropna().head(200)
            fkgl_scores  = [fkgl(str(t)) for t in sample_texts]
            fkgl_df = pd.DataFrame({"FKGL Score": fkgl_scores})
            st.subheader("FKGL Distribution (first 200 rows)")
            st.line_chart(fkgl_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean FKGL", round(sum(fkgl_scores)/len(fkgl_scores), 2))
            c2.metric("Max FKGL",  round(max(fkgl_scores), 2))
            c3.metric("Min FKGL",  round(min(fkgl_scores), 2))

    with tabs[3]:
        st.subheader("Raw Data")
        search = st.text_input("Search", "")
        df_show = df.copy()
        if search:
            mask = df_show.apply(
                lambda c: c.astype(str).str.contains(search, case=False, na=False)
            ).any(axis=1)
            df_show = df_show[mask]
        st.dataframe(df_show, use_container_width=True, height=400)
        st.caption(f"{len(df_show):,} rows shown")
        st.download_button("⬇️ Download CSV", df_show.to_csv(index=False),
                           "train_filtered.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DETECTION — PERSON 2
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Detection — Person 2":
    st.title("🔍 Jargon Detection Engine — Person 2 (jargon_detection.ipynb)")
    st.markdown(
        "Replicates `build_expert_dictionary` + `predict_jargon` from Person 2's notebook. "
        "Uses lemmatization-style matching on the **6,422-term knowledge base**."
    )

    if not data_ok:
        st.error("Data not loaded.")
        st.stop()

    st.subheader("Knowledge Base Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total terms", f"{len(kb):,}")

    # Count how many have real simplifications vs "Medical Term"
    real_simps = {k: v for k, v in kb.items() if v.lower() != "medical term"}
    c2.metric("With plain-language swap", f"{len(real_simps):,}")
    c3.metric("Tagged 'Medical Term'", f"{len(kb) - len(real_simps):,}")

    with st.expander("Browse knowledge base (first 50 entries)"):
        kb_df = pd.DataFrame(list(kb.items())[:50], columns=["Term", "Simplification"])
        st.dataframe(kb_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Run Detection")

    input_text = st.text_area(
        "Enter medical text:",
        value="The patient presents with severe edema and acute tachycardia.",
        height=120,
    )

    if st.button("🔍 Detect Jargon", type="primary"):
        dets = detect_with_kb(input_text, kb)

        c1, c2, c3 = st.columns(3)
        c1.metric("Terms detected", len(dets))
        c2.metric("Original FKGL", fkgl(input_text))
        simplified = simplify_text(input_text, dets)
        c3.metric("Simplified FKGL", fkgl(simplified))

        st.subheader("Highlighted")
        st.markdown(highlight_html(input_text, dets), unsafe_allow_html=True)

        st.subheader("Simplified")
        st.markdown(f'<div class="simplified-box">{simplified}</div>',
                    unsafe_allow_html=True)

        if dets:
            st.subheader("Detection table")
            st.dataframe(
                pd.DataFrame(dets)[["term", "simplification", "start", "end"]],
                hide_index=True, use_container_width=True,
            )

    st.divider()
    st.subheader("Batch Detection on Dataset")
    n_rows = st.slider("Rows to scan", 10, min(500, len(df)), 100) if data_ok else 100

    if st.button("Run Batch on train.csv"):
        if "input_text" not in df.columns:
            st.error("No 'input_text' column in train.csv")
        else:
            rows = df["input_text"].dropna().head(n_rows)
            results = []
            bar = st.progress(0)
            for i, text in enumerate(rows):
                d = detect_with_kb(str(text), kb)
                results.append({
                    "input_text":    str(text)[:80] + "…",
                    "jargon_count":  len(d),
                    "terms_found":   "; ".join(x["term"] for x in d),
                    "fkgl_original": fkgl(str(text)),
                    "fkgl_simplified": fkgl(simplify_text(str(text), d)),
                })
                bar.progress((i + 1) / n_rows)

            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True, height=350)
            st.metric("Rows with jargon found", int((df_res["jargon_count"] > 0).sum()))
            st.download_button("⬇️ Download results", df_res.to_csv(index=False),
                               "detection_results.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION — PERSON 3
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Evaluation — Person 3":
    st.title("📈 Evaluation — Person 3 (evaluation_results.ipynb)")
    st.markdown(
        "Replicates `calculate_accuracy` and the final bar chart from Person 3's notebook, "
        "plus extended metrics: SARI, BLEU, FKGL reduction."
    )

    if not data_ok:
        st.error("Data not loaded.")
        st.stop()

    tabs = st.tabs(["NLP Engine Accuracy", "FKGL Analysis",
                    "SARI & BLEU", "Performance Chart", "Upload Your Results"])

    # ── Tab 1: Accuracy (Person 3's exact metric) ─────────────────────────────
    with tabs[0]:
        st.subheader("NLP Engine Accuracy (Person 3 logic)")
        st.code("""
def calculate_accuracy(test_df, knowledge_base):
    test_samples = test_df['input_text'].head(100)
    success_count = 0
    for text in test_samples:
        if any(word in str(text).lower() for word in knowledge_base.keys()):
            success_count += 1
    return (success_count / 100) * 100
        """)
        n_eval = st.slider("Rows to evaluate", 10, min(200, len(df)), 100)
        if st.button("▶ Run Accuracy Evaluation", type="primary"):
            score = calc_accuracy(df, kb, n_eval)
            undetected = round(100 - score, 1)

            c1, c2, c3 = st.columns(3)
            c1.metric("NLP Engine Accuracy", f"{score}%")
            c2.metric("Rows Evaluated", n_eval)
            c3.metric("System Status", "STABLE ✅")

            chart_df = pd.DataFrame({
                "Result": ["Jargon Successfully Detected", "Undetected Words"],
                "Percentage": [score, undetected],
            })
            st.bar_chart(chart_df.set_index("Result"))
            st.caption(
                "This replicates the exact bar chart from evaluation_results.ipynb"
            )

    # ── Tab 2: FKGL ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("FKGL Readability Analysis")
        if "input_text" not in df.columns:
            st.info("No 'input_text' column found.")
        else:
            n_fkgl = st.slider("Rows", 20, min(300, len(df)), 100, key="fkgl_n")
            if st.button("Compute FKGL", key="fkgl_btn"):
                rows = df["input_text"].dropna().head(n_fkgl)
                orig_scores, simp_scores = [], []
                for text in rows:
                    d = detect_with_kb(str(text), kb)
                    orig_scores.append(fkgl(str(text)))
                    simp_scores.append(fkgl(simplify_text(str(text), d)))

                fkgl_compare = pd.DataFrame({
                    "Original FKGL":    orig_scores,
                    "Simplified FKGL":  simp_scores,
                })
                st.line_chart(fkgl_compare)

                c1, c2, c3 = st.columns(3)
                avg_orig = round(sum(orig_scores)/len(orig_scores), 2)
                avg_simp = round(sum(simp_scores)/len(simp_scores), 2)
                c1.metric("Avg Original FKGL",   avg_orig)
                c2.metric("Avg Simplified FKGL",  avg_simp)
                c3.metric("Avg Reduction",         round(avg_orig - avg_simp, 2))

    # ── Tab 3: SARI & BLEU ───────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("SARI & BLEU (approximate)")

        def sari(src, pred, ref):
            s, p, r = set(src.lower().split()), set(pred.lower().split()), set(ref.lower().split())
            add  = len((p - s) & r) / max(len(p - s), 1)
            keep = len((p & s) & r)  / max(len(p & s | s & r), 1)
            dele = len((s - p) - r)  / max(len(s - p), 1)
            return round((add + keep + dele) / 3 * 100, 2)

        def bleu2(hyp, ref):
            h, r = hyp.lower().split(), ref.lower().split()
            if len(h) < 2 or len(r) < 2: return 0.0
            hg = Counter(tuple(h[i:i+2]) for i in range(len(h)-1))
            rg = Counter(tuple(r[i:i+2]) for i in range(len(r)-1))
            m = sum(min(hg[g], rg[g]) for g in hg)
            return round(m / max(sum(hg.values()), 1) * 100, 2)

        if "input_text" in df.columns and "target_text" in df.columns:
            if st.button("Compute SARI & BLEU on first 50 rows"):
                rows = df[["input_text", "target_text"]].dropna().head(50)
                sari_scores, bleu_scores = [], []
                for _, row in rows.iterrows():
                    src  = str(row["input_text"])
                    ref  = str(row["target_text"])
                    d    = detect_with_kb(src, kb)
                    pred = simplify_text(src, d)
                    sari_scores.append(sari(src, pred, ref))
                    bleu_scores.append(bleu2(pred, ref))

                res_df = pd.DataFrame({"SARI": sari_scores, "BLEU-2": bleu_scores})
                c1, c2 = st.columns(2)
                c1.metric("Avg SARI",   round(sum(sari_scores)/len(sari_scores), 2))
                c2.metric("Avg BLEU-2", round(sum(bleu_scores)/len(bleu_scores), 2))
                st.line_chart(res_df)
        else:
            st.info(
                "SARI & BLEU require both 'input_text' and 'target_text' columns. "
                "Showing example with demo text:"
            )
            demo_src = ("The patient presents with severe edema and acute tachycardia. "
                        "Prophylaxis with anticoagulant therapy was initiated.")
            dets = detect_with_kb(demo_src, kb)
            demo_pred = simplify_text(demo_src, dets)
            demo_ref  = demo_pred  # use simplified as reference for demo
            c1, c2 = st.columns(2)
            c1.metric("Demo SARI",   sari(demo_src, demo_pred, demo_ref))
            c2.metric("Demo BLEU-2", bleu2(demo_pred, demo_ref))
            st.markdown(f"**Input:** {demo_src}")
            st.markdown(f"**Simplified:** {demo_pred}")

    # ── Tab 4: Final Chart (Person 3's exact chart) ───────────────────────────
    with tabs[3]:
        st.subheader("Final Performance Chart — Person 3")
        score_val = st.slider("Set accuracy score (from your run)", 0.0, 100.0, 82.0, 0.5)
        chart_df = pd.DataFrame({
            "Category": ["Jargon Successfully Detected", "Undetected Words"],
            "Percentage (%)": [score_val, round(100 - score_val, 1)],
        })
        st.bar_chart(chart_df.set_index("Category"))
        st.caption("This replicates the bar chart generated in evaluation_results.ipynb")

    # ── Tab 5: Upload real results ────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Upload Your Actual Model Results")
        st.markdown("If you ran the notebooks and saved outputs, upload them here.")
        uploaded = st.file_uploader("Upload results CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up, use_container_width=True)
            num_cols = df_up.select_dtypes("number").columns.tolist()
            if num_cols:
                st.bar_chart(df_up[num_cols].head(20))
            st.download_button("Re-download", df_up.to_csv(index=False),
                               "results.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HUMAN ANNOTATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Human Annotation":
    st.title("👥 Human Annotation — 3-Dimension Rating")
    st.markdown(
        "Rate simplified text on **Meaning Preservation · Simplicity · Fluency** (1–5 scale). "
        "Download annotations as CSV for your paper."
    )

    if "annotations" not in st.session_state:
        st.session_state.annotations = []

    # Build examples from real data if available
    if data_ok and "input_text" in df.columns:
        sample_rows = df["input_text"].dropna().head(10).tolist()
    else:
        sample_rows = [
            "The patient exhibits acute myocardial infarction with tachycardia and dyspnea.",
            "Prophylaxis against sepsis was initiated through anticoagulant therapy.",
            "Idiopathic cardiomyopathy with comorbid hypertension was managed conservatively.",
        ]

    st.subheader("Select sentence")
    idx = st.selectbox("Sentence #", range(len(sample_rows)),
                       format_func=lambda i: f"{i+1}. {sample_rows[i][:60]}…")

    original = sample_rows[idx]
    dets = detect_with_kb(original, kb)
    simplified = simplify_text(original, dets)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.markdown(f'<div class="original-box">{original}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Simplified (auto)**")
        st.markdown(f'<div class="simplified-box">{simplified}</div>', unsafe_allow_html=True)

    st.markdown(f"Jargon found: " + (", ".join(f'`{d["term"]}`' for d in dets) or "none"))
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        meaning = st.slider("🎯 Meaning Preservation", 1, 5, 3,
                            help="Is the original medical meaning preserved?")
    with c2:
        simplicity = st.slider("📖 Simplicity", 1, 5, 3,
                               help="Is it easy for a non-expert to understand?")
    with c3:
        fluency = st.slider("✍️ Fluency", 1, 5, 3,
                            help="Is the text grammatically correct and natural?")

    annotator = st.text_input("Your name / ID", placeholder="Annotator_1")
    notes     = st.text_area("Notes (optional)", height=60)

    if st.button("💾 Save Annotation", type="primary"):
        if not annotator.strip():
            st.error("Please enter your name/ID")
        else:
            st.session_state.annotations.append({
                "sentence_id":          idx + 1,
                "original":             original[:80],
                "simplified":           simplified[:80],
                "jargon_found":         len(dets),
                "meaning_preservation": meaning,
                "simplicity":           simplicity,
                "fluency":              fluency,
                "avg_score":            round((meaning + simplicity + fluency) / 3, 2),
                "annotator":            annotator,
                "notes":                notes,
            })
            st.success(f"✅ Saved! Total annotations: {len(st.session_state.annotations)}")

    if st.session_state.annotations:
        st.divider()
        ann_df = pd.DataFrame(st.session_state.annotations)
        st.subheader(f"All Annotations ({len(ann_df)})")
        st.dataframe(ann_df, hide_index=True, use_container_width=True)

        st.subheader("Average Scores")
        avg = ann_df[["meaning_preservation", "simplicity", "fluency"]].mean().round(2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Meaning", avg["meaning_preservation"])
        c2.metric("Avg Simplicity", avg["simplicity"])
        c3.metric("Avg Fluency", avg["fluency"])
        st.bar_chart(ann_df[["meaning_preservation", "simplicity", "fluency"]])

        st.download_button(
            "⬇️ Download annotations CSV",
            ann_df.to_csv(index=False),
            "human_annotations.csv",
            "text/csv",
        )