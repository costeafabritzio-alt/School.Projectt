import streamlit as st
import pandas as pd
from collections import Counter, defaultdict
from io import StringIO

st.set_page_config(page_title="Exam Topic Predictor (Prototype)", layout="centered")
st.title("📘 Exam Topic Predictor — Prototype")

st.write("Choose a subject, level, and a target year. The app uses past papers to predict likely chapters.")

# --- SUBJECT SELECTION (visual breadth; only Maths populated) ---
subject = st.selectbox("Which subject?", ["Maths", "English", "Business", "Biology"])

# --- Level (appearance only; we use AS-level Maths for prediction) ---
level = st.selectbox("Level (appearance only)", ["AS-level", "A-level"])

st.write("Note: Only **Maths (AS-level)** has data in this demo. Other subjects are placeholders.")

# --- Built-in dataset for Maths AS-level using the counts you supplied ---
# Format: Year -> {chapter: count}, plus total questions for that year
MATHS_DATA = {
    2018: {"Chapter4":1, "Chapter5":1, "Chapter6":1, "Chapter7":2, "Chapter8":1, "Chapter9/10":2, "Chapter11":1, "Chapter12":1, "Chapter13":2, "Chapter14":3, "_total":15},
    2019: {"Chapter1":1, "Chapter4":2, "Chapter5":3, "Chapter6":2, "Chapter7":1, "Chapter8":1, "Chapter9/10":2, "Chapter11":1, "Chapter12":1, "Chapter13":2, "Chapter14":1, "_total":15},
    2020: {"Chapter1":2, "Chapter4":1, "Chapter5":1, "Chapter6":1, "Chapter7":1, "Chapter8":1, "Chapter9/10":2, "Chapter11":1, "Chapter13":1, "Chapter14":2, "_total":13},
    2022: {"Chapter4":1, "Chapter6":1, "Chapter7":2, "Chapter8":1, "Chapter9/10":2, "Chapter11":1, "Chapter12":1, "Chapter13":2, "Chapter14":3, "_total":14},
}
# Note: you didn't give 2021 counts above; if you have them, we can add them.

# Optionally allow user to upload their own CSV to override built-in Maths data
st.write("---")
use_uploaded = st.checkbox("Upload my own CSV to override built-in Maths data (optional)")

uploaded_df = None
if use_uploaded:
    uploaded = st.file_uploader("Upload CSV (columns: Year, Topic, Count OPTIONAL). Example rows:\n2018,Chapter4,1\n2018,Chapter5,1\n(If Count omitted assume 1 each occurrence.)", type=["csv"])
    if uploaded:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.success("Uploaded CSV loaded. It will override built-in Maths data for prediction.")
            st.dataframe(uploaded_df.head(50))
        except Exception as e:
            st.error("Error reading CSV: " + str(e))
            uploaded_df = None

# --- If subject is Maths, show the predictor UI ---
if subject == "Maths":
    st.header("Maths → AS-level predictor (demo)")

    target_year = st.number_input("Enter the exam year you want to predict (target year):", min_value=2019, value=2023, step=1)
    st.write("The predictor uses all past years strictly *before* the target year.")

    # Build training dataset from either uploaded CSV or built-in data
    if uploaded_df is not None:
        # Expect columns: Year, Topic, optional Count
        df = uploaded_df.copy()
        if "Count" not in df.columns:
            df["Count"] = 1
        # Convert to nested dict: year->{topic:count}
        data_by_year = {}
        for y in sorted(df["Year"].unique()):
            subset = df[df["Year"] == y]
            d = {}
            total = 0
            for _, r in subset.iterrows():
                topic = str(r["Topic"]).strip()
                cnt = int(r.get("Count", 1))
                d[topic] = d.get(topic, 0) + cnt
                total += cnt
            d["_total"] = total
            data_by_year[y] = d
    else:
        data_by_year = MATHS_DATA.copy()

    # Collect years that are < target_year
    train_years = sorted([y for y in data_by_year.keys() if y < target_year])
    if not train_years:
        st.warning("No training years available before the target year. Add older data or lower the target year.")
    else:
        st.write(f"Using training years: {train_years}")

        # Aggregate counts across training years
        topic_total_counts = Counter()
        year_presence = defaultdict(int)  # how many years a topic appeared in
        total_questions_sum = 0
        # For recent-2-year check, determine the two most recent training years
        recent_two = sorted(train_years)[-2:] if len(train_years) >= 2 else train_years

        # Gather counts
        for y in train_years:
            year_dict = data_by_year[y]
            year_total = year_dict.get("_total", sum(v for k,v in year_dict.items() if k != "_total"))
            total_questions_sum += year_total
            for topic, cnt in year_dict.items():
                if topic == "_total":
                    continue
                topic_total_counts[topic] += cnt
                if cnt > 0:
                    year_presence[topic] += 1

        # Compute occurrences in recent two years for penalty rule
        recent_counts = Counter()
        for y in recent_two:
            yd = data_by_year.get(y, {})
            for t,c in yd.items():
                if t == "_total": continue
                recent_counts[t] += c

        # Base probabilities: occurrences / total_questions_sum
        raw_probs = {}
        for topic, occ in topic_total_counts.items():
            base = occ / total_questions_sum if total_questions_sum > 0 else 0.0
            multiplier = 1.0

            # Core boost: if topic appears in >= half of the training years -> boost
            if year_presence[topic] >= max(1, len(train_years) // 2):
                multiplier *= 1.15  # slight boost for core topics

            # Repetition penalty: if topic repeats a lot in the most recent two years -> dampen
            if recent_counts[topic] >= 5:
                multiplier *= 0.70
            elif recent_counts[topic] >= 3:
                multiplier *= 0.88

            raw_probs[topic] = base * multiplier

        # If some topics never appeared (unlikely here), still handle
        if not raw_probs:
            st.warning("No topic occurrences found in training years.")
        else:
            # Normalize to sum to 1
            total_raw = sum(raw_probs.values())
            norm_probs = {t: (p / total_raw) if total_raw > 0 else 0.0 for t,p in raw_probs.items()}

            # Sort descending
            sorted_preds = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)

            st.subheader(f"Predicted chapter probabilities for {target_year}")
            for topic, p in sorted_preds:
                st.write(f"- **{topic}** — {p*100:.1f}%")

            top_k = 5
            top_list = sorted_preds[:top_k]
            st.write("")
            st.markdown("**Top predictions:**")
            for i, (t,p) in enumerate(top_list, start=1):
                st.markdown(f"{i}. **{t}** — {p*100:.1f}%")

            st.write("---")
            st.write("How this prediction works (brief):")
            st.write("- Counts exact chapter occurrences across past papers (you gave counts).")
            st.write("- Topics that appear in many different years get a small boost (core topics).")
            st.write("- Topics that appear *very often* in the most recent two years receive a slight penalty (to avoid obvious repetition).")
            st.write("- Final probabilities are normalized so their percentages sum to 100%.")

            # Offer a CSV download of the aggregated training counts & probabilities
            agg_rows = []
            for t in topic_total_counts:
                agg_rows.append({
                    "Topic": t,
                    "TotalOccurrences": int(topic_total_counts[t]),
                    "YearsAppeared": int(year_presence[t]),
                    "Recent2YearsOccurrences": int(recent_counts.get(t,0)),
                    "RawScore": raw_probs[t],
                    "NormalizedProbability": norm_probs[t]
                })
            agg_df = pd.DataFrame(agg_rows).sort_values("NormalizedProbability", ascending=False)
            st.subheader("Aggregated data & final probabilities")
            st.dataframe(agg_df)

            csv_bytes = agg_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv_bytes, file_name=f"predictions_{target_year}.csv", mime="text/csv")

# Non-maths subjects: placeholders
else:
    st.info("This subject is a placeholder. The Maths predictor is the working demo.")
    st.write("You can still upload a CSV (Year, Topic, Count optional) to test custom data for other subjects below.")
    use_other = st.checkbox("Upload CSV for custom subject (Year, Topic, Count optional)")
    if use_other:
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            try:
                df2 = pd.read_csv(up)
                st.success("CSV loaded.")
                st.dataframe(df2.head(50))
                st.write("This demo does not run built-in predictions for other subjects, but you can download this CSV for editing and re-uploading under Maths to run predictions.")
            except Exception as e:
                st.error("Could not read CSV: " + str(e))
