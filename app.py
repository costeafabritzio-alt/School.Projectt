import streamlit as st
import pandas as pd
from collections import Counter
from io import StringIO

st.set_page_config(page_title="Exam Topic Predictor", layout="centered")
st.title("📘 Exam Topic Predictor (Prototype)")

st.write("This app predicts exam topics based on past papers.")

subject = st.selectbox("Choose your subject:", ["Maths"])

if subject == "Maths":
    st.success("You selected Maths. Upload your Maths past paper dataset.")

    uploaded = st.file_uploader("Upload a CSV (columns: Year, Topic). Separate multiple topics with ';'", type=["csv"])
    example = st.checkbox("Load example Maths dataset")

    if example:
        csv = """Year,Topic
2017,Algebra;Geometry
2018,Probability;Algebra
2019,Geometry;Statistics
2020,Algebra;Trigonometry
2021,Probability;Statistics
2022,Geometry;Trigonometry
"""
        df = pd.read_csv(StringIO(csv))
    elif uploaded:
        df = pd.read_csv(uploaded)
    else:
        df = None

    if df is not None:
        st.subheader("Dataset preview")
        st.dataframe(df.head())

        target_year = st.number_input("Enter the exam year you want to predict:", min_value=2018, value=2023, step=1)

        train = df[df["Year"] < target_year]

        if train.empty:
            st.warning("No past papers found before this year.")
        else:
            rows = []
            for _, r in train.iterrows():
                year = int(r['Year'])
                topics = str(r['Topic']).split(';')
                for t in topics:
                    t = t.strip()
                    if t:
                        rows.append({'Year': year, 'Topic': t})
            expanded = pd.DataFrame(rows)

            counts = Counter(expanded['Topic'])
            total_years = train['Year'].nunique()
            preds = [(topic, cnt, cnt/total_years) for topic, cnt in counts.most_common()]

            st.subheader(f"Predicted topics for {target_year}")
            if not preds:
                st.warning("No topics found.")
            else:
                for topic, cnt, score in preds[:5]:
                    st.write(f"- **{topic}** — appeared {cnt} times ({score:.2f} likelihood)")

                if st.button("Show Top 3 Prediction"):
                    top3 = [t for t,_,_ in preds[:3]]
                    st.success("Top 3 predicted topics: " + ", ".join(top3))

