import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

st.set_page_config(page_title="Recommendations from Reviews", page_icon="🛍️", layout="wide")

@st.cache_resource
def load_models():
    from predictor import predict_text, predict_batch
    return predict_text, predict_batch

predict_text, predict_batch = load_models()

st.title("🛍️ Recommendations from Reviews")
st.markdown("**NLP Sentiment and Prediction Analysis Platform**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Single Review", "📋 Bulk Analysis", "📤 CSV Upload"])

with tab1:
    st.subheader("Analyze a Single Review")
    text = st.text_area("Enter an Amazon product review:", height=150)
    model = st.selectbox("Select Model:", ["classical", "lstm"], key="m1")
    if st.button("Analyze Sentiment", key="btn1"):
        if text.strip():
            with st.spinner("Analyzing..."):
                result = predict_text(text, model)
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", "POSITIVE" if result['label'] == 1 else "NEGATIVE")
            col2.metric("Confidence", f"{result['confidence']*100:.1f}%")
            col3.metric("Label", result['label'])
            st.info(f"Cleaned: {result['cleaned_text']}")
        else:
            st.warning("Please enter a review!")

with tab2:
    st.subheader("Analyze Multiple Reviews")
    reviews_input = st.text_area("Enter one review per line:", height=200)
    model2 = st.selectbox("Select Model:", ["classical", "lstm"], key="m2")
    if st.button("Analyze All", key="btn2"):
        texts = [t.strip() for t in reviews_input.split('\n') if t.strip()]
        if texts:
            with st.spinner("Analyzing..."):
                results = predict_batch(texts, model2)
            pos = sum(1 for r in results if r['label'] == 1)
            neg = len(results) - pos
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(results))
            col2.metric("Positive", pos)
            col3.metric("Negative", neg)
            df = pd.DataFrame([{
                "Review": t[:80],
                "Sentiment": "Positive" if r['label'] == 1 else "Negative",
                "Confidence": f"{r['confidence']*100:.1f}%",
                "Label": r['label']
            } for t, r in zip(texts, results)])
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Enter at least one review!")

with tab3:
    st.subheader("Upload CSV File")
    st.markdown("CSV must have a column named review_text")
    uploaded = st.file_uploader("Choose CSV file", type="csv")
    model3 = st.selectbox("Select Model:", ["classical", "lstm"], key="m3")
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        st.write(f"File loaded: {len(df_in)} rows")
        st.dataframe(df_in.head(3), use_container_width=True)
        if st.button("Analyze CSV", key="btn3"):
            col = next((c for c in ["review_text", "text", "review"] if c in df_in.columns), None)
            if col:
                texts = df_in[col].dropna().astype(str).tolist()[:500]
                with st.spinner("Processing..."):
                    results = predict_batch(texts, model3)
                df_out = df_in.head(len(results)).copy()
                df_out['label'] = [r['label'] for r in results]
                df_out['sentiment'] = ["positive" if r['label'] == 1 else "negative" for r in results]
                df_out['confidence'] = [f"{r['confidence']*100:.1f}%" for r in results]
                pos = sum(1 for r in results if r['label'] == 1)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(results))
                col2.metric("Positive", pos)
                col3.metric("Negative", len(results) - pos)
                st.dataframe(df_out, use_container_width=True)
                st.download_button("Download Results", df_out.to_csv(index=False), "results.csv")
            else:
                st.error("No review_text column found!")

st.markdown("---")
st.markdown("Recommendations from Reviews | Built with Flask + PyTorch + Streamlit")





