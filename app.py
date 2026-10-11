"""Streamlit demo for QA Multi-hop pipeline."""
import streamlit as st
import requests
import time

st.set_page_config(page_title="QA Multi-hop Demo", page_icon="🔍", layout="wide")

st.title("🔍 QA Multi-hop Demo")
st.caption("REBEL → Neo4j → GATv2 → Qwen2.5-1.5B QLoRA 4-bit")

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000/ask")
HEALTH_URL = st.sidebar.text_input("Health URL", "http://localhost:8000/health")

# Health check
if st.sidebar.button("Check API Health"):
    try:
        resp = requests.get(HEALTH_URL, timeout=5)
        if resp.status_code == 200:
            st.sidebar.success(f"API OK: {resp.json()}")
        else:
            st.sidebar.error(f"API Error: {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

question = st.text_area(
    "Question",
    placeholder="Ex: What county is Keene High School located in?",
    height=100
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Processing..."):
        start = time.time()
        try:
            resp = requests.post(API_URL, json={"question": question}, timeout=60)
            latency = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Latency", f"{latency:.2f}s")
                col2.metric("Format OK", "✅" if data.get("format_ok") else "❌")
                col3.metric("Abstained", "✅" if data.get("abstained") else "❌")
                
                st.subheader("Answer")
                st.write(data.get("answer", "No answer"))
                
                if data.get("sources"):
                    st.subheader("Sources Cited")
                    for i, src in enumerate(data["sources"], 1):
                        st.markdown(f"**[{i}]** ({src['h']}, {src['r']}, {src['t']}) — *label={src['label']}, score={src['score']:.3f}*")
                        if src.get("source_title"):
                            st.caption(f"Source: {src['source_title']}")
                
                if data.get("abstention_reason"):
                    st.warning(f"Abstention: {data['abstention_reason']}")
                    
            else:
                st.error(f"API Error {resp.status_code}: {resp.text}")
                
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.markdown("**Pipeline:** REBEL → Neo4j → GATv2 (R@5=0.729) → Qwen2.5-1.5B QLoRA 4-bit")
st.markdown("**Test Metrics:** EM=0.163, F1=0.264, Recall@5=0.901, Hallucination=41% (emb 0.5)")