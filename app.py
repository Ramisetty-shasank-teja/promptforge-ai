"""Streamlit dashboard.  Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

from promptforge.guardrails import Guardrails
from promptforge.llm import get_client
from promptforge.pipeline import run_pipeline
from promptforge.storage import Store

st.set_page_config(page_title="PromptForge AI", page_icon="🔥", layout="wide")
store = Store()
client = get_client()

st.title("🔥 PromptForge AI")
st.caption(f"Create, test, evaluate and optimize prompts systematically. Model: **{client.name}**")

tab_lab, tab_hist = st.tabs(["Prompt Lab", "History & Versions"])

with tab_lab:
    left, right = st.columns([2, 1])
    with left:
        prompt = st.text_area("Your prompt", "Write a professional email requesting leave for two days.", height=120)
        context = st.text_area("Source context for hallucination check (optional)", height=100)
    with right:
        samples = st.slider("Samples per prompt (consistency)", 1, 5, 2)
        forbidden = st.text_input("Forbidden phrases (comma separated)")
        name = st.text_input("Save as", "my-prompt")
        judge = st.checkbox("Use LLM-as-judge (real model only)")
    if st.button("Optimize & Evaluate", type="primary"):
        g = Guardrails(forbidden_phrases=[x.strip() for x in forbidden.split(",") if x.strip()])
        res = run_pipeline(prompt, context, samples, client, g, store, name, use_judge=judge)
        if res["blocked"]:
            st.error(res["reason"]); st.json(res["violations"])
        else:
            a = res["analysis"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Clarity of original", a["clarity_score"])
            c2.metric("Best variant", res["best"])
            c3.metric("Improvement", f"+{res['improvement_points']} pts")
            st.write("**Missing elements:**", ", ".join(a["missing"]) or "none")
            df = pd.DataFrame([{"variant": r["name"], **{k: v for k, v in r["score"].items()
                               if isinstance(v, float)}} for r in res["results"]]).set_index("variant")
            st.bar_chart(df[["overall"]])
            st.dataframe(df)
            for r in res["results"]:
                with st.expander(f"{r['name']} - {r['technique']} - overall {r['score']['overall']}"):
                    st.code(r["prompt"], language="text")
                    st.markdown("**Output**"); st.write(r["output"])
                    if r["constraint_violations"]:
                        st.warning(r["constraint_violations"])
                    if r["score"]["flagged_claims"]:
                        st.warning("Unsupported claims: " + "; ".join(r["score"]["flagged_claims"]))

with tab_hist:
    names = [p["name"] for p in store.list_prompts()]
    if names:
        sel = st.selectbox("Prompt", names)
        hist = store.history(sel)
        st.dataframe(pd.DataFrame([{"version": h["version"], "variant": h["variant"],
                                    "overall": (h["scores"] or {}).get("overall"), "created": h["created"]}
                                   for h in hist]))
        if len(hist) > 1:
            a = st.number_input("From version", 1, len(hist), 1)
            b = st.number_input("To version", 1, len(hist), len(hist))
            st.code(store.diff(sel, int(a), int(b)) or "No differences", language="diff")
    else:
        st.info("No saved prompts yet. Run one in the Prompt Lab.")
