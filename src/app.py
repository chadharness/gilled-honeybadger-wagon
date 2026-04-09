"""Streamlit demo — interactive query interface for the Deposit Insight Agents."""

import os
import sys

import streamlit as st

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.manager_agent import build_graph
from src.utils.sanitize_output import sanitize
from src.utils.tracing import TraceContext

DEFAULT_DATE = os.environ.get("SIM_TODAY", "2024-10-01")


@st.cache_resource
def init_graph():
    """Build and cache the compiled LangGraph."""
    return build_graph()


st.set_page_config(page_title="Deposit Insight Agents", layout="wide")
st.title("Deposit Insight Agents")
st.caption("Ask questions about deposit activity, transaction flows, and banking analytics.")

# Sidebar config
with st.sidebar:
    st.header("Configuration")
    effective_date = st.text_input("Effective Date", value=DEFAULT_DATE)
    st.divider()
    st.markdown(f"**Graph compiled:** Yes")

# Query input
query = st.chat_input("Enter your query...")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.text(msg["content"])
        else:
            st.write(msg["content"])
        if msg.get("trace_summary"):
            with st.expander("Trace Summary"):
                for span_info in msg["trace_summary"]:
                    st.text(span_info)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Running pipeline..."):
            graph = init_graph()
            trace = TraceContext()
            result = graph.invoke({
                "query": query,
                "effective_date": effective_date,
                "trace": trace,
            })

            raw_response = result.get("response", "No response generated.")
            response = sanitize(raw_response)

        st.text(response)

        # Build trace summary
        trace_summary = []
        for span in trace.spans:
            line = f"{span.component}"
            if span.model_id:
                line += f"  model={span.model_id}"
            if span.latency_ms > 0:
                line += f"  {span.latency_ms:.0f}ms"
            if span.metadata:
                for k, v in span.metadata.items():
                    line += f"  {k}={v}"
            trace_summary.append(line)

        with st.expander("Trace Summary"):
            for span_info in trace_summary:
                st.text(span_info)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "trace_summary": trace_summary,
    })
