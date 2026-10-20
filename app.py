"""ScrapeResearch.ai — Streamlit frontend for the multi-agent research assistant.

Backend contract (agents.py):
    build_search_agent().invoke({"messages": [{"role": "user", "content": prompt}]})
    build_reader_agent().invoke({"messages": [{"role": "user", "content": prompt}]})
    writer_chain.invoke({"topic": str, "research": str}) -> str
    critic_chain.invoke({"report": str}) -> str

Run: streamlit run app.py
"""

from __future__ import annotations

import os
import re
import traceback
from datetime import datetime
from typing import Any, Callable

import streamlit as st
from dotenv import load_dotenv

# Load environment variables before any project backend modules are imported.
load_dotenv()

st.set_page_config(
    page_title="ScrapeResearch.ai | Multi-Agent Research",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = (
    ("search", "Search Agent", "Find current web sources and initial evidence."),
    ("reader", "Reader Agent", "Open selected webpages and extract useful content."),
    ("writer", "Writer Chain", "Synthesize gathered evidence into a structured report."),
    ("critic", "Critic Chain", "Review the report and give specific feedback."),
)

# The app theme belongs in .streamlit/config.toml. CSS here is intentionally
# limited to the authored page components rather than fragile global overrides.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.035em; }
    .sr-brand { font: 700 1.38rem/1.25 'Space Grotesk', sans-serif; letter-spacing: -0.045em; color: #f0f3f8; }
    .sr-brand span, .sr-accent { color: #ff8a3d; }
    .sr-eyebrow { color: #ff9b50; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .85rem; }
    .sr-hero { padding: 1.1rem 0 1.6rem; }
    .sr-hero-title { font: 700 clamp(2.3rem, 5vw, 4.25rem)/1.06 'Space Grotesk', sans-serif; letter-spacing: -.055em; margin: 0 0 1rem; color: #f0f3f8; }
    .sr-hero-copy { color: #b7c1d0 !important; font-size: 1.02rem; line-height: 1.8; max-width: 780px; margin: 0; }
    .sr-label { color: #b7c1d0; font-size: .73rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; margin: .35rem 0 .8rem; }
    .sr-card { background: #141923; border: 1px solid #2a3342; border-radius: 14px; padding: 1.15rem 1.25rem; }
    .sr-card p { margin: .65rem 0; color: #f0f3f8; }
    .sr-muted { color: #b7c1d0 !important; font-size: .86rem; line-height: 1.65; }
    .sr-agent { background: #141923; border: 1px solid #2a3342; border-radius: 12px; padding: .85rem 1rem; margin: .55rem 0; }
    .sr-agent-title { color: #f0f3f8; font-weight: 700; font-size: .93rem; }
    .sr-agent-copy { color: #b7c1d0 !important; font-size: .81rem; line-height: 1.55; margin-top: .28rem; }
    .sr-result-title { color: #ff9b50; font-size: .75rem; text-transform: uppercase; letter-spacing: .13em; font-weight: 700; border-bottom: 1px solid #2a3342; padding-bottom: .7rem; margin: .8rem 0 1rem; }
    .sr-metric { background: #141923; border: 1px solid #2a3342; border-radius: 12px; padding: .9rem 1rem; min-height: 78px; }
    .sr-metric-label { color: #b7c1d0; font-size: .75rem; }
    .sr-metric-value { color: #f0f3f8; font-size: .98rem; font-weight: 700; margin-top: .45rem; }
    .sr-footer { border-top: 1px solid #2a3342; margin-top: 3rem; padding-top: 1.25rem; text-align: center; color: #aab4c3; font-size: .78rem; line-height: 1.8; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: linear-gradient(135deg, #ff9b50, #ff772e) !important; color: #17110d !important; border: 0 !important; border-radius: 10px !important; font-weight: 700 !important; min-height: 2.85rem; }
    div[data-testid="stTextArea"] textarea { background: #10151e; border-color: #343d4c; border-radius: 10px; }
    div[data-testid="stExpander"] { border-radius: 12px; }
    [data-testid="stStatusWidget"] { border-radius: 12px; overflow: hidden; }
    [data-testid="stStatusWidget"] summary { font-weight: 600; }
    @media (max-width: 800px) { .sr-hero-title { font-size: 2.35rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ utilities ------------------------------------

def content_to_text(content: Any) -> str:
    """Convert LangChain text or content-block output into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                parts.append(block.strip())
            elif isinstance(block, dict):
                text_part = block.get("text")
                if isinstance(text_part, str) and text_part.strip():
                    parts.append(text_part.strip())
        return "\n\n".join(parts).strip()
    return str(content).strip()


def extract_agent_output(response: Any) -> str:
    """Extract the final assistant message from the create_agent graph result."""
    if isinstance(response, str):
        output = response.strip()
        if output:
            return output
        raise ValueError("The agent returned an empty string.")

    if isinstance(response, dict):
        messages = response.get("messages")
        if isinstance(messages, (list, tuple)):
            # create_agent returns user, AI and tool messages. Avoid returning the
            # tool's raw output or an AI message that is only requesting a tool.
            for message in reversed(messages):
                if isinstance(message, dict):
                    role = str(message.get("role", message.get("type", ""))).lower()
                    if role in {"user", "human", "tool", "function", "toolmessage"}:
                        continue
                    message_content = content_to_text(message.get("content"))
                    if message_content and not message.get("tool_calls"):
                        return message_content
                    continue

                msg_type = str(getattr(message, "type", "")).lower()
                if msg_type in {"human", "user", "tool", "function"}:
                    continue
                message_content = content_to_text(getattr(message, "content", None))
                if message_content and not getattr(message, "tool_calls", None):
                    return message_content

        for key in ("output", "result", "answer", "content", "text"):
            if key in response:
                output = content_to_text(response[key])
                if output:
                    return output
        raise ValueError(
            "The agent response did not contain a final assistant message. "
            "Check the model/tool response in the terminal traceback."
        )

    if hasattr(response, "content"):
        output = content_to_text(response.content)
        if output:
            return output

    raise TypeError(f"Unsupported agent response type: {type(response).__name__}")


def normalize_chain_output(response: Any) -> str:
    """Normalize LCEL chain output to text."""
    output = content_to_text(response)
    if output:
        return output
    if isinstance(response, dict):
        for key in ("text", "output", "answer", "content", "result"):
            output = content_to_text(response.get(key))
            if output:
                return output
    raise ValueError("The chain returned no readable text.")


def run_agent(agent: Any, prompt: str) -> str:
    """Invoke LangChain create_agent using its documented messages state."""
    response = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    return extract_agent_output(response)


def safe_filename(topic: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", topic.strip()).strip("_-")[:70]
    return stem or "scraperesearch_report"


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%d %b %Y, %I:%M %p %Z")


def run_stage(
    number: int,
    key: str,
    title: str,
    description: str,
    action: Callable[[], str],
    progress: Any,
    results: dict[str, str],
) -> str:
    """Render status, run one stage, and keep the error attached to its stage."""
    with st.status(f"Stage {number}/4 · {title}", expanded=True) as status:
        st.write(description)
        try:
            output = action()
            if not isinstance(output, str) or not output.strip():
                raise ValueError(f"{title} returned no readable text.")
        except Exception as exc:
            status.update(
                label=f"Stage {number}/4 · {title} failed",
                state="error",
                expanded=True,
            )
            raise RuntimeError(f"{title} failed: {type(exc).__name__}: {exc}") from exc
        status.update(
            label=f"Stage {number}/4 · {title} completed",
            state="complete",
            expanded=False,
        )
    results[key] = output
    progress.progress(number * 25, text=f"Completed {number} of 4 stages")
    return output


def render_agent_card(number: int, title: str, description: str) -> None:
    st.markdown(
        f"""<div class="sr-agent">
        <div class="sr-agent-title">{number:02d} · {title}</div>
        <div class="sr-agent-copy">{description}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def select_example(topic: str) -> None:
    st.session_state.topic_input = topic


# ------------------------------ session state --------------------------------
if "research_result" not in st.session_state:
    st.session_state.research_result = None
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


# -------------------------------- sidebar -------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sr-brand">ScrapeResearch<span>.ai</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("AUTONOMOUS RESEARCH WORKSPACE")
    st.divider()
    st.markdown("### The four-stage workflow")
    for number, (_, title, description) in enumerate(STEPS, start=1):
        render_agent_card(number, title, description)
    st.divider()
    st.markdown("### Built for research")
    st.write("Search online sources, extract readable content, draft a report and evaluate its quality.")
    st.caption("Streamlit · LangChain · Gemini · Tavily · BeautifulSoup")
    if st.button("Clear current report", use_container_width=True):
        st.session_state.research_result = None
        st.rerun()


# ---------------------------------- hero --------------------------------------
st.markdown(
    """<div class="sr-hero">
        <div class="sr-eyebrow">Multi-agent research assistant</div>
        <div class="sr-hero-title">Research deeper.<br>Discover <span class="sr-accent">smarter.</span></div>
        <p class="sr-hero-copy">Give ScrapeResearch.ai a topic. Its specialized agents search for information, read relevant webpages, write a structured report, and ask a critic chain to evaluate the result.</p>
    </div>""",
    unsafe_allow_html=True,
)

input_col, overview_col = st.columns([1.55, 1], gap="large")
with input_col:
    st.markdown('<div class="sr-label">Start a research task</div>', unsafe_allow_html=True)
    with st.form("research_form", clear_on_submit=False):
        topic = st.text_area(
            "Research topic",
            key="topic_input",
            placeholder="Example: What are the latest developments in AI agents, their real-world applications, and their limitations?",
            height=130,
            help="Use a clear question or a specific topic for more focused research.",
        )
        submitted = st.form_submit_button("Start research  →", type="primary", use_container_width=True)
    st.caption("The workflow uses live web search; completion time depends on API response times and webpage availability.")
    st.markdown("**Try an example**")
    sample_cols = st.columns(3)
    sample_topics = (
        "Latest AI agent developments",
        "AI in healthcare",
        "Cybersecurity trends",
    )
    for col, example in zip(sample_cols, sample_topics):
        with col:
            st.button(
                example,
                key=f"example_{safe_filename(example)}",
                use_container_width=True,
                on_click=select_example,
                args=(example,),
            )

with overview_col:
    st.markdown('<div class="sr-label">Your research workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="sr-card">
        <p><strong>01 · Search</strong><br><span class="sr-muted">Find relevant online sources and initial evidence.</span></p><hr>
        <p><strong>02 · Read</strong><br><span class="sr-muted">Extract useful information from selected webpages.</span></p><hr>
        <p><strong>03 · Write</strong><br><span class="sr-muted">Create an organized research report.</span></p><hr>
        <p><strong>04 · Critique</strong><br><span class="sr-muted">Evaluate the report and provide constructive feedback.</span></p>
        </div>""",
        unsafe_allow_html=True,
    )


# ------------------------------ pipeline run ----------------------------------
if submitted:
    topic_value = topic.strip()
    if not topic_value:
        st.warning("Enter a research topic before starting the pipeline.")
    else:
        # Fail early with clear messages rather than a vague Stage 1 failure.
        missing_keys = []
        if not (os.getenv("MISTRAL_API_KEY")):
            missing_keys.append("MISTRAL_API_KEY")
        if not os.getenv("TAVILY_API_KEY"):
            missing_keys.append("TAVILY_API_KEY")

        if missing_keys:
            st.error("Missing required environment variable(s): " + ", ".join(missing_keys))
            st.info("Add the missing keys to your local .env file, then restart Streamlit. Never paste keys into the code or share them in chat.")
        else:
            st.session_state.research_result = None
            results: dict[str, str] = {}
            progress = st.progress(0, text="Preparing the research pipeline…")
            st.subheader("Research in progress")
            st.caption(f"Topic: {topic_value}")

            try:
                # Lazy import keeps the UI renderable even if a backend dependency
                # or model configuration is broken. Import failures show below.
                from agents import (  # noqa: PLC0415
                    build_reader_agent,
                    build_search_agent,
                    writer_chain,
                    critic_chain,
                )

                results["search"] = run_stage(
                    1, "search", "Search Agent",
                    "Searching the live web for relevant information and source URLs.",
                    lambda: run_agent(
                        build_search_agent(),
                        "Use the web_search tool to research this topic: "
                        f"{topic_value}\n\nFind recent, relevant, credible sources. "
                        "Summarize the key findings and include the exact source URLs returned by the tool. "
                        "Do not invent sources or claim certainty when evidence is unclear.",
                    ),
                    progress, results,
                )

                search_context = results["search"][:12000]
                if len(results["search"]) > 12000:
                    search_context += "\n\n[Search output truncated to fit the reader prompt.]"
                results["reader"] = run_stage(
                    2, "reader", "Reader Agent",
                    "Opening selected webpages and extracting details from the source content.",
                    lambda: run_agent(
                        build_reader_agent(),
                        f"Research topic: {topic_value}\n\n"
                        "Use the scrape_url tool to read the most relevant URLs in the search results below. "
                        "Extract important facts, dates, claims, context and source URLs. If a page cannot be "
                        "accessed, say so; never pretend you read it.\n\nSEARCH RESULTS:\n"
                        f"{search_context}",
                    ),
                    progress, results,
                )

                search_for_writer = results["search"][:8000]
                reader_for_writer = results["reader"][:24000]
                combined_research = (
                    "SEARCH AGENT FINDINGS:\n\n" + search_for_writer
                    + "\n\nREADER AGENT EXTRACTED CONTENT:\n\n" + reader_for_writer
                )
                if len(results["search"]) > 8000:
                    combined_research += "\n\n[Search findings were truncated for context length.]"
                if len(results["reader"]) > 24000:
                    combined_research += "\n\n[Reader output was truncated for context length.]"

                results["writer"] = run_stage(
                    3, "writer", "Writer Chain",
                    "Writing a professional report from the search results and extracted evidence.",
                    lambda: normalize_chain_output(
                        writer_chain.invoke({"topic": topic_value, "research": combined_research})
                    ),
                    progress, results,
                )

                results["critic"] = run_stage(
                    4, "critic", "Critic Chain",
                    "Reviewing clarity, coverage, support for claims and report structure.",
                    lambda: normalize_chain_output(critic_chain.invoke({"report": results["writer"]})),
                    progress, results,
                )

                completed_at = timestamp()
                st.session_state.research_result = {
                    "topic": topic_value,
                    "results": dict(results),
                    "completed_at": completed_at,
                    "incomplete": False,
                }
                st.session_state.research_history.insert(0, {"topic": topic_value, "completed_at": completed_at})
                st.session_state.research_history = st.session_state.research_history[:8]
                progress.progress(100, text="All four stages completed.")
                st.success("Research completed. Your report and critic feedback are ready below.")

            except Exception as exc:
                st.session_state.research_result = {
                    "topic": topic_value,
                    "results": dict(results),
                    "completed_at": timestamp(),
                    "incomplete": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                st.error("The workflow failed. The technical details below identify the actual error.")
                with st.expander("Technical error details", expanded=True):
                    st.code(f"{type(exc).__name__}: {exc}", language="text")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc(), language="text")
                st.info("For a Search Agent error, check the Misral API key/model configuration and Tavily API key/quota. The traceback indicates which call actually failed.")


# ------------------------------ results ---------------------------------------
final_result = st.session_state.research_result
if final_result:
    results = final_result.get("results", {})
    report = results.get("writer", "")
    critic_feedback = results.get("critic", "")

    st.divider()
    st.markdown('<div class="sr-eyebrow">Research workspace</div>', unsafe_allow_html=True)
    st.header(final_result.get("topic", "Research report"))
    if final_result.get("incomplete"):
        st.warning("This run is incomplete. Outputs from stages that succeeded are preserved below.")
    else:
        st.caption(f"Completed {final_result.get('completed_at', '')}")

    metric_cols = st.columns(4)
    for col, (key, name, _) in zip(metric_cols, STEPS):
        with col:
            complete = bool(results.get(key))
            st.markdown(
                f"""<div class="sr-metric">
                <div class="sr-metric-label">{name}</div>
                <div class="sr-metric-value">{'✓ Completed' if complete else '— Not completed'}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    if report:
        report_tab, critic_tab, evidence_tab = st.tabs(["📄 Final report", "🧠 Critic feedback", "🔎 Research evidence"])
        with report_tab:
            st.markdown('<div class="sr-result-title">Generated research report</div>', unsafe_allow_html=True)
            st.markdown(report)
            st.download_button(
                "Download report (.md)",
                data=report,
                file_name=f"{safe_filename(final_result.get('topic', 'research'))}_report.md",
                mime="text/markdown",
                type="primary",
            )
        with critic_tab:
            st.markdown('<div class="sr-result-title">Critic evaluation</div>', unsafe_allow_html=True)
            if critic_feedback:
                st.markdown(critic_feedback)
            else:
                st.info("Critic feedback was not produced in this run.")
        with evidence_tab:
            if results.get("search"):
                with st.expander("Search Agent output", expanded=True):
                    st.text(results["search"])
            if results.get("reader"):
                with st.expander("Reader Agent extracted content"):
                    st.text(results["reader"])
    else:
        st.info("No final report was generated. Available successful outputs are shown below.")
        for key, title in (("search", "Search Agent output"), ("reader", "Reader Agent output")):
            if results.get(key):
                with st.expander(title, expanded=(key == "search")):
                    st.text(results[key])

    if final_result.get("incomplete") and final_result.get("error_message"):
        with st.expander("Why did this run stop?", expanded=True):
            st.code(
                f"{final_result.get('error_type', 'Error')}: {final_result['error_message']}",
                language="text",
            )

if st.session_state.research_history:
    with st.expander("Recent completed research sessions"):
        for item in st.session_state.research_history:
            st.markdown(f"**{item['topic']}**")
            st.caption(item["completed_at"])

st.markdown(
    """<div class="sr-footer"><strong>ScrapeResearch.ai</strong><br>
    Multi-agent research · Web search · Content extraction · Report review<br>
    Built with Streamlit, LangChain, Gemini and Tavily</div>""",
    unsafe_allow_html=True,
)
