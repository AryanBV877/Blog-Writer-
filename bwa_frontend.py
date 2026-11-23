from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import your compiled LangGraph app
# -----------------------------
from bwa_backend import app


# -----------------------------
# Page Config & Custom Styling
# -----------------------------
st.set_page_config(
    page_title="Agentic Blog Writing Studio",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    width: 100%;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.30);
}

section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #CBD5E1;
    font-weight: 500;
}

/* Hero Banner */
.hero-card {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #1E3A8A 100%);
    border-radius: 16px;
    padding: 1.65rem 2rem;
    margin-bottom: 1.25rem;
    color: #F8FAFC;
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
}

.hero-badge {
    display: inline-block;
    background: rgba(96, 165, 250, 0.18);
    border: 1px solid rgba(147, 197, 253, 0.35);
    color: #93C5FD;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    margin-bottom: 0.65rem;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 0.35rem 0;
    color: #FFFFFF;
}

.hero-sub {
    font-size: 0.94rem;
    color: #CBD5E1;
    margin: 0;
}

/* KPI Metric Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.95rem 1.15rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    margin-bottom: 1rem;
}

.kpi-label {
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.25rem;
}

.kpi-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
}

/* Metadata Pills */
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.5rem 0 1rem 0;
}

.meta-pill {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 999px;
    padding: 0.28rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
}

/* Article Paper Container */
.article-surface {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 2rem 2.25rem;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    margin-bottom: 1rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: #F1F5F9;
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-weight: 500;
    color: #475569;
}

.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 600;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
}

/* Code / Logs */
textarea {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))

        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            for k, v in inner.items():
                if k == "sections" and isinstance(v, list):
                    existing = current_state.get("sections") or []
                    current_state["sections"] = existing + v
                else:
                    current_state[k] = v
        else:
            current_state.update(step_payload)
    return current_state


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Stream graph progress in a single run without re-invoking the graph on runtime errors.
    """
    accumulated: Dict[str, Any] = dict(inputs)
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
            accumulated = extract_latest_state(accumulated, step)
        yield ("final", accumulated)
        return
    except (AttributeError, TypeError):
        pass

    out = graph_app.invoke(inputs)
    yield ("final", out)


# -----------------------------
# Markdown renderer that supports local images
# -----------------------------
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))

        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]

        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)

        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}` (looked for `{img_path}`)")

        i += 1


# -----------------------------
# Past blogs helpers
# -----------------------------
def list_past_blogs() -> List[Path]:
    cwd = Path(".")
    files = [p for p in cwd.glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            return t or fallback
    return fallback


def _get_task_count(plan_obj: Any) -> Optional[int]:
    if not plan_obj:
        return None
    if isinstance(plan_obj, dict):
        return len(plan_obj.get("tasks") or [])
    if hasattr(plan_obj, "tasks"):
        return len(plan_obj.tasks or [])
    return None


def _render_kpi(col, label: str, value: str):
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Hero Header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">Agentic LangGraph Workflow</div>
        <h1 class="hero-title">Blog Writing Studio</h1>
        <p class="hero-sub">
            Automated topic routing, live Tavily web research, segmented multi-worker drafting, and visual diagram generation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.markdown("### ✍️ New Article")
    topic = st.text_area(
        "Topic or Prompt",
        placeholder="e.g., Aladdin Software by BlackRock: Architecture & Risk Analytics",
        height=115,
    )
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary")

    st.divider()
    st.markdown("### 📚 Past Blogs")

    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved blogs found (`*.md` in current folder).")
        selected_md_file = None
    else:
        options: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog to load",
            options=options,
            index=0,
            label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        if st.button("📂 Load Selected Blog", type="secondary"):
            if selected_md_file:
                md_text = read_md_file(selected_md_file)
                st.session_state["last_out"] = {
                    "plan": None,
                    "evidence": [],
                    "image_specs": [],
                    "final": md_text,
                }
                st.session_state["topic_prefill"] = extract_title_from_md(
                    md_text, selected_md_file.stem
                )

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

logs: List[str] = []


def log(msg: str):
    logs.append(msg)


# -----------------------------
# Graph Execution
# -----------------------------
if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    with st.status("Running agentic workflow…", expanded=True) as status:
        progress_area = st.empty()
        current_state: Dict[str, Any] = {}
        last_node = None

        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"➡️ Executing `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)
                summary = {
                    "mode": current_state.get("mode") or "routing…",
                    "needs_research": current_state.get("needs_research"),
                    "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                    "evidence_count": len(current_state.get("evidence", []) or []),
                    "segments_planned": _get_task_count(current_state.get("plan")),
                    "segments_written": len(current_state.get("sections", []) or []),
                    "images_planned": len(current_state.get("image_specs", []) or []),
                }
                progress_area.json(summary)
                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                st.session_state["last_out"] = payload
                status.update(label="✅ Article generation complete", state="complete", expanded=False)
                log("[final] received final state")


# -----------------------------
# Output Dashboard
# -----------------------------
out = st.session_state.get("last_out")
if out:
    mode_val = (out.get("mode") or "loaded").replace("_", " ").title()
    ev_count = len(out.get("evidence") or [])
    seg_count = _get_task_count(out.get("plan")) or len(out.get("sections") or [])
    img_count = len(out.get("image_specs") or [])

    k1, k2, k3, k4 = st.columns(4)
    _render_kpi(k1, "Research Mode", mode_val)
    _render_kpi(k2, "Web Sources", str(ev_count))
    _render_kpi(k3, "Article Segments", str(seg_count))
    _render_kpi(k4, "Generated Visuals", str(img_count))

tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    ["📝 Article Preview", "🧩 Segment Plan", "🔎 Web Evidence", "🖼️ Visuals", "🧾 Execution Logs"]
)

if out:
    # --- Preview tab ---
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No final markdown found.")
        else:
            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            bundle = bundle_zip(final_md, md_filename, Path("images"))

            dl1, dl2, _ = st.columns([1.2, 1.4, 2.4])
            dl1.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode("utf-8"),
                file_name=md_filename,
                mime="text/markdown",
                use_container_width=True,
            )
            dl2.download_button(
                "📦 Download Bundle (MD + Images)",
                data=bundle,
                file_name=f"{safe_slug(blog_title)}_bundle.zip",
                mime="application/zip",
                use_container_width=True,
            )

            st.markdown('<div class="article-surface">', unsafe_allow_html=True)
            render_markdown_with_local_images(final_md)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- Plan tab ---
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan available (loaded from an existing Markdown file).")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"#### {plan_dict.get('blog_title', 'Untitled')}")
            st.markdown(
                f"""
                <div class="pill-row">
                    <span class="meta-pill">Audience: {plan_dict.get('audience', 'General')}</span>
                    <span class="meta-pill">Tone: {plan_dict.get('tone', 'Informative')}</span>
                    <span class="meta-pill">Format: {plan_dict.get('blog_kind', 'explainer')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tasks = plan_dict.get("tasks", [])
            if tasks:
                df = pd.DataFrame(
                    [
                        {
                            "Segment": t.get("id"),
                            "Heading": t.get("title"),
                            "Goal": t.get("goal"),
                            "Target Words": t.get("target_words"),
                            "Citations": "Yes" if t.get("requires_citations") else "No",
                        }
                        for t in tasks
                    ]
                ).sort_values("Segment")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("View raw segment tasks JSON"):
                    st.json(tasks)

    # --- Evidence tab ---
    with tab_evidence:
        queries_used = out.get("queries") or []
        if queries_used:
            st.markdown("**Tavily Queries Executed:** " + " · ".join(f"`{q}`" for q in queries_used))

        evidence = out.get("evidence") or []
        if not evidence:
            st.info("No web evidence returned (closed-book mode or no Tavily results).")
        else:
            rows = []
            for e in evidence:
                if hasattr(e, "model_dump"):
                    e = e.model_dump()
                rows.append(
                    {
                        "Title": e.get("title"),
                        "URL": e.get("url"),
                        "Published": e.get("published_at") or "—",
                        "Snippet": (e.get("snippet") or "")[:180],
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "URL": st.column_config.LinkColumn("Source URL"),
                },
            )

    # --- Images tab ---
    with tab_images:
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No visuals generated for this article.")
        else:
            if images_dir.exists():
                files = sorted([p for p in images_dir.iterdir() if p.is_file()])
                if not files:
                    st.warning("`images/` directory exists but is empty.")
                else:
                    cols = st.columns(min(2, len(files)))
                    for idx, p in enumerate(files):
                        with cols[idx % len(cols)]:
                            st.image(str(p), caption=p.name, use_container_width=True)

                    z = images_zip(images_dir)
                    if z:
                        st.download_button(
                            "⬇️ Download All Visuals (.zip)",
                            data=z,
                            file_name="images.zip",
                            mime="application/zip",
                        )

            if specs:
                with st.expander("View image generation plan"):
                    st.json(specs)

    # --- Logs tab ---
    with tab_logs:
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)

        st.text_area(
            "Workflow Event Log",
            value="\n\n".join(st.session_state["logs"][-80:]),
            height=480,
        )
else:
    with tab_preview:
        st.info("Enter a topic in the left sidebar and click **🚀 Generate Blog** to start.")