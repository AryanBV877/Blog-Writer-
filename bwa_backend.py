from __future__ import annotations

import json
import operator
import os
import re
import threading
import time
import urllib.request
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

try:
    from langgraph.types import Send
except ImportError:
    from langgraph.constants import Send  # type: ignore

load_dotenv()

# ============================================================
# BLOG WRITER (Segmented Multi-Agent Workflow + Tavily Web Search & Images)
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] = Field(default_factory=list)
    target_words: int = 180
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    tags: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ] = "explainer"
    tasks: List[Task] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class ImageSpec(BaseModel):
    id: int
    filename: str
    alt: str
    caption: str
    prompt: str
    placement_heading: str = ""
    diagram_nodes: List[str] = Field(default_factory=list)
    source_url: Optional[str] = None


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    web_images: List[dict]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str


# -----------------------------
# 2) GROQ LLM + Rate-Limit Safe Invoke
# -----------------------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=900,
)

_LLM_LOCK = threading.Lock()


def _invoke_llm(messages: list, max_retries: int = 5) -> str:
    """
    Serialize parallel worker calls and automatically wait/retry on Groq 429 TPM rate limits.
    """
    for attempt in range(max_retries):
        try:
            with _LLM_LOCK:
                res = llm.invoke(messages)
                time.sleep(0.8)
            return (res.content or "").strip()
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = (
                "429" in msg
                or "rate_limit" in msg.lower()
                or "RateLimitError" in type(exc).__name__
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            match = re.search(
                r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE
            )
            wait_s = (
                float(match.group(1)) + 1.0
                if match
                else min(20.0, 3.0 * (attempt + 1))
            )
            time.sleep(wait_s)
    return ""


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog-writing agent.
Decide whether web research is needed and generate concise search queries.

Rules:
- Set "needs_research": true and "mode": "open_book" or "hybrid" for topics involving current trends, recent developments, benchmarks, tools, libraries, products, or real-world facts.
- Set "needs_research": false and "mode": "closed_book" only for purely timeless conceptual or personal-outline topics.
- Whenever "needs_research" is true, provide 2 to 3 focused web search queries in "queries".

Return ONLY valid JSON matching this schema:
{
  "needs_research": true,
  "mode": "hybrid",
  "queries": ["query 1", "query 2"]
}
"""


def router_node(state: State) -> dict:
    as_of = state.get("as_of") or date.today().isoformat()
    response = _invoke_llm(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
        ]
    )

    cleaned = _strip_json_fences(response)
    try:
        data = json.loads(cleaned)
    except Exception:
        data = {
            "needs_research": True,
            "mode": "hybrid",
            "queries": [state["topic"]],
        }

    needs_research = data.get("needs_research", True)
    if isinstance(needs_research, str):
        needs_research = needs_research.strip().lower() == "true"

    mode = data.get("mode", "hybrid")
    if mode not in ("open_book", "hybrid", "closed_book"):
        mode = "hybrid" if needs_research else "closed_book"

    queries = [
        str(q).strip()
        for q in (data.get("queries") or [])
        if str(q).strip()
    ]
    if needs_research and not queries:
        queries = [state["topic"].strip()]

    if mode == "open_book":
        recency_days = 7
    elif mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": bool(needs_research),
        "mode": mode,
        "queries": queries[:3],
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    return "research" if state.get("needs_research") else "orchestrator"


# -----------------------------
# 4) Tavily Web Search + Image Retrieval
# -----------------------------
def _normalize_tavily_results(raw_results: Any) -> List[dict]:
    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except Exception:
            return []

    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results", [])

    if not isinstance(raw_results, list):
        return []

    out: List[dict] = []
    for r in raw_results:
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        title = (r.get("title") or url or "Web Source").strip()
        snippet = (r.get("content") or r.get("snippet") or "").strip()
        if not url and not snippet:
            continue
        out.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return out


def _normalize_tavily_images(raw_images: Any) -> List[dict]:
    if not isinstance(raw_images, list):
        return []
    out: List[dict] = []
    for item in raw_images:
        if isinstance(item, str) and item.startswith("http"):
            out.append({"url": item, "description": ""})
        elif isinstance(item, dict):
            u = (item.get("url") or "").strip()
            if u.startswith("http"):
                out.append(
                    {
                        "url": u,
                        "description": (item.get("description") or "").strip(),
                    }
                )
    return out


def _tavily_rest_search(
    query: str, max_results: int = 3, include_images: bool = True
) -> Tuple[List[dict], List[dict]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return [], []

    # 1) Direct Tavily REST API (always supports include_images=True)
    try:
        payload = json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_images": include_images,
                "include_image_descriptions": include_images,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (
            _normalize_tavily_results(data),
            _normalize_tavily_images(data.get("images")),
        )
    except Exception:
        pass

    # 2) Fallback: tavily-python SDK
    try:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)
        res = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images,
        )
        return (
            _normalize_tavily_results(res),
            _normalize_tavily_images(res.get("images") if isinstance(res, dict) else []),
        )
    except Exception:
        pass

    # 3) Fallback: langchain_community TavilySearchResults
    try:
        from langchain_community.tools.tavily_search import (  # type: ignore
            TavilySearchResults,
        )

        tool = TavilySearchResults(max_results=max_results)
        res = tool.invoke({"query": query})
        return _normalize_tavily_results(res), []
    except Exception:
        return [], []


def research_node(state: State) -> dict:
    queries = (state.get("queries") or [state["topic"]])[:3]
    raw: List[dict] = []
    images: List[dict] = []
    seen_urls = set()
    seen_imgs = set()

    for q in queries:
        res_items, img_items = _tavily_rest_search(
            q, max_results=3, include_images=True
        )
        for item in res_items:
            url = item.get("url", "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            raw.append(item)
        for img in img_items:
            u = img.get("url", "")
            if u and u not in seen_imgs:
                seen_imgs.add(u)
                images.append(img)

    evidence: List[EvidenceItem] = []
    for r in raw[:5]:
        try:
            evidence.append(
                EvidenceItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("snippet", ""),
                    published_at=r.get("published_at"),
                    source=r.get("source"),
                )
            )
        except Exception:
            continue

    return {"evidence": evidence, "web_images": images[:6]}


# -----------------------------
# 5) Orchestrator (Segmented Plan)
# -----------------------------
ORCHESTRATOR_SYSTEM = """You are a technical blog architect.
Break the blog post into 4 distinct, logical segments (tasks) that will be written separately and merged into one article.

Return ONLY valid JSON matching this schema:
{
  "blog_title": "string",
  "audience": "string",
  "tone": "string",
  "blog_kind": "explainer" | "tutorial" | "news_roundup" | "comparison" | "system_design",
  "tasks": [
    {
      "id": 1,
      "title": "Section heading",
      "goal": "What this segment must accomplish",
      "bullets": ["key point 1", "key point 2"],
      "target_words": 160,
      "requires_research": true,
      "requires_citations": true,
      "requires_code": false,
      "tags": ["tag1"]
    }
  ]
}
"""


def _default_tasks(topic: str, has_evidence: bool) -> List[Task]:
    return [
        Task(
            id=1,
            title="Introduction and Overview",
            goal=f"Introduce {topic} and explain its purpose and significance.",
            bullets=["Background", "Core purpose", "Why it matters"],
            target_words=140,
            requires_research=has_evidence,
            requires_citations=has_evidence,
        ),
        Task(
            id=2,
            title="Core Architecture and Capabilities",
            goal=f"Explain how {topic} works and its key capabilities.",
            bullets=[
                "Key components",
                "Workflow and data integration",
                "Primary capabilities",
            ],
            target_words=180,
            requires_research=has_evidence,
            requires_citations=has_evidence,
        ),
        Task(
            id=3,
            title="Real-World Impact and Limitations",
            goal=f"Examine practical applications, strengths, and challenges of {topic}.",
            bullets=["Industry adoption", "Operational benefits", "Key limitations"],
            target_words=180,
            requires_research=has_evidence,
            requires_citations=has_evidence,
        ),
        Task(
            id=4,
            title="Conclusion and Outlook",
            goal=f"Summarize key takeaways and future direction for {topic}.",
            bullets=["Summary of insights", "Future outlook"],
            target_words=120,
            requires_research=False,
            requires_citations=False,
        ),
    ]


def orchestrator_node(state: State) -> dict:
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    has_evidence = len(evidence) > 0

    evidence_text = "\n".join(
        f"- {e.title} ({e.url}): {(e.snippet or '')[:120]}"
        for e in evidence[:5]
    )

    response = _invoke_llm(
        [
            SystemMessage(content=ORCHESTRATOR_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of date: {state.get('as_of', '')}\n"
                    f"Evidence:\n{evidence_text or 'None'}"
                )
            ),
        ]
    )

    cleaned = _strip_json_fences(response)
    try:
        data = json.loads(cleaned)
    except Exception:
        data = {}

    raw_tasks = data.get("tasks") or []
    tasks: List[Task] = []
    for idx, t in enumerate(raw_tasks[:5], start=1):
        if not isinstance(t, dict):
            continue
        try:
            tasks.append(
                Task(
                    id=int(t.get("id", idx)),
                    title=str(t.get("title") or f"Section {idx}"),
                    goal=str(t.get("goal") or ""),
                    bullets=[str(b) for b in (t.get("bullets") or [])],
                    target_words=int(t.get("target_words") or 160),
                    requires_research=bool(t.get("requires_research", has_evidence)),
                    requires_citations=bool(t.get("requires_citations", has_evidence)),
                    requires_code=bool(t.get("requires_code", False)),
                    tags=[str(x) for x in (t.get("tags") or [])],
                )
            )
        except Exception:
            continue

    if not tasks:
        tasks = _default_tasks(state["topic"], has_evidence)

    blog_kind = data.get("blog_kind", "explainer")
    if blog_kind not in (
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design",
    ):
        blog_kind = "explainer"

    plan = Plan(
        blog_title=str(data.get("blog_title") or state["topic"]),
        audience=str(data.get("audience") or "technical readers"),
        tone=str(data.get("tone") or "clear and informative"),
        blog_kind=blog_kind,
        tasks=tasks,
    )

    return {"plan": plan, "sections": []}


def fanout_workers(state: State) -> List[Send]:
    plan = state.get("plan")
    if not plan or not plan.tasks:
        return []
    return [
        Send(
            "worker",
            {
                "task": t,
                "topic": state["topic"],
                "plan": plan,
                "evidence": state.get("evidence", []),
                "as_of": state.get("as_of", ""),
            },
        )
        for t in plan.tasks
    ]


# -----------------------------
# 6) Segmented Blog Writer (Worker)
# -----------------------------
WORKER_SYSTEM = """You are a technical blog writer.
Write ONLY the assigned section of the blog in clean Markdown.

Rules:
- Start directly with a level-2 heading: ## <Section Title>
- Do NOT repeat the main blog title (# ...).
- Cover the section goal and bullet points concisely within the target word count.
- When evidence is provided and relevant, cite sources inline using Markdown links: [Source Title](url).
"""


def worker_node(payload: dict) -> dict:
    task: Task = payload["task"]
    plan: Plan = payload["plan"]
    topic: str = payload["topic"]
    evidence: List[EvidenceItem] = payload.get("evidence") or []

    evidence_text = "\n".join(
        f"- {e.title} ({e.url}): {(e.snippet or '')[:140]}"
        for e in evidence[:4]
    )
    bullets_text = "\n".join(f"- {b}" for b in task.bullets)

    response = _invoke_llm(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Topic: {topic}\n"
                    f"Audience: {plan.audience} | Tone: {plan.tone}\n"
                    f"Section #{task.id}: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Points:\n{bullets_text}\n"
                    f"Target words: ~{task.target_words}\n"
                    f"Evidence:\n{evidence_text or 'None'}"
                )
            ),
        ]
    )

    if not response.startswith("## "):
        response = f"## {task.title}\n\n{response}"

    return {"sections": [(task.id, response)]}


# -----------------------------
# 7) Reducer (Merge Segments)
# -----------------------------
def merge_content(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    ordered_sections = [
        md.strip()
        for _, md in sorted(state.get("sections", []), key=lambda x: x[0])
        if md and md.strip()
    ]

    parts = [f"# {plan.blog_title}", "\n\n".join(ordered_sections)]

    evidence = state.get("evidence") or []
    if evidence:
        refs = "\n".join(f"- [{e.title}]({e.url})" for e in evidence if e.url)
        if refs:
            parts.append(f"## References\n\n{refs}")

    merged_md = "\n\n".join(p for p in parts if p).strip()
    return {"merged_md": merged_md}


# -----------------------------
# 8) Image Planning & Web + Diagram Generation
# -----------------------------
IMAGE_PLAN_SYSTEM = """You are a visual editor for a technical blog.
Plan 2 visuals for the blog placed under relevant section headings.

Return ONLY valid JSON matching this schema:
{
  "images": [
    {
      "id": 1,
      "filename": "visual_1.png",
      "alt": "Short alt text",
      "caption": "Descriptive figure caption",
      "prompt": "Visual description",
      "placement_heading": "## Exact heading from the markdown",
      "diagram_nodes": ["Step 1", "Step 2", "Step 3", "Step 4"]
    }
  ]
}
"""


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def _insert_placeholder_after_heading(md: str, heading: str, placeholder: str) -> str:
    if not heading:
        return md + f"\n\n{placeholder}"
    lines = md.splitlines()
    target = heading.strip().lower()
    for i, line in enumerate(lines):
        if line.strip().lower() == target or target.lstrip("# ") in line.strip().lower():
            lines.insert(i + 1, f"\n{placeholder}\n")
            return "\n".join(lines)
    return md + f"\n\n{placeholder}"


def decide_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None
    md = state["merged_md"]
    slug = _safe_slug(plan.blog_title)

    headings = [line.strip() for line in md.splitlines() if line.startswith("## ")]

    response = _invoke_llm(
        [
            SystemMessage(content=IMAGE_PLAN_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Headings: {headings}\n"
                    f"Excerpt:\n{md[:900]}"
                )
            ),
        ]
    )

    cleaned = _strip_json_fences(response)
    try:
        data = json.loads(cleaned)
        raw_imgs = data.get("images") or []
    except Exception:
        raw_imgs = []

    specs: List[ImageSpec] = []
    for idx, item in enumerate(raw_imgs[:2], start=1):
        if not isinstance(item, dict):
            continue
        fname = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            str(item.get("filename") or f"{slug}_fig_{idx}.png"),
        )
        if not fname.endswith(".png"):
            fname = os.path.splitext(fname)[0] + ".png"
        nodes = [
            str(n).strip()
            for n in (item.get("diagram_nodes") or [])
            if str(n).strip()
        ][:5]
        specs.append(
            ImageSpec(
                id=idx,
                filename=f"{slug}_{idx}_{fname}"
                if not fname.startswith(slug)
                else fname,
                alt=str(item.get("alt") or f"{plan.blog_title} Visual {idx}"),
                caption=str(item.get("caption") or f"Visual overview of {plan.blog_title}"),
                prompt=str(item.get("prompt") or f"Visual for {plan.blog_title}"),
                placement_heading=str(
                    item.get("placement_heading")
                    or (headings[min(idx - 1, len(headings) - 1)] if headings else "")
                ),
                diagram_nodes=nodes or [t.title for t in plan.tasks[:4]],
            )
        )

    if not specs:
        specs = [
            ImageSpec(
                id=1,
                filename=f"{slug}_visual_1.png",
                alt=f"{plan.blog_title} Overview",
                caption=f"Overview of {plan.blog_title}",
                prompt=f"Visual for {plan.blog_title}",
                placement_heading=headings[0] if headings else "",
                diagram_nodes=[t.title for t in plan.tasks[:4]] or [plan.blog_title],
            )
        ]

    md_with_placeholders = md
    for spec in specs:
        placeholder = f"[[IMAGE_{spec.id}]]"
        md_with_placeholders = _insert_placeholder_after_heading(
            md_with_placeholders, spec.placement_heading, placeholder
        )

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": [s.model_dump() for s in specs],
    }


def _download_web_image(url: str, out_path: Path) -> bool:
    """
    Download a web image URL, verify it is a valid image with PIL, and save as PNG.
    """
    try:
        from PIL import Image

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; BlogWritingAgent/1.0)"
            },
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if ctype and "image" not in ctype and "octet-stream" not in ctype:
                return False
            raw = resp.read(8 * 1024 * 1024)
        if len(raw) < 2048:
            return False
        img = Image.open(BytesIO(raw))
        if img.width < 150 or img.height < 100:
            return False
        img = img.convert("RGB")
        img.save(out_path, format="PNG")
        return True
    except Exception:
        return False


def _render_local_diagram_png(spec: dict, blog_title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nodes = spec.get("diagram_nodes") or [blog_title]
    nodes = [str(n)[:34] for n in nodes[:5]]
    n_count = max(1, len(nodes))

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=180)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    header = FancyBboxPatch(
        (0.3, 3.25),
        9.4,
        0.72,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#0F172A",
        edgecolor="none",
    )
    ax.add_patch(header)
    ax.text(
        5.0,
        3.63,
        spec.get("alt") or blog_title,
        ha="center",
        va="center",
        color="#FFFFFF",
        fontsize=13,
        fontweight="bold",
    )

    margin_x = 0.45
    total_w = 10.0 - 2 * margin_x
    gap = 0.35
    box_w = (total_w - gap * (n_count - 1)) / n_count
    box_h = 1.35
    y0 = 1.25

    palette = ["#EFF6FF", "#ECFEFF", "#F0FDF4", "#FEF3C7", "#F5F3FF"]
    borders = ["#2563EB", "#0891B2", "#16A34A", "#D97706", "#7C3AED"]

    for idx, label in enumerate(nodes):
        x0 = margin_x + idx * (box_w + gap)
        box = FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=palette[idx % len(palette)],
            edgecolor=borders[idx % len(borders)],
            linewidth=1.8,
        )
        ax.add_patch(box)

        wrapped = "\n".join(
            [label[i : i + 16] for i in range(0, len(label), 16)]
        )
        ax.text(
            x0 + box_w / 2,
            y0 + box_h / 2,
            f"{idx + 1}. {wrapped}",
            ha="center",
            va="center",
            color="#0F172A",
            fontsize=9.5,
            fontweight="semibold",
        )

        if idx < n_count - 1:
            arr = FancyArrowPatch(
                (x0 + box_w + 0.04, y0 + box_h / 2),
                (x0 + box_w + gap - 0.04, y0 + box_h / 2),
                arrowstyle="->,head_length=6,head_width=4",
                color="#475569",
                linewidth=1.8,
            )
            ax.add_patch(arr)

    ax.text(
        5.0,
        0.45,
        (spec.get("caption") or blog_title)[:95],
        ha="center",
        va="center",
        color="#475569",
        fontsize=9.5,
        style="italic",
    )

    plt.tight_layout(pad=0.2)
    fig.savefig(
        out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    specs = state.get("image_specs") or []
    web_images = list(state.get("web_images") or [])

    # If closed-book skipped research, still try fetching topic web images via Tavily
    if not web_images and os.getenv("TAVILY_API_KEY", "").strip():
        _, fetched_imgs = _tavily_rest_search(
            state["topic"], max_results=3, include_images=True
        )
        web_images.extend(fetched_imgs)

    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)

    web_idx = 0
    for spec in specs:
        filename = spec["filename"]
        out_path = images_dir / filename
        downloaded = False

        while web_idx < len(web_images):
            candidate = web_images[web_idx]
            web_idx += 1
            url = candidate.get("url", "")
            if url and _download_web_image(url, out_path):
                downloaded = True
                spec["source_url"] = url
                if candidate.get("description"):
                    spec["caption"] = candidate["description"][:140]
                break

        if not downloaded:
            _render_local_diagram_png(spec, plan.blog_title, out_path)

        placeholder = f"[[IMAGE_{spec['id']}]]"
        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        if placeholder in md:
            md = md.replace(placeholder, img_md)
        else:
            md += f"\n\n{img_md}"

    md_filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(md_filename).write_text(md, encoding="utf-8")

    return {"final": md, "image_specs": specs}


# -----------------------------
# 9) Build Graph
# -----------------------------
g = StateGraph(State)

g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("merge_content", merge_content)
g.add_node("decide_images", decide_images)
g.add_node("generate_and_place_images", generate_and_place_images)

g.add_edge(START, "router")
g.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator",
    },
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout_workers, ["worker"])
g.add_edge("worker", "merge_content")
g.add_edge("merge_content", "decide_images")
g.add_edge("decide_images", "generate_and_place_images")
g.add_edge("generate_and_place_images", END)

app = g.compile()