Agentic Blog Writing Studio is an end-to-end AI content generation system built on a stateful LangGraph workflow, Groq (openai/gpt-oss-20b), Tavily web search, and a Streamlit dashboard.

Rather than generating an entire article in a single prompt, the system breaks writing into an explicit, inspectable multi-agent pipeline:

Intelligent Routing: Analyzes the topic and target date to decide between live web research (open_book / hybrid) and closed_book generation.

Web Evidence & Image Retrieval: Uses Tavily to gather cited web sources and topic-relevant images, automatically validating and saving visuals locally (with a matplotlib diagram fallback when needed).

Segmented Planning & Drafting: The orchestrator decomposes the blog into structured section tasks, dispatches parallel worker nodes via LangGraph Send, and merges the completed segments into a cohesive Markdown article with references.

Interactive Dashboard: Displays the segment plan, web evidence table, live article preview with embedded visuals, execution logs, and one-click Markdown or ZIP bundle downloads.
