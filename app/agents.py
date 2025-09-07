# app/agents.py
from __future__ import annotations

from typing import Optional, Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_openai import ChatOpenAI

from .config import OPENAI_MODEL, OPENAI_API_KEY, TOP_K_DEFAULT
from .tools import ToolBinder


SYSTEM_PROMPT = """You are a concise, helpful multimodal research assistant.

PRIORITIES:
1) Prefer the dataset tools first (text_search, image_search, mixed_search, chart_tool).
2) Only call web_search when dataset results are insufficient or the user asks beyond the dataset. If you use web_search,
   explicitly mention that dataset context was weak and you looked online.

IMAGE / CHART BEHAVIOR:
- Call image_search when the user asks for images or supplied an image.
- Call chart_tool only when the user asks to plot/visualize/trend/over-time.When calling chart_tool, PASS a 'chart_spec' JSON like:
{{"dataset":"posts","chart_type":"bar","x":"top_hashtag","y":"count","filters":[],"time_bin":null}}

If you’re unsure, choose a simple bar chart,line chart,time series graph of top hashtags from posts or any other related topic.If you dont have sufficient data ask user to provide it and then create chart.

ANSWER STYLE:
- 4–8 sentences, factual and to the point, with bullets when helpful.
- Cite evidence concisely by listing 2–6 evidence IDs actually used (IDs you saw in tool results).
- If you generated a chart, briefly summarize what it shows.
- When showing images, propose up to three short, caption-like lines for the chosen images.

If the user asks small talk (hi/hello/how are you), respond briefly and mention what you can do."""
# ^ Keep this short and operational. The tools will provide structured context; you keep answers crisp.


ROUTER_HINT = """Intent routing guideline:
- wants_images: if the user asked to show images OR supplied an image.
- wants_chart: if the user asked to plot/chart/visualize/trend/over time.
Routing:
1) If wants_images and not wants_chart: call image_search (optionally text_search for context).
2) If wants_chart: call text_search (optionally image_search), then chart_tool.
3) Else: use mixed_search.
If dataset results are absent/weak for the main intent, call web_search, then answer clearly noting that web info was used."""


def build_prompt() -> ChatPromptTemplate:
    """
    IMPORTANT: For tool-using OpenAI agents, LangChain requires an agent_scratchpad MessagesPlaceholder.
    We also pass chat_history so the agent can be context-aware.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("system", ROUTER_HINT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            # Required by create_openai_tools_agent:
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


def _to_lc_history(chat_history: Optional[List[Dict[str, str]]]) -> List:
    """Convert [{'role': 'user'|'assistant', 'content': '...'}, ...] into LC messages."""
    if not chat_history:
        return []
    out = []
    for m in chat_history:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            out.append(AIMessage(content=content))
    return out


def run_agent(
    retriever,
    user_query: str,
    query_image_path: Optional[str] = None,
    top_k: int = TOP_K_DEFAULT,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Run the LangChain OpenAI Tools Agent with your dataset tools.
    Returns a dict consumable by your Streamlit app:
      {
        "answer": str,
        "evidence_ids": List[str],
        "image_ids": List[str]
      }
    """

    # 1) LLM (ensure API key is provided explicitly)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, api_key=OPENAI_API_KEY)

    # 2) Tools bound to this retriever
    tb = ToolBinder.bind(retriever)  # your helper that returns bound callables
    tools = [tb.text_search, tb.image_search, tb.mixed_search, tb.chart_tool, tb.web_search]

    # 3) Prompt with required placeholders
    prompt = build_prompt()

    # 4) Agent + executor (scratchpad is managed automatically)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=6,
        verbose=False,
        return_intermediate_steps=False,
    )

    # 5) Build inputs: include a hint if an image was uploaded
    image_hint = f"(User uploaded an image at: {query_image_path})" if query_image_path else ""
    lc_history = _to_lc_history(chat_history)

    result = executor.invoke(
        {
            "input": f"{user_query}\n{image_hint}".strip(),
            "chat_history": lc_history,
            # DO NOT pass agent_scratchpad yourself; AgentExecutor manages it.
        }
    )

    # 6) Post-pass for UI: ensure we expose image IDs and evidence IDs for the app to render nicely.
    wants_images = any(
        kw in user_query.lower()
        for kw in ["image", "images", "show me an image", "visual", "thumbnail", "thumbnails", "pictures", "pics"]
    )

    # If the LLM didn't call image_search when user asked for images, we still populate some image hits for the UI.
    image_ids: List[str] = []
    if wants_images:
        image_hits = retriever.search_image_topk(query_text=user_query, query_image_path=query_image_path, k=min(6, top_k))
        image_ids = [h["meta"].id for h in image_hits]

    # A small set of text hits for provenance bullets, in case the answer didn’t list them explicitly.
    text_hits = retriever.search_text_topk(user_query, k=min(6, top_k))
    evidence_ids = [h["meta"].id for h in text_hits] + image_ids
    evidence_ids = list(dict.fromkeys(evidence_ids))[:20]  # de-dupe, cap

    return {
        "answer": result.get("output", "").strip(),
        "evidence_ids": evidence_ids,
        "image_ids": image_ids,
    }
