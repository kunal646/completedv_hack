"""
SpecFlow — Multi-agent LangGraph implementation
  Agent 0 : Classifier      (lane + title + priority)
  Agent 1 : Issue Interpreter
  Agent 2 : Implementation Architect
"""

import json
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# ── Prompts ────────────────────────────────────────────────────────────────────

ISSUE_INTERPRETER_PROMPT = """You are an Issue Interpreter for software development teams. You receive raw, unstructured feedback — messy messages from WhatsApp, Slack, email, or call notes — and turn them into structured, actionable issue cards.

When given raw feedback, you must:

1. CLASSIFY: Is this a Bug, Feature Request, Improvement, or Question?
2. EXTRACT the core requirement in one clear sentence.
3. IDENTIFY:
   - What the user actually wants (not what they literally said)
   - Acceptance criteria (when is this "done"?)
   - Edge cases they didn't mention but should be handled
   - Severity/Priority (Critical / High / Medium / Low)
4. FLAG any ambiguity — things that need clarification before building.

OUTPUT in this exact format:

## Issue Card

**Type:** [Bug / Feature / Improvement / Question]
**Priority:** [Critical / High / Medium / Low]
**Summary:** [One clear sentence]

**What the user said:**
> [Original raw message]

**What they actually need:**
[Your interpretation — clear, specific, technical]

**Acceptance Criteria:**
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

**Edge Cases to Handle:**
- [Edge case 1]
- [Edge case 2]

**Needs Clarification:**
- [Question 1 — if any]

**Estimated Complexity:** [Small / Medium / Large]

Be precise. Never add fluff. If the original message is vague, say exactly what's missing."""


IMPLEMENTATION_ARCHITECT_PROMPT = """You are an Implementation Architect for software development teams. You receive a structured issue card (already reviewed and approved by a human) and convert it into a concrete, build-ready technical implementation plan.

Your plan must be specific enough for a developer to start coding immediately without needing further clarification.

OUTPUT in this exact format:

## Implementation Plan

**Issue:** [One-line summary from the issue card]
**Type:** [Bug / Feature / Improvement]
**Priority:** [Critical / High / Medium / Low]
**Estimated Effort:** [e.g. 2–4 hours / 1–2 days]

---

### 🏗️ Affected Areas
- **[Component / File / Service]** — [why it needs to change]
- **[Component / File / Service]** — [why it needs to change]

---

### 📋 Implementation Steps
1. **[Step title]** — [Specific action: which file, function, or API to modify and how]
2. **[Step title]** — [Specific action]
3. **[Step title]** — [Specific action]

---

### ✅ Testing Checklist
- [ ] **Unit:** [Specific test case]
- [ ] **Integration:** [Specific test case]
- [ ] **E2E / Manual:** [Specific test scenario]

---

### ⚠️ Risks & Mitigations
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| [Risk description] | High / Medium / Low | [How to prevent or handle it] |

---

### 🔗 Dependencies & Notes
- [Any libraries, APIs, team dependencies, or pre-conditions]

Be precise and technical. Assume a mid-level developer is reading this. No fluff."""


# ── Shared state ───────────────────────────────────────────────────────────────

class InterpreterState(TypedDict):
    raw_feedback: str
    issue_card: Optional[str]
    error: Optional[str]
    debug: Optional[str]


class ArchitectState(TypedDict):
    issue_card: str
    tech_plan: Optional[str]
    error: Optional[str]
    debug: Optional[str]


class ClassifierState(TypedDict):
    raw_feedback: str
    lane: Optional[str]       # issue | feature | idea | marketing
    title: Optional[str]
    summary: Optional[str]
    priority: Optional[str]   # critical | high | medium | low
    error: Optional[str]
    debug: Optional[str]


CLASSIFIER_PROMPT = """You are a product intelligence classifier for a software team.
Given a raw message from a team channel (Slack, WhatsApp, Discord), classify it into exactly one lane and extract key metadata.

Lanes:
- issue     : bugs, errors, broken flows, user complaints about things not working
- feature   : requested new capabilities, improvements to existing functionality
- idea      : exploratory thoughts, "what if" scenarios, experiments, early-stage concepts
- marketing : copy ideas, campaign thoughts, positioning signals, competitor mentions, growth ideas

OUTPUT: valid JSON only, no markdown, no explanation.
{
  "lane": "issue" | "feature" | "idea" | "marketing",
  "title": "short descriptive title max 7 words",
  "priority": "critical" | "high" | "medium" | "low",
  "summary": "one sentence summary of the core signal"
}

Priority guide:
- critical : production broken, data loss, security issue, users blocked
- high     : significant user impact, important feature, strong team signal
- medium   : noticeable issue or valuable idea with moderate urgency
- low      : minor polish, casual idea, low frequency

Be decisive. Every message belongs to exactly one lane."""


# ── LLM factory ────────────────────────────────────────────────────────────────

def get_llm(api_key: str):
    """Auto-detect provider from key prefix."""
    if api_key.startswith("sk-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0), "OpenAI / GPT-4o"
    elif api_key.startswith("gsk_"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        ), "Groq / Llama-3.3-70b"
    else:
        raise ValueError(
            f"Unrecognised API key (starts with '{api_key[:6]}...'). "
            "Use an OpenAI key (sk-...) or Groq key (gsk_...)."
        )


# ── Agent 1 — Issue Interpreter ────────────────────────────────────────────────

def make_interpreter_node(api_key: str, context_str: str = ""):
    def interpret(state: InterpreterState) -> InterpreterState:
        debug = []
        try:
            llm, provider = get_llm(api_key)
            debug.append(f"Agent 1 — Issue Interpreter | {provider}")
            debug.append("Calling model...")
            system_prompt = (context_str + "\n\n" + ISSUE_INTERPRETER_PROMPT) if context_str else ISSUE_INTERPRETER_PROMPT
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Raw feedback:\n{state['raw_feedback']}"),
            ])
            debug.append("✅ Issue card generated successfully")
            return {**state, "issue_card": response.content, "error": None, "debug": "\n".join(debug)}
        except Exception as e:
            debug.append(f"❌ Error: {e}")
            return {**state, "issue_card": None, "error": str(e), "debug": "\n".join(debug)}
    return interpret


def build_interpreter_graph(api_key: str, context_str: str = ""):
    builder = StateGraph(InterpreterState)
    builder.add_node("interpret", make_interpreter_node(api_key, context_str))
    builder.add_edge(START, "interpret")
    builder.add_edge("interpret", END)
    return builder.compile()


# ── Agent 2 — Implementation Architect ────────────────────────────────────────

def make_architect_node(api_key: str, context_str: str = ""):
    def architect(state: ArchitectState) -> ArchitectState:
        debug = []
        try:
            llm, provider = get_llm(api_key)
            debug.append(f"Agent 2 — Implementation Architect | {provider}")
            debug.append("Calling model...")
            system_prompt = (context_str + "\n\n" + IMPLEMENTATION_ARCHITECT_PROMPT) if context_str else IMPLEMENTATION_ARCHITECT_PROMPT
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Approved issue card:\n\n{state['issue_card']}"),
            ])
            debug.append("✅ Implementation plan generated successfully")
            return {**state, "tech_plan": response.content, "error": None, "debug": "\n".join(debug)}
        except Exception as e:
            debug.append(f"❌ Error: {e}")
            return {**state, "tech_plan": None, "error": str(e), "debug": "\n".join(debug)}
    return architect


def build_architect_graph(api_key: str, context_str: str = ""):
    builder = StateGraph(ArchitectState)
    builder.add_node("architect", make_architect_node(api_key, context_str))
    builder.add_edge(START, "architect")
    builder.add_edge("architect", END)
    return builder.compile()


# ── Agent 0 — Classifier ───────────────────────────────────────────────────────

def make_classifier_node(api_key: str, context_str: str = ""):
    def classify(state: ClassifierState) -> ClassifierState:
        debug = []
        try:
            llm, provider = get_llm(api_key)
            debug.append(f"Agent 0 — Classifier | {provider}")
            system_prompt = (context_str + "\n\n" + CLASSIFIER_PROMPT) if context_str else CLASSIFIER_PROMPT
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Message to classify:\n{state['raw_feedback']}"),
            ])
            raw = response.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            lane     = data.get("lane", "idea")
            title    = data.get("title", state["raw_feedback"][:60])
            summary  = data.get("summary", "")
            priority = data.get("priority", "medium")
            debug.append(f"✅ Classified as [{lane}] | priority={priority}")
            return {**state, "lane": lane, "title": title, "summary": summary,
                    "priority": priority, "error": None, "debug": "\n".join(debug)}
        except Exception as e:
            debug.append(f"❌ Error: {e}")
            # Fallback defaults
            return {**state, "lane": "idea", "title": state["raw_feedback"][:60],
                    "summary": "", "priority": "medium", "error": str(e),
                    "debug": "\n".join(debug)}
    return classify


def build_classifier_graph(api_key: str, context_str: str = ""):
    builder = StateGraph(ClassifierState)
    builder.add_node("classify", make_classifier_node(api_key, context_str))
    builder.add_edge(START, "classify")
    builder.add_edge("classify", END)
    return builder.compile()


# ── Shared lane-processing state ───────────────────────────────────────────────

class LaneProcessState(TypedDict):
    raw_feedback: str
    lane: str
    output: Optional[str]    # processed markdown output
    error: Optional[str]
    debug: Optional[str]


# ── Agent 3 — Feature Analyst ──────────────────────────────────────────────────

FEATURE_ANALYST_PROMPT = """You are a Feature Analyst. Given a raw feature request, produce a structured feature brief.

OUTPUT in this exact format:

## Feature Brief

**Summary:** [One clear sentence — what the feature is]

**User Story:**
> As a [type of user], I want [feature] so that [benefit].

**Problem It Solves:**
[One sentence — what pain this removes or value it adds]

**Acceptance Criteria:**
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

**Effort Estimate:** [Small < 1 day / Medium 1–3 days / Large 1+ week]

**Business Value:** [High / Medium / Low] — [One-line reason]

**Dependencies:**
- [Any required work, APIs, or team dependencies — or "None"]

Be specific. No fluff."""


# ── Agent 4 — Idea Summarizer ──────────────────────────────────────────────────

IDEA_SUMMARIZER_PROMPT = """You are an Idea Evaluator. Given a raw exploratory idea from a team chat, distil it into a structured idea brief.

OUTPUT in this exact format:

## Idea Brief

**Core Concept:** [One sentence — what the idea actually is]

**Potential Impact:** [High / Medium / Low] — [One-line reason]

**What Would Need to Be True:**
- [Assumption or condition 1]
- [Assumption or condition 2]

**Risks / Challenges:**
- [Risk 1]

**Recommended Next Action:** [Research / Prototype / Team Discussion / Park]

**Related Areas:** [Existing features, products, or initiatives this connects to]

Be concise. Preserve the spirit of the original idea."""


# ── Agent 5 — Marketing Extractor ─────────────────────────────────────────────

MARKETING_EXTRACTOR_PROMPT = """You are a Marketing Signal Extractor. Given a raw message from a team chat containing a marketing-related thought, extract the key signal.

OUTPUT in this exact format:

## Marketing Signal

**Theme:** [Positioning / Copy / Campaign / Competitor / Growth / UX / Other]

**Core Message:** [The key marketing insight in one sentence]

**Target Audience Implied:** [Who this message is about or for]

**Signal Quality:** [Strong — clear and actionable / Weak — vague, needs development]

**Suggested Action:** [Use as-is / Develop further / File for later / Discuss with team]

**Keywords / Tags:** [comma-separated relevant tags]

Be precise. Capture the marketing intent even if the original message was informal."""


def _make_lane_node(api_key: str, prompt: str, agent_label: str, context_str: str = ""):
    def process(state: LaneProcessState) -> LaneProcessState:
        debug = []
        try:
            llm, provider = get_llm(api_key)
            debug.append(f"{agent_label} | {provider}")
            system_prompt = (context_str + "\n\n" + prompt) if context_str else prompt
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Raw message:\n{state['raw_feedback']}"),
            ])
            debug.append("✅ Done")
            return {**state, "output": response.content, "error": None, "debug": "\n".join(debug)}
        except Exception as e:
            debug.append(f"❌ {e}")
            return {**state, "output": None, "error": str(e), "debug": "\n".join(debug)}
    return process


def build_feature_graph(api_key: str, context_str: str = ""):
    b = StateGraph(LaneProcessState)
    b.add_node("process", _make_lane_node(api_key, FEATURE_ANALYST_PROMPT, "Agent 3 — Feature Analyst", context_str))
    b.add_edge(START, "process"); b.add_edge("process", END)
    return b.compile()


def build_idea_graph(api_key: str, context_str: str = ""):
    b = StateGraph(LaneProcessState)
    b.add_node("process", _make_lane_node(api_key, IDEA_SUMMARIZER_PROMPT, "Agent 4 — Idea Summarizer", context_str))
    b.add_edge(START, "process"); b.add_edge("process", END)
    return b.compile()


def build_marketing_graph(api_key: str, context_str: str = ""):
    b = StateGraph(LaneProcessState)
    b.add_node("process", _make_lane_node(api_key, MARKETING_EXTRACTOR_PROMPT, "Agent 5 — Marketing Extractor", context_str))
    b.add_edge(START, "process"); b.add_edge("process", END)
    return b.compile()


# ── Relevance filter ──────────────────────────────────────────────────────────

def check_relevance(raw_feedback: str, api_key: str, context_str: str) -> dict:
    """
    Uses the product context to decide if a message is relevant to the product.
    Returns: {"relevant": bool, "reason": str}
    If no context_str is provided, always returns relevant=True (no filter).
    """
    if not context_str or not context_str.strip():
        return {"relevant": True, "reason": "No product context configured — passing through."}

    llm, _ = get_llm(api_key)

    system_prompt = f"""You are a relevance filter for a product feedback system.

Your job is to decide whether a message is relevant to the product described below.
A message is RELEVANT if it:
- Reports a bug, issue, or problem with the product
- Requests a feature, improvement, or change to the product
- Provides feedback, opinions, or ideas about the product
- Discusses usage of the product or asks a product-related question
- Contains any signal that the product team should know about

A message is IRRELEVANT if it:
- Is casual conversation unrelated to the product (greetings, jokes, off-topic chat)
- Is about a completely different product or system
- Is spam, noise, or has no actionable content
- Is a system/bot message or automated notification with no user feedback

--- PRODUCT CONTEXT ---
{context_str}
-----------------------

Respond ONLY with valid JSON in this exact format:
{{"relevant": true, "reason": "one short sentence explaining why"}}
or
{{"relevant": false, "reason": "one short sentence explaining why"}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Message to evaluate:\n{raw_feedback[:800]}"),
        ])
        text = response.content.strip()
        # Extract JSON from response
        import re as _re
        json_match = _re.search(r'\{.*?\}', text, _re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "relevant": bool(result.get("relevant", True)),
                "reason": result.get("reason", ""),
            }
    except Exception:
        pass
    # Default: pass through if filter fails
    return {"relevant": True, "reason": "Filter check failed — passing through."}


# ── Unified auto-process dispatcher ───────────────────────────────────────────

def auto_process(raw_feedback: str, api_key: str, context_str: str = "") -> dict:
    """
    Full auto-intake pipeline:
      1. Classify → lane, title, priority, summary
      2. Run lane-appropriate agent
      3. For issues: returns issue_card only (human review needed before plan)
      4. For feature/idea/marketing: returns full output directly

    context_str: grounded product/codebase context injected into all agent prompts.
    Returns dict with keys: lane, title, priority, summary, output, needs_review, error
    """
    # Step 1 — Classify
    clf = build_classifier_graph(api_key, context_str)
    cls_result = clf.invoke({
        "raw_feedback": raw_feedback, "lane": None, "title": None,
        "summary": None, "priority": None, "error": None, "debug": None,
    })
    lane     = cls_result["lane"]
    title    = cls_result["title"]
    priority = cls_result["priority"]
    summary  = cls_result["summary"]

    base = {"lane": lane, "title": title, "priority": priority,
            "summary": summary, "output": None, "needs_review": False, "error": None}

    # Step 2 — Lane-specific agent
    try:
        if lane == "issue":
            g = build_interpreter_graph(api_key, context_str)
            r = g.invoke({"raw_feedback": raw_feedback, "issue_card": None, "error": None, "debug": None})
            base["output"] = r.get("issue_card")
            base["needs_review"] = True
            base["error"] = r.get("error")

        elif lane == "feature":
            g = build_feature_graph(api_key, context_str)
            r = g.invoke({"raw_feedback": raw_feedback, "lane": lane, "output": None, "error": None, "debug": None})
            base["output"] = r.get("output"); base["error"] = r.get("error")

        elif lane == "idea":
            g = build_idea_graph(api_key, context_str)
            r = g.invoke({"raw_feedback": raw_feedback, "lane": lane, "output": None, "error": None, "debug": None})
            base["output"] = r.get("output"); base["error"] = r.get("error")

        elif lane == "marketing":
            g = build_marketing_graph(api_key, context_str)
            r = g.invoke({"raw_feedback": raw_feedback, "lane": lane, "output": None, "error": None, "debug": None})
            base["output"] = r.get("output"); base["error"] = r.get("error")

    except Exception as e:
        base["error"] = str(e)

    return base
