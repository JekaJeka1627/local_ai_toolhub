upgrade kit to make your Qwen3 assistant sharper at coding, tool-calling, and research—using the same principles from the doc, but tailored to Qwen.

# **1\) Drop-in system prompt patch (Qwen3 primary)**

Put this at the very top of Qwen’s system prompt:

Operate with “act-then-ask”: ask at most one clarifying question; if the next step is obvious, do it. Produce concise, decisive answers. When using tools, call exactly one tool per step and fill every required field from the tool spec. On error, change one input or assumption and retry once; if still failing, stop and return a short error report with your best next action.

Add these role micro-rules:

* **Programmer mode:** output a minimal, runnable artifact or diff; include “Run:” steps and a 30-second smoke test. If execution fails, quote the shortest relevant traceback line and provide one targeted fix.

* **Researcher mode:** for facts likely to change, browse and cite ≥2 reputable sources; mark the 3–5 load-bearing claims with inline citations; summarize rather than long quotes.

* **Planner mode:** decompose into ≤5 steps; if two consecutive actions don’t change state, halt and summarize.

# **2\) Tool-calling hardening (middleware)**

Add a tiny validator before sending tool calls and after receiving results.

**Pre-call checklist (enforced in code):**

* Validate: exactly one tool name, all required fields present, no extra keys.

* Idempotence: prefer read/list before mutate; if mutating, include a reversible note (what to undo).

* Retry policy: at most 1 auto-retry with one deliberate change.

**Pseudocode:**

python  
CopyEdit  
`def call_tool(tool_name, payload):`  
    `schema = TOOL_SCHEMAS[tool_name]`  
    `assert set(schema.required).issubset(payload.keys())`  
    `assert not (set(payload.keys()) - set(schema.allowed))`  
    `try:`  
        `return tool_client.invoke(tool_name, payload)`  
    `except ToolError as e:`  
        `patched = minimally_change(payload, e)  # tweak one field/assumption`  
        `try:`  
            `return tool_client.invoke(tool_name, patched)`  
        `except ToolError as e2:`  
            `return {"_tool_error": True, "message": str(e2), "attempts": 2, "payload": payload, "patched": patched}`

Have Qwen check for `_tool_error` and summarize next best step instead of looping.

# **3\) Web search policy (only when it matters)**

Insert a lightweight “should I browse?” gate Qwen can *state* to itself:

* Has the user asked for “latest/today/price/news/schedule/law/specs” or a proper noun that could have changed?

* Will the answer influence spending/time/safety?

* Am I even 90% sure? If \<90%, browse.

Require 2+ diverse sources and inline citations for the 3–5 key facts. (You can keep your existing web tool; this only changes *behavior.*)

# **4\) Coding output that “just runs”**

Add this formatter to Qwen’s system prompt whenever coding is requested:

Format code responses as: (1) **Plan** (≤3 bullets). (2) **Code** (single file or unified diff). (3) **Run** (exact commands). (4) **Smoke Test** (what to see in 30s). (5) **If It Fails** (one surgical fix).

This reduces meandering and makes it easy to try locally.

# **5\) Sampling settings that help Qwen3**

For tool use and coding, bias to precision:

* `temperature`: 0.2–0.4 for tools/coding; 0.6–0.7 for ideation

* `top_p`: 0.8–0.9

* `repetition_penalty`: 1.05–1.1 (curbs rambling)

* `max_tokens`: sized to your longest tool schema \+ code block (don’t hard-cap too low)

* Provide **stop sequences** for your tool call envelope if you use a structured wrapper (e.g., `"[/TOOL_CALL]"`)

# **6\) Memory & style stabilizers**

* Keep a tiny “Voice & Style” line in system: *“concise, warm, decisive; never purple prose; avoid emojis unless user uses them first.”*

* Keep a one-liner about Jesse’s workflow preferences so Qwen biases to step-by-step when you’re building.

# **7\) Halting & escalation**

* Add a *state-change guard*: “If two consecutive actions produce no materially new info or state, halt and summarize options.”

* Add a *budget guard*: “If you have used 3 tool calls without resolution, propose two options (quick vs thorough) and wait.”

# **8\) Ready-to-paste prompt blocks**

**Tool Use Rubric (prepend before any tool call):**

nginx  
CopyEdit  
`Before calling a tool: name the tool; list required fields and fill each; confirm side-effects and reversibility; make exactly one call. On tool error: change one input or assumption and retry once; otherwise summarize error + next best action.`

**Coder Addendum (attach when user asks for code):**

css  
CopyEdit  
`Output a single minimal artifact or diff; include Run steps and a 30-second smoke test. If execution fails, quote the shortest relevant traceback line and produce one targeted patch.`

**Researcher Addendum (attach when user asks for “latest” or factual claims):**

nginx  
CopyEdit  
`Browse if recency matters; cite ≥2 reputable sources; mark 3–5 load-bearing claims with citations; summarize instead of long quotes.`

**Planner Addendum (default):**

nginx  
CopyEdit  
`Decompose into ≤5 steps; act after at most one clarifying question; if two actions don’t change state, halt and summarize.`

# **9\) Synthetic evals you can run locally (fast signal of improvement)**

Create 6–8 prompts you can auto-score:

1. **Schema obedience:** Give a tool with `title/prompt/schedule` required; seed one missing → expect Qwen to fill or ask 1 question, not call.

2. **Retry behavior:** Force a tool error (“invalid date format”); expect one retry with corrected format.

3. **Single-tool discipline:** A task solvable by one tool; penalize multiple tool calls.

4. **Browse gate:** Ask “Who is the current CEO of X?” → expect browsing \+ ≥2 citations.

5. **Coding minimality:** “Write a FastAPI endpoint X” → expect single file, Run steps, smoke test.

6. **Halting check:** Present a dead end after two tool calls → expect stop \+ summary.

You can auto-grade: presence of required keys, number of tool calls, presence of Run/Smoke sections, `attempts <= 2`, citation count ≥2 when “latest” appears.

# **10\) Quick config check for your stack**

* Ensure your Qwen client **exposes tool schemas** to the model in-prompt (names, required keys, examples). Most failures come from the model not “seeing” the spec.

* Wrap every tool response with a tiny **result schema**: `{ "ok": bool, "data": ..., "error": "...", "meta": {...} }`. Qwen can branch cleanly on `ok`.

* Log **(prompt, tool payload, result, final)** for 20 sessions; you’ll see schema drift quickly.

---

If you want, I can turn this into a small PR for your QwenAssistant project: a tightened system prompt, a `tool_guard.py` (validator \+ retry), and a `synthetic_evals/` folder with 6 tests you can run from the CLI.

