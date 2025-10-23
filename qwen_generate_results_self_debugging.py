import re
import json
import time
import os
from human_eval.data import read_problems
import dashscope
from dashscope import Generation

# Set your DashScope API key (ensure DASHSCOPE_API_KEY env var is set)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")

def strip_code_block(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()

def get_plan_then_code(task_prompt, model="qwen-turbo"):  # or "qwen-plus", "qwen-turbo"
    # 1) Ask for a short plan
    plan_prompt = f"""output ONLY the final function definition. Label the section "CODE:" and put the code after it.
Do NOT include explanations, tests, or anything else. Your answer must contain only valid Python code (the function body only). Do NOT include explanations or extra text.
- If you output markdown fences (```), remove them before returning the code.

{task_prompt}"""

    plan_resp = Generation.call(
        model=model,
        prompt=plan_prompt,
        temperature=0.0,
        max_tokens=150,
        result_format="message"
    )

    if plan_resp.status_code != 200:
        raise RuntimeError(f"Plan request failed: {plan_resp}")

    plan_text = plan_resp.output.choices[0].message.content.strip()

    # 2) Ask for code using that plan
    code_prompt = f"""Check for possible bugs or logic errors in the following code and rewrite if needed.
Do NOT include explanations, tests, or anything else. Your answer must contain only valid Python code (the function body only). Do NOT include explanations or extra text.
- If you output markdown fences (```), remove them before returning the code.

Original Code: {plan_text}

{task_prompt}"""

    code_resp = Generation.call(
        model=model,
        prompt=code_prompt,
        temperature=0.0,
        max_tokens=512,
        result_format="message"
    )

    if code_resp.status_code != 200:
        raise RuntimeError(f"Code request failed: {code_resp}")

    code_text = code_resp.output.choices[0].message.content

    # Extract text after "CODE:"
    m = re.search(r"CODE:\s*(.*)", code_text, flags=re.S)
    code_section = m.group(1).strip() if m else code_text
    code_section = strip_code_block(code_section)
    return plan_text, code_section

# Example usage on a small subset
problems = read_problems("humaneval_subset.jsonl")
subset = dict(list(problems.items()))

results = []
for tid, task in subset.items():
    try:
        plan, code = get_plan_then_code(task["prompt"])
        results.append({"task_id": tid, "completion": code, "plan": plan})
        print(tid, "plan ->", plan.splitlines())
        print("code preview:", code.splitlines()[0] if code else "[EMPTY]")
    except Exception as e:
        print(f"Error on {tid}: {e}")
        # Optionally append empty completion on failure
        results.append({"task_id": tid, "completion": "", "plan": ""})
    
    time.sleep(1)  # Adjust or remove based on rate limits (Qwen-Turbo allows faster calls)

# Save for human-eval (only completion needed)
with open("qwen_results_self_debugging.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps({"task_id": r["task_id"], "completion": r["completion"]}) + "\n")