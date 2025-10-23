from openai import OpenAI
import re, json
from human_eval.data import read_problems
import time

client = OpenAI()

def strip_code_block(s: str) -> str:
    # remove ```python or ``` and trailing ```
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:python)?\n", "", s)
        s = re.sub(r"\n```$", "", s)
    return s.strip()

def get_plan_then_code(task_prompt, model="gpt-4o-mini"):
    # 1) Ask for a short plan
    plan_prompt = f"""
output ONLY the final function definition. Label the section "CODE:" and put the code after it.
Do NOT include explanations, tests, or anything else. Your answer must contain only valid Python code (the function body only). Do NOT include explanations or extra text.
- If you output markdown fences (```), remove them before returning the code.

{task_prompt}
"""
    plan_resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a precise coding assistant."},
                  {"role":"user","content":plan_prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    plan_text = plan_resp.choices[0].message.content.strip()

    # 2) Ask for code using that plan; instruct to output only CODE: and nothing else
    code_prompt = f"""
Check for possible bugs or logic errors in the following code and rewrite if needed.
Do NOT include explanations, tests, or anything else. Your answer must contain only valid Python code (the function body only). Do NOT include explanations or extra text.
- If you output markdown fences (```), remove them before returning the code.

Original Code: {plan_text}

{task_prompt}
"""
    code_resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a precise coding assistant."},
                  {"role":"user","content":code_prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    code_text = code_resp.choices[0].message.content
    # extract text after CODE: and sanitize fences
    m = re.search(r"CODE:\s*(.*)", code_text, flags=re.S)
    code_section = m.group(1).strip() if m else code_text
    code_section = strip_code_block(code_section)
    return plan_text, code_section

# Example usage on a small subset
problems = read_problems("humaneval_subset.jsonl")
subset = dict(list(problems.items()))

results = []
for tid, task in subset.items():
    for i in range(10):
        plan, code = get_plan_then_code(task["prompt"])
        results.append({"task_id": tid, "completion": code, "plan": plan})
        print(tid, "plan ->", plan.splitlines())
        print("code preview:", code.splitlines()[0])
        time.sleep(15)
    

# Save for later evaluation (only completion used by human-eval)
with open("results_self_debugging.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps({"task_id": r["task_id"], "completion": r["completion"]}) + "\n")
