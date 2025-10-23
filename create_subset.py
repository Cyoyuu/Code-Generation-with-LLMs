from human_eval.data import read_problems

problems = read_problems()
print(len(problems))  # should show 164

k = 10  # number of tasks you want
subset = list(problems.items())[100:110]
print(subset)

import json

with open("humaneval_subset.jsonl", "w") as f:
    for task in subset:
        f.write(json.dumps(task[1]) + "\n")
