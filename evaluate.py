from human_eval.evaluation import evaluate_functional_correctness

if __name__ == "__main__":
    passk = evaluate_functional_correctness(sample_file="results/qwen_results_my.jsonl", n_workers=1, problem_file="humaneval_subset.jsonl")
    print(passk)