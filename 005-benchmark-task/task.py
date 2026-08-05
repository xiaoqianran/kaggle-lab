# %% [markdown]
# kaggle-lab 005 — minimal arithmetic smoke task
# Validate: `python task.py` (needs MODEL_PROXY_* in env / .env.model-proxy)
# Push:     `kaggle b t push kaggle-lab-smoke-math -f task.py --wait`

# %%
import kaggle_benchmarks as kbench

# %%
CASES = [
    ("What is 2 + 2? Reply with the number only.", "4"),
    ("What is 7 * 6? Reply with the number only.", "42"),
    ("What is 15 - 8? Reply with the number only.", "7"),
]


@kbench.task(name="kaggle-lab-smoke-math")
def smoke_math(llm) -> dict:
    """Tiny arithmetic checks — good first push/run for Model Proxy."""
    results = []
    for prompt, expected in CASES:
        response = llm.prompt(prompt)
        text = (response or "").strip()
        ok = expected in text
        kbench.assertions.assert_in(
            expected,
            text,
            expectation=f"response should contain {expected!r}, got {text!r}",
        )
        results.append({"prompt": prompt, "expected": expected, "got": text, "ok": ok})
    return {"n": len(results), "passed": sum(1 for r in results if r["ok"])}


# Required — without .run() the task is a silent no-op on the server.
smoke_math.run(kbench.llm)
