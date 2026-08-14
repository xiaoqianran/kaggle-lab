# %% [markdown]
# kaggle-lab auto-authored benchmark task
# slug: __SLUG__
# generated_by: 014-camel-workforce-bench
# Validate: python task.py
# Push:     kaggle b t push __SLUG__ -f task.py --wait

# %%
import kaggle_benchmarks as kbench

# %%
# (prompt, expected_substring)
CASES = __CASES__


@kbench.task(name="__SLUG__")
def __FN__(llm) -> dict:
    """__DOC__"""
    results = []
    for prompt, expected in CASES:
        response = llm.prompt(prompt)
        text = (response or "").strip()
        kbench.assertions.assert_in(
            expected,
            text,
            expectation=f"expected {expected!r} in response, got {text!r}",
        )
        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "got": text,
                "ok": expected in text,
            }
        )
    return {
        "n": len(results),
        "passed": sum(1 for r in results if r["ok"]),
    }


# Required — without .run() the task is a silent no-op on the server.
__FN__.run(kbench.llm)
