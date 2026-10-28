# tiny-ai-workflow

> **Stateless AI agent workflow orchestration engine.**  
> Multi-step workflows with retry, branching, parallel fan-out, and human-in-the-loop checkpoints.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://github.com/hussain-alsaibai/tiny-ai-workflow#install)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why?

AI agents need to orchestrate complex multi-step tasks: fetch → analyze → decide → act → report. `tiny-ai-workflow` provides a clean, testable framework for this — without the weight of Airflow, Temporal, or LangGraph.

**Zero dependencies.** Pure Python.

## Features

- **Step dependencies** — DAG-based execution order
- **Retry with backoff** — per-step retry count and delay
- **Conditional branching** — `if`/`else` logic based on context
- **Parallel execution** — async fan-out for independent steps
- **Human-in-the-loop** — approval checkpoints for critical steps
- **Timeout support** — per-step time limits
- **Hook system** — on_step_start / on_step_end callbacks
- **Checkpoint & resume** — save/restore workflow state
- **Full async** — `run_async()` for asyncio workloads

## Installation

```bash
pip install tiny-ai-workflow
```

## Quick Start

```python
from tiny_ai_workflow import Workflow

wf = Workflow(name="bounty-hunter")

@wf.step()
def fetch_bounties(ctx):
    """Fetch open bounties from all platforms."""
    return ctx.set("bounties", get_open_bounties())

@wf.step(depends_on="fetch_bounties")
def score_bounties(ctx):
    """Score and rank bounties by priority."""
    bounties = ctx.result("fetch_bounties")
    scored = sorted(bounties, key=lambda b: b.reward / b.effort, reverse=True)
    ctx.set("top_bounty", scored[0])
    return scored

@wf.step(depends_on="score_bounties")
def claim_bounty(ctx):
    """Claim the top-ranked bounty."""
    top = ctx.get("top_bounty")
    return claim(top)

result = wf.run()
print(result.summary())
# Workflow abc123: completed (3/3 done, 0 failed, 0 skipped) in 2.34s
```

## Advanced: Branching

```python
wf = Workflow(name="auto-fix")

@wf.branch(condition=lambda ctx: ctx.result("scan") is not None)
def on_vulnerable():
    @wf.step(depends_on="scan")
    def analyze(ctx):
        return analyze_vulns(ctx.result("scan"))

    @wf.step(depends_on="analyze")
    def submit_fix(ctx):
        return submit_pull_request(ctx.result("analyze"))

    return [analyze, submit_fix]
```

## Advanced: Parallel Execution

```python
@wf.parallel(name="scrape_all", wait_for="fetch_repos")
def parallel_tasks():
    @wf.step(depends_on=[])
    def check_stars():
        return check_star_count()

    @wf.step(depends_on=[])
    def check_readme():
        return check_readme_quality()

    return [check_stars, check_readme]
```

## Async with Human Approval

```python
async def ask_human(ctx, step) -> bool:
    # Integrate with your notification system
    return await send_approval_request("High-value bounty detected. Approve?")

wf = Workflow(
    name="auto-bounty",
    approval_fn=ask_human,
)

@wf.step(requires_approval=True, approval_prompt="Submit this $5000 bounty fix?")
def submit_high_value_fix(ctx):
    return submit_pr(high_value_bounty)
```

## Hooks & Monitoring

```python
def on_step_start(ctx, name):
    print(f"[START] {name}")

def on_step_end(ctx, name, result_or_error):
    if isinstance(result_or_error, Exception):
        print(f"[FAIL]  {name}: {result_or_error}")
    else:
        print(f"[DONE]  {name}")

wf = Workflow(
    name="monitored",
    on_step_start=on_step_start,
    on_step_end=on_step_end,
)
```

## API Reference

### `Workflow(name, ...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Workflow identifier |
| `max_iterations` | `int` | Max loop iterations (default 10) |
| `approval_fn` | `callable` | `(ctx, step) → bool` for human approval |
| `on_step_start` | `callable` | Hook before step execution |
| `on_step_end` | `callable` | Hook after step completion |

### `@wf.step()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `depends_on` | `str \| list` | Upstream step names |
| `retry` | `int` | Retry attempts on failure |
| `retry_delay` | `float` | Seconds between retries |
| `timeout` | `float` | Max seconds before timeout |
| `requires_approval` | `bool` | Human approval gate |
| `approval_prompt` | `str` | Message shown for approval |

## License

MIT — do anything, just attribute.
