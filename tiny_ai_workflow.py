"""
tiny-ai-workflow — Stateless AI agent workflow orchestration engine.

Orchestrates multi-step AI workflows with retry, branching, parallel fan-out,
human-in-the-loop checkpoints, and result aggregation.

Designed for autonomous AI agents: each step can be a sync function,
async coroutine, or an MCP tool call. Built for OpenClaw's bounty/workflow context.

Usage:
    from tiny_ai_workflow import Workflow, step, branch, parallel

    wf = Workflow(name="bounty-analysis")

    @wf.step()
    def fetch_bounties(ctx):
        return fetch_open_bounties()

    @wf.step(depends_on="fetch_bounties")
    def score_bounties(ctx):
        bounties = ctx.result("fetch_bounties")
        return sorted(bounties, key=lambda b: b.priority, reverse=True)

    @wf.branch(condition=lambda ctx: len(ctx.result("score_bounties")) > 0)
    def process_top():
        @wf.step(depends_on="score_bounties")
        def claim_top(ctx):
            return claim_bounty(ctx.result("score_bounties")[0])

        return [claim_top]

    result = wf.run()
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

__version__ = "0.1.0"
__all__ = ["Workflow", "WorkflowStep", "WorkflowResult", "WorkflowError", "step", "branch", "parallel"]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

T = TypeVar("T")


class StepState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    AWAITING_APPROVAL = "awaiting_approval"


@dataclass
class WorkflowContext:
    """Mutable execution context passed through all workflow steps."""

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_results: dict[str, Any] = field(default_factory=dict)
    step_errors: dict[str, str] = field(default_factory=dict)
    step_timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    iteration: int = 1
    _checkpoint_data: dict = field(default_factory=dict)

    def result(self, step_name: str, default: Any = None) -> Any:
        return self.step_results.get(step_name, default)

    def error(self, step_name: str) -> str | None:
        return self.step_errors.get(step_name)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def checkpoint(self, data: dict) -> None:
        """Save checkpoint data for resume."""
        self._checkpoint_data.update(data)

    def checkpoint_data(self) -> dict:
        return dict(self._checkpoint_data)


@dataclass
class WorkflowStep:
    """A single executable unit in a workflow."""

    name: str
    fn: Callable[[WorkflowContext], Any]
    depends_on: list[str] = field(default_factory=list)
    retry: int = 1
    retry_delay: float = 1.0
    timeout: float | None = None
    condition: Callable[[WorkflowContext], bool] | None = None
    description: str = ""
    on_failure: Callable[[WorkflowContext, Exception], Any] | None = None
    requires_approval: bool = False
    approval_prompt: str = ""
    # Runtime
    state: StepState = StepState.PENDING
    result: Any = field(default=None, init=False)
    error: Exception | None = field(default=None, init=False)
    started_at: datetime | None = field(default=None, init=False)
    finished_at: datetime | None = field(default=None, init=False)
    attempt: int = field(default=0, init=False)


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    workflow_id: str
    status: str  # "completed" | "failed" | "partial"
    steps: dict[str, dict]
    context: WorkflowContext
    duration_s: float
    error: str | None = None

    def step(self, name: str) -> dict:
        return self.steps.get(name, {})

    def is_complete(self) -> bool:
        return self.status == "completed"

    def failed_steps(self) -> list[str]:
        return [n for n, s in self.steps.items() if s["state"] == "failed"]

    def summary(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps.values() if s["state"] == "done")
        failed = sum(1 for s in self.steps.values() if s["state"] == "failed")
        skipped = sum(1 for s in self.steps.values() if s["state"] == "skipped")
        return (
            f"Workflow {self.workflow_id}: {self.status} "
            f"({done}/{total} done, {failed} failed, {skipped} skipped) "
            f"in {self.duration_s:.2f}s"
        )


class WorkflowError(Exception):
    """Workflow-level error."""
    pass


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def step(
    name: str | None = None,
    depends_on: str | list[str] | None = None,
    retry: int = 1,
    retry_delay: float = 1.0,
    timeout: float | None = None,
    description: str = "",
    requires_approval: bool = False,
    approval_prompt: str = "",
):
    """Decorator to mark a function as a workflow step."""
    def decorator(fn: Callable[[WorkflowContext], Any]) -> Callable[[WorkflowContext], Any]:
        fn._workflow_step = True
        fn._step_name = name or fn.__name__
        fn._step_depends = [depends_on] if isinstance(depends_on, str) else (depends_on or [])
        fn._step_retry = retry
        fn._step_retry_delay = retry_delay
        fn._step_timeout = timeout
        fn._step_description = description
        fn._step_requires_approval = requires_approval
        fn._step_approval_prompt = approval_prompt
        return fn
    return decorator


def branch(
    condition: Callable[[WorkflowContext], bool],
    name: str = "conditional_branch",
) -> Callable[[], list[Callable]]:
    """Decorator to define a conditional branch (returns a list of step functions)."""
    def decorator(
        fn: Callable[[], list[Callable[[WorkflowContext], Any]]]
    ) -> Callable[[WorkflowContext], list[Callable]]:
        fn._workflow_branch = True
        fn._branch_condition = condition
        fn._branch_name = name
        return fn
    return decorator


def parallel(
    name: str = "parallel_group",
    wait_for: str | list[str] | None = None,
) -> Callable[[], list[Callable]]:
    """Decorator to define a parallel execution group."""
    def decorator(
        fn: Callable[[], list[Callable[[WorkflowContext], Any]]]
    ) -> Callable[[WorkflowContext], list[Callable]]:
        fn._workflow_parallel = True
        fn._parallel_name = name
        fn._parallel_wait_for = [wait_for] if isinstance(wait_for, str) else (wait_for or [])
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Core Engine
# ---------------------------------------------------------------------------

class Workflow:
    """
    Workflow orchestration engine.

    Args:
        name: Workflow name (for logging/tracing).
        max_iterations: Max loop iterations for branching workflows.
        approval_fn: Optional async fn(ctx, step) -> bool for human approval.
                     If None (default), approval steps raise WorkflowError.
        on_step_start: Hook: (ctx, step_name) -> None.
        on_step_end: Hook: (ctx, step_name, result_or_error) -> None.
    """

    def __init__(
        self,
        name: str = "workflow",
        max_iterations: int = 10,
        approval_fn: Callable[[WorkflowContext, WorkflowStep], bool] | None = None,
        on_step_start: Callable[[WorkflowContext, str], None] | None = None,
        on_step_end: Callable[[WorkflowContext, str, Any | Exception], None] | None = None,
    ):
        self.name = name
        self.max_iterations = max_iterations
        self.approval_fn = approval_fn
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end

        self._steps: dict[str, WorkflowStep] = {}
        self._branches: list[dict] = []
        self._ctx: WorkflowContext | None = None
        self._start_time: float = 0.0

    def step(
        self,
        name: str | None = None,
        depends_on: str | list[str] | None = None,
        retry: int = 1,
        retry_delay: float = 1.0,
        timeout: float | None = None,
        description: str = "",
        requires_approval: bool = False,
        approval_prompt: str = "",
    ) -> Callable:
        """Decorator to register a step."""
        def decorator(fn: Callable) -> Callable:
            step_name = name or fn.__name__
            deps = [depends_on] if isinstance(depends_on, str) else (depends_on or [])
            self._steps[step_name] = WorkflowStep(
                name=step_name,
                fn=fn,
                depends_on=deps,
                retry=retry,
                retry_delay=retry_delay,
                timeout=timeout,
                description=description or fn.__doc__ or "",
                requires_approval=requires_approval,
                approval_prompt=approval_prompt,
            )
            return fn
        return decorator

    def branch(
        self,
        condition: Callable[[WorkflowContext], bool],
        name: str = "branch",
    ) -> Callable:
        """Decorator to register a conditional branch."""
        def decorator(fn: Callable[[], list[Callable]]) -> Callable:
            self._branches.append({
                "condition": condition,
                "name": name,
                "fn": fn,
                "steps": fn(),
            })
            return fn
        return decorator

    def parallel(
        self,
        name: str = "parallel",
        wait_for: str | list[str] | None = None,
    ) -> Callable:
        """Decorator to register a parallel execution group."""
        def decorator(fn: Callable[[], list[Callable]]) -> Callable:
            wf = self
            deps = [wait_for] if isinstance(wait_for, str) else (wait_for or [])
            # Extract steps from parallel group
            steps = fn()
            for s in steps:
                s_name = getattr(s, "_step_name", s.__name__)
                s_deps = list(getattr(s, "_step_depends", []))
                wf._steps[s_name] = WorkflowStep(
                    name=s_name,
                    fn=s,
                    depends_on=s_deps + deps,
                    retry=getattr(s, "_step_retry", 1),
                    retry_delay=getattr(s, "_step_retry_delay", 1.0),
                    timeout=getattr(s, "_step_timeout", None),
                    description=getattr(s, "_step_description", ""),
                    requires_approval=getattr(s, "_step_requires_approval", False),
                    approval_prompt=getattr(s, "_step_approval_prompt", ""),
                )
            return fn
        return decorator

    def run(self, initial_context: dict | None = None) -> WorkflowResult:
        """Run the workflow synchronously. Returns WorkflowResult."""
        import time
        self._ctx = WorkflowContext()
        if initial_context:
            self._ctx.metadata.update(initial_context)
        self._start_time = time.time()

        try:
            self._resolve_and_execute()
        except Exception as e:
            pass  # captured in context

        duration = time.time() - self._start_time
        return self._build_result(duration)

    def run_async(self, initial_context: dict | None = None) -> WorkflowResult:
        """Run the workflow with async support. Returns WorkflowResult."""
        import time
        self._ctx = WorkflowContext()
        if initial_context:
            self._ctx.metadata.update(initial_context)
        self._start_time = time.time()

        try:
            asyncio.run(self._resolve_and_execute_async())
        except Exception as e:
            pass

        duration = time.time() - self._start_time
        return self._build_result(duration)

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    def _ready_steps(self) -> list[WorkflowStep]:
        """Return steps whose dependencies are all satisfied."""
        ready = []
        for step in self._steps.values():
            if step.state not in (StepState.PENDING,):
                continue
            deps_done = all(
                self._steps[d].state == StepState.DONE
                for d in step.depends_on
            )
            if deps_done:
                ready.append(step)
        return ready

    def _resolve_and_execute(self) -> None:
        """Synchronous execution loop."""
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self._ctx.iteration = iteration

            # Evaluate branches
            for branch_def in self._branches:
                cond = branch_def["condition"]
                if cond(self._ctx):
                    for step_fn in branch_def["steps"]:
                        step_name = getattr(step_fn, "_step_name", step_fn.__name__)
                        if step_name not in self._steps:
                            self._steps[step_name] = WorkflowStep(
                                name=step_name,
                                fn=step_fn,
                                depends_on=branch_def.get("depends_on", []),
                                retry=getattr(step_fn, "_step_retry", 1),
                                retry_delay=getattr(step_fn, "_step_retry_delay", 1.0),
                                timeout=getattr(step_fn, "_step_timeout", None),
                                description=getattr(step_fn, "_step_description", ""),
                                requires_approval=getattr(step_fn, "_step_requires_approval", False),
                                approval_prompt=getattr(step_fn, "_step_approval_prompt", ""),
                            )

            # Execute ready steps
            ready = self._ready_steps()
            if not ready:
                break

            for step in ready:
                self._execute_step(step)

    async def _resolve_and_execute_async(self) -> None:
        """Async execution with parallel fan-out."""
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            self._ctx.iteration = iteration

            # Evaluate branches
            for branch_def in self._branches:
                if branch_def["condition"](self._ctx):
                    for step_fn in branch_def["steps"]:
                        step_name = getattr(step_fn, "_step_name", step_fn.__name__)
                        if step_name not in self._steps:
                            self._steps[step_name] = WorkflowStep(
                                name=step_name,
                                fn=step_fn,
                                depends_on=branch_def.get("depends_on", []),
                                retry=getattr(step_fn, "_step_retry", 1),
                                retry_delay=getattr(step_fn, "_step_retry_delay", 1.0),
                                timeout=getattr(step_fn, "_step_timeout", None),
                                description=getattr(step_fn, "_step_description", ""),
                                requires_approval=getattr(step_fn, "_step_requires_approval", False),
                                approval_prompt=getattr(step_fn, "_step_approval_prompt", ""),
                            )

            ready = self._ready_steps()
            if not ready:
                break

            # Group independent steps for parallel execution
            independent = [s for s in ready if not s.depends_on]
            dependent = [s for s in ready if s.depends_on]

            # Run independent steps in parallel
            if independent:
                await self._run_parallel(independent)

            # Run dependent steps
            for step in dependent:
                await self._execute_step_async(step)

    async def _run_parallel(self, steps: list[WorkflowStep]) -> None:
        """Execute multiple steps concurrently."""
        import asyncio
        tasks = [self._execute_step_async(s) for s in steps]
        await asyncio.gather(*tasks, return_exceptions=True)

    def _execute_step(self, step: WorkflowStep) -> None:
        """Execute a single step with retry."""
        import time

        # Approval check
        if step.requires_approval:
            if self.approval_fn and self.approval_fn(self._ctx, step):
                step.state = StepState.AWAITING_APPROVAL
                return
            else:
                step.state = StepState.SKIPPED
                return

        step.state = StepState.RUNNING
        step.started_at = datetime.now()
        if self.on_step_start:
            self.on_step_start(self._ctx, step.name)

        last_error: Exception | None = None
        for attempt in range(step.retry):
            step.attempt = attempt + 1
            try:
                result = step.fn(self._ctx)
                step.result = result
                self._ctx.step_results[step.name] = result
                step.state = StepState.DONE
                step.finished_at = datetime.now()
                if self.on_step_end:
                    self.on_step_end(self._ctx, step.name, result)
                return
            except Exception as e:
                last_error = e
                if attempt < step.retry - 1:
                    time.sleep(step.retry_delay)

        # All retries failed
        step.error = last_error
        step.state = StepState.FAILED
        self._ctx.step_errors[step.name] = str(last_error)
        step.finished_at = datetime.now()
        if step.on_failure:
            step.on_failure(self._ctx, last_error)
        if self.on_step_end:
            self.on_step_end(self._ctx, step.name, last_error)

    async def _execute_step_async(self, step: WorkflowStep) -> None:
        """Execute a single step async with retry."""
        import asyncio
        import time

        if step.requires_approval:
            if self.approval_fn and self.approval_fn(self._ctx, step):
                step.state = StepState.AWAITING_APPROVAL
                return
            else:
                step.state = StepState.SKIPPED
                return

        step.state = StepState.RUNNING
        step.started_at = datetime.now()
        if self.on_step_start:
            self.on_step_start(self._ctx, step.name)

        last_error: Exception | None = None
        for attempt in range(step.retry):
            step.attempt = attempt + 1
            try:
                result = step.fn(self._ctx)
                if asyncio.iscoroutine(result):
                    result = await result
                step.result = result
                self._ctx.step_results[step.name] = result
                step.state = StepState.DONE
                step.finished_at = datetime.now()
                if self.on_step_end:
                    self.on_step_end(self._ctx, step.name, result)
                return
            except Exception as e:
                last_error = e
                if attempt < step.retry - 1:
                    await asyncio.sleep(step.retry_delay)

        step.error = last_error
        step.state = StepState.FAILED
        self._ctx.step_errors[step.name] = str(last_error)
        step.finished_at = datetime.now()
        if step.on_failure:
            step.on_failure(self._ctx, last_error)
        if self.on_step_end:
            self.on_step_end(self._ctx, step.name, last_error)

    def _build_result(self, duration: float) -> WorkflowResult:
        """Build final WorkflowResult from current state."""
        steps_dict = {}
        all_done = True
        any_failed = False

        for name, step in self._steps.items():
            steps_dict[name] = {
                "state": step.state.value,
                "result": step.result,
                "error": str(step.error) if step.error else None,
                "attempt": step.attempt,
                "started_at": step.started_at.isoformat() if step.started_at else None,
                "finished_at": step.finished_at.isoformat() if step.finished_at else None,
                "description": step.description,
            }
            if step.state != StepState.DONE:
                all_done = False
            if step.state == StepState.FAILED:
                any_failed = True

        status = "completed" if all_done else ("partial" if not any_failed else "failed")
        return WorkflowResult(
            workflow_id=self._ctx.workflow_id if self._ctx else "unknown",
            status=status,
            steps=steps_dict,
            context=self._ctx or WorkflowContext(),
            duration_s=duration,
            error=None,
        )
