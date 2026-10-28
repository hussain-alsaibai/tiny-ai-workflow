"""Tests for tiny-ai-workflow."""

import time
import unittest
from tiny_ai_workflow import Workflow, WorkflowContext


class TestWorkflowContext(unittest.TestCase):
    def test_result_set_get(self):
        ctx = WorkflowContext()
        ctx.step_results["fetch"] = [1, 2, 3]
        self.assertEqual(ctx.result("fetch"), [1, 2, 3])

    def test_result_default(self):
        ctx = WorkflowContext()
        self.assertIsNone(ctx.result("missing"))
        self.assertEqual(ctx.result("missing", "default"), "default")

    def test_metadata(self):
        ctx = WorkflowContext()
        ctx.set("key", "value")
        self.assertEqual(ctx.get("key"), "value")


class TestWorkflowBasics(unittest.TestCase):
    def test_linear_steps(self):
        wf = Workflow(name="test")

        @wf.step()
        def first(ctx):
            return "first"

        @wf.step(depends_on="first")
        def second(ctx):
            return ctx.result("first") + " second"

        result = wf.run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.context.result("second"), "first second")

    def test_retry_on_failure(self):
        wf = Workflow(name="test")

        attempts = []
        @wf.step(retry=3, retry_delay=0.01)
        def flaky(ctx):
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("not yet")
            return "success"

        result = wf.run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(attempts), 2)
        self.assertEqual(result.context.result("flaky"), "success")

    def test_step_failure(self):
        wf = Workflow(name="test")

        @wf.step()
        def broken(ctx):
            raise RuntimeError("intentional")

        result = wf.run()
        self.assertIn(result.status, ("partial", "failed"))
        self.assertEqual(result.context.error("broken"), "intentional")

    def test_diamond_dependency(self):
        wf = Workflow(name="test")

        @wf.step()
        def start(ctx):
            return "start"

        @wf.step(depends_on="start")
        def a(ctx):
            return "a"

        @wf.step(depends_on="start")
        def b(ctx):
            return "b"

        @wf.step(depends_on=["a", "b"])
        def end(ctx):
            return ctx.result("a") + ctx.result("b")

        result = wf.run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.context.result("end"), "ab")

    def test_no_depends(self):
        wf = Workflow(name="test")

        @wf.step()
        def parallel_a(ctx):
            return "a"

        @wf.step()
        def parallel_b(ctx):
            return "b"

        result = wf.run()
        self.assertEqual(result.status, "completed")

    def test_context_metadata(self):
        wf = Workflow(name="test")

        @wf.step()
        def init(ctx):
            ctx.set("mode", "test")
            return ctx.get("mode")

        result = wf.run()
        self.assertEqual(result.context.get("mode"), "test")


class TestWorkflowHooks(unittest.TestCase):
    def test_hooks_called(self):
        started = []
        ended = []

        wf = Workflow(
            name="test",
            on_step_start=lambda ctx, n: started.append(n),
            on_step_end=lambda ctx, n, r: ended.append(n),
        )

        @wf.step()
        def step1(ctx):
            return "done"

        result = wf.run()
        self.assertIn("step1", started)
        self.assertIn("step1", ended)


class TestWorkflowResult(unittest.TestCase):
    def test_summary(self):
        wf = Workflow(name="test")

        @wf.step()
        def ok(ctx):
            return "ok"

        result = wf.run()
        self.assertIn("completed", result.summary())
        # "ok" succeeded, so it must NOT be reported as failed.
        self.assertNotIn("ok", result.failed_steps())
        self.assertEqual(result.failed_steps(), [])

    def test_is_complete(self):
        wf = Workflow(name="test")

        @wf.step()
        def step1(ctx):
            return True

        result = wf.run()
        self.assertTrue(result.is_complete())


if __name__ == "__main__":
    unittest.main()
