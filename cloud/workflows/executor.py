"""Workflow executor — runs pipeline steps in dependency order.

Handles timing instrumentation, error handling, and WebSocket broadcast
of step progress to the dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    step_id: str
    status: str  # "success", "failed", "skipped"
    duration_ms: float
    output: Any = None
    error: str | None = None


@dataclass
class WorkflowResult:
    """Result of full pipeline execution."""

    steps: list[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0.0
    status: str = "success"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "steps": [
                {
                    "step_id": s.step_id,
                    "status": s.status,
                    "duration_ms": round(s.duration_ms, 1),
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


class WorkflowExecutor:
    """Execute pipeline steps in dependency order with instrumentation."""

    def __init__(
        self,
        broadcast_fn: Callable[[dict], Coroutine] | None = None,
    ) -> None:
        self._steps: dict[str, Callable] = {}
        self._broadcast = broadcast_fn

    def register(self, step_id: str, handler: Callable) -> None:
        """Register a handler function for a pipeline step."""
        self._steps[step_id] = handler

    async def _broadcast_event(self, event: dict) -> None:
        if self._broadcast:
            try:
                await self._broadcast(event)
            except Exception:
                pass

    async def execute_step(
        self,
        step_id: str,
        context: dict,
        optional: bool = False,
        timeout: float = 30.0,
    ) -> StepResult:
        """Execute a single pipeline step with timing and error handling."""
        handler = self._steps.get(step_id)
        if handler is None:
            if optional:
                return StepResult(step_id=step_id, status="skipped", duration_ms=0.0)
            return StepResult(
                step_id=step_id,
                status="failed",
                duration_ms=0.0,
                error=f"No handler registered for step '{step_id}'",
            )

        await self._broadcast_event(
            {"event": "workflow_step_start", "step_id": step_id}
        )

        t0 = time.monotonic()
        try:
            if asyncio.iscoroutinefunction(handler):
                output = await asyncio.wait_for(handler(context), timeout=timeout)
            else:
                output = handler(context)
            duration = (time.monotonic() - t0) * 1000

            await self._broadcast_event(
                {
                    "event": "workflow_step_done",
                    "step_id": step_id,
                    "status": "success",
                    "duration_ms": round(duration, 1),
                }
            )

            return StepResult(
                step_id=step_id,
                status="success",
                duration_ms=duration,
                output=output,
            )

        except asyncio.TimeoutError:
            duration = (time.monotonic() - t0) * 1000
            logger.warning("Step %s timed out after %.0fms", step_id, duration)
            if optional:
                return StepResult(
                    step_id=step_id,
                    status="skipped",
                    duration_ms=duration,
                    error="timeout",
                )
            return StepResult(
                step_id=step_id,
                status="failed",
                duration_ms=duration,
                error="timeout",
            )

        except Exception as e:
            duration = (time.monotonic() - t0) * 1000
            logger.error("Step %s failed: %s", step_id, e)
            if optional:
                return StepResult(
                    step_id=step_id,
                    status="skipped",
                    duration_ms=duration,
                    error=str(e),
                )
            return StepResult(
                step_id=step_id,
                status="failed",
                duration_ms=duration,
                error=str(e),
            )

    async def run(
        self,
        step_ids: list[str],
        context: dict,
        optional_steps: set[str] | None = None,
    ) -> WorkflowResult:
        """Run a sequence of pipeline steps, collecting results."""
        optional_steps = optional_steps or set()
        result = WorkflowResult()
        t0 = time.monotonic()

        for step_id in step_ids:
            step_result = await self.execute_step(
                step_id,
                context,
                optional=step_id in optional_steps,
            )
            result.steps.append(step_result)
            context[f"result_{step_id}"] = step_result

            # Stop on non-optional failure
            if step_result.status == "failed" and step_id not in optional_steps:
                result.status = "failed"
                break

        result.total_duration_ms = (time.monotonic() - t0) * 1000
        if result.status != "failed":
            result.status = "success"

        await self._broadcast_event(
            {
                "event": "workflow_complete",
                "status": result.status,
                "total_duration_ms": round(result.total_duration_ms, 1),
            }
        )

        return result
