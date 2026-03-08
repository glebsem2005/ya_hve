"""Yandex Workflows API integration stub.

Yandex Workflows is a managed orchestration service.  This module
provides the interface for registering and running pipelines via
the Workflows API when it becomes available.

Currently returns mock responses with the local pipeline definition.
"""

from __future__ import annotations

import logging
import os

from cloud.workflows.pipeline import get_pipeline_definition

logger = logging.getLogger(__name__)

WORKFLOWS_ENDPOINT = os.getenv("YANDEX_WORKFLOWS_ENDPOINT", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")


async def register_workflow() -> dict:
    """Register the pipeline with Yandex Workflows (stub).

    In production, this would POST the pipeline definition to the
    Workflows API and return the workflow ID.
    """
    if WORKFLOWS_ENDPOINT:
        logger.info("Would register workflow at %s", WORKFLOWS_ENDPOINT)
        # TODO: implement real API call when Yandex Workflows API is available
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post(
        #         f"{WORKFLOWS_ENDPOINT}/v1/workflows",
        #         json=get_pipeline_definition(),
        #         headers={"Authorization": f"Bearer {YANDEX_API_KEY}"},
        #     )
        #     return resp.json()

    pipeline = get_pipeline_definition()
    logger.warning("Yandex Workflows stub: register_workflow (local mode)")
    return {
        "status": "registered_locally",
        "workflow_id": "local-forestguard-v2",
        "name": pipeline["name"],
        "steps": pipeline["total_steps"],
    }


async def run_workflow(workflow_id: str, input_data: dict) -> dict:
    """Trigger a workflow run (stub).

    In production, this would POST to the Workflows API to start
    an execution with the given input data.
    """
    if WORKFLOWS_ENDPOINT:
        logger.info("Would run workflow %s at %s", workflow_id, WORKFLOWS_ENDPOINT)

    logger.warning("Yandex Workflows stub: run_workflow(%s)", workflow_id)
    return {
        "status": "running_locally",
        "workflow_id": workflow_id,
        "execution_id": f"exec-{workflow_id}-local",
        "input": input_data,
    }


async def get_workflow_status(execution_id: str) -> dict:
    """Get workflow execution status (stub)."""
    logger.warning("Yandex Workflows stub: get_workflow_status(%s)", execution_id)
    return {
        "execution_id": execution_id,
        "status": "completed",
        "message": "Local execution (Yandex Workflows API not configured)",
    }
