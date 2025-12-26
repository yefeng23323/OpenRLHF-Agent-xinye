"""Search tool that calls a local retrieval server and formats passages."""

from __future__ import annotations

import json
from typing import Any, Dict

from openrlhf_agent.agentkit.tools import ToolBase


class LocalSearchTool(ToolBase):
    """Calls a local Search-R1 style retriever and returns formatted passages."""

    name = "search"
    description = "Searches for information related to `queries` and displays `topn` results."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "queries": {"type": "array", "items": {"type": "string"}, "description": "List of queries to send to the retriever."},
            "topk": {"type": "integer", "description": "How many passages to return for each query.", "minimum": 1, "maximum": 10, "default": 3},
        },
        "required": ["queries"],
    }

    def __init__(self, *, base_url: str, timeout: float = 10.0):
        self.retriever_url = base_url
        self.timeout = timeout

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        import httpx
        queries = arguments.get("queries", [])
        if not queries:
            return "queries are required"
        max_results = int(arguments.get("topk", 3))
        
        request_payload = {
            "queries": queries,
            "topk": max_results,
            "return_scores": True,
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.retriever_url, json=request_payload)
                resp.raise_for_status()
                results = [self._passages2string(result) for result in resp.json()['result']]
        except Exception as exc:
            return f"request failed: {exc}"
        
        return json.dumps(results, ensure_ascii=False)
