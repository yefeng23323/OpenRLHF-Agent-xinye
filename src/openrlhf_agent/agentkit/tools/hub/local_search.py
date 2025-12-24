"""Search tool that calls a local retrieval server and formats passages."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openrlhf_agent.agentkit.tools import ToolBase


class LocalSearchTool(ToolBase):
    """Calls a local Search-R1 style retriever and returns formatted passages."""

    name = "local_search"
    description = "Retrieve supporting passages from a local search backend. Use this to gather references before drafting the answer."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "User query to send to the retriever."},
            "topk": {"type": "integer", "description": "How many passages to return.", "minimum": 1, "maximum": 10, "default": 3},
            "return_scores": {"type": "boolean", "description": "Whether to include similarity scores.", "default": True},
        },
        "required": ["query"],
    }

    def __init__(self, *, base_url: str, timeout: float = 10.0):
        self.retriever_url = base_url
        self.timeout = timeout

    def _format_retrieval_hits(self, retrieval_hits: List[Dict[str, Any]]) -> str:
        """Format retrieval hits into a readable string."""

        formatted_hits = []
        for rank, hit in enumerate(retrieval_hits, start=1):
            document = hit.get("document", {}) or {}
            document_lines = str(document.get("contents", "")).split("\n")
            title_line = document_lines[0].strip() if document_lines else ""
            body_text = "\n".join(line for line in document_lines[1:] if line).strip()
            similarity_score = hit.get("score")

            label_parts = [f"Doc {rank}"]
            if title_line:
                label_parts.append(f"(Title: {title_line})")
            if similarity_score is not None:
                label_parts.append(f"[score: {similarity_score}]")

            hit_text = " ".join(label_parts)
            if body_text:
                hit_text = f"{hit_text} {body_text}"
            formatted_hits.append(hit_text)

        return "\n".join(formatted_hits)

    async def call(self, *, context: Dict[str, Any], arguments: Dict[str, Any]) -> str:
        import httpx

        query_text = str(arguments.get("query", "")).strip()
        if not query_text:
            return json.dumps({"ok": False, "error": "query is required"})

        max_results = int(arguments.get("topk") or 3)
        include_scores = bool(arguments.get("return_scores", True))

        request_payload = {
            "queries": [query_text],
            "topk": max_results,
            "return_scores": include_scores,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.retriever_url, json=request_payload)
                resp.raise_for_status()
                response_payload = resp.json()
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"request failed: {exc}"}, ensure_ascii=False)

        results_by_query = response_payload.get("result", [])
        hits_for_first_query = results_by_query[0] if results_by_query else []
        formatted_references = self._format_retrieval_hits(hits_for_first_query)

        return json.dumps({"ok": True, "references": formatted_references}, ensure_ascii=False)
