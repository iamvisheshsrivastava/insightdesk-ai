# src/agentic/tools/routing_tool.py
"""
Routing Tool for Agentic AI System

Implements issue #21: once a ticket is classified, suggest which support
team/queue it should be routed to (e.g. "Billing" -> Finance queue,
"Network" -> Infra on-call, "Hardware" -> Field Ops).

The mapping is config-driven (config/routing_map.json) so ops can edit it
without code changes, leaving room for a learned routing model later.
"""
from typing import Any, Dict
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# config/routing_map.json lives at the project root, two levels above src/agentic/tools/
DEFAULT_ROUTING_MAP_PATH = Path(__file__).resolve().parents[3] / "config" / "routing_map.json"

# High-urgency priorities get escalated to an on-call variant of the queue
# rather than the regular triage queue.
_ESCALATED_PRIORITIES = {"high", "critical"}


class RoutingTool:
    """
    Tool that maps a classified ticket's (category, priority) to a
    suggested support team and queue.
    """

    name = "routing_suggester"
    description = (
        "Suggests which support team/queue a classified ticket should be "
        "routed to, based on its predicted category and priority."
    )

    def __init__(self, routing_map_path: Path = None):
        self.routing_map_path = routing_map_path or DEFAULT_ROUTING_MAP_PATH
        self.routing_map = self._load_routing_map(self.routing_map_path)

    @staticmethod
    def _load_routing_map(path: Path) -> Dict[str, Dict[str, str]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception as e:
            logger.warning(f"Could not load routing map from {path}: {e}. Using built-in default.")
            return {"_default": {"team": "General Support", "queue": "general-triage"}}

    def run(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest a routing destination for a ticket.

        Args:
            tool_input: Dict that must contain the predicted "category"
                (as produced by ClassificationTool) and may contain
                "priority" (low/medium/high/critical) from the original
                ticket data.

        Returns:
            Dict with:
                - suggested_team (str)
                - suggested_queue (str)
                - escalated (bool): whether priority bumped it to an on-call queue
                - reasoning (str)
        """
        category = (tool_input.get("category") or "").strip().lower()
        priority = (tool_input.get("priority") or "").strip().lower()

        default_entry = self.routing_map.get("_default", {"team": "General Support", "queue": "general-triage"})
        entry = self.routing_map.get(category, default_entry)

        team = entry.get("team", default_entry.get("team"))
        queue = entry.get("queue", default_entry.get("queue"))

        escalated = priority in _ESCALATED_PRIORITIES
        if escalated and not queue.endswith("-escalated"):
            queue = f"{queue}-escalated"

        reasoning = f"Category '{category or 'unknown'}' maps to {team} ({queue})."
        if escalated:
            reasoning += f" Escalated due to '{priority}' priority."

        logger.info(f"RoutingTool: category={category!r} priority={priority!r} -> team={team!r} queue={queue!r}")

        return {
            "suggested_team": team,
            "suggested_queue": queue,
            "escalated": escalated,
            "reasoning": reasoning,
        }
