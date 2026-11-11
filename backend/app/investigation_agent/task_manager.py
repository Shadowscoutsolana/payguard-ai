"""
Investigation Task Manager.
Generates AI case reports in the background (FastAPI BackgroundTasks), outside the synchronous
pre-capture decision path, and stores them on the alert. Every run is audit-logged with the
CATEGORIES of evidence sent to the LLM (never the values) and which path produced the report.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from .llm_service import InvestigationAgentService
from ..database import db

logger = logging.getLogger(__name__)


class InvestigationTaskManager:
    def __init__(self):
        self.service = InvestigationAgentService()
        self._task_status: Dict[str, Dict[str, Any]] = {}

    async def execute_investigation(self, alert_id: str, pack: Dict[str, Any]) -> Dict[str, Any]:
        """pack: anonymizer.build_case_evidence output (stored evidence only)."""
        self._task_status[alert_id] = {"status": "PROCESSING", "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            report = await self.service.generate_investigation_report(pack)
            db.update_alert_llm_report(alert_id, report)
            db.log_audit("ALERT", alert_id, "AI_REPORT_GENERATED", {
                "llm_status": report["llm_status"],
                "model": self.service.model if report["llm_status"] == "GEMINI" else None,
                "evidence_categories_sent_to_llm": report["evidence_categories_sent"],
                "evidence_item_count": len(pack["evidence"]),
                "fallback_reason": report["fallback_reason"],
            })
            self._task_status[alert_id] = {"status": "COMPLETED", "llm_status": report["llm_status"],
                                           "completed_at": datetime.now(timezone.utc).isoformat()}
            return report
        except Exception as exc:
            logger.exception("Investigation report failed for %s", alert_id)
            error_report = {"llm_status": "FAILED", "generated_by": "System error handler",
                            "error": type(exc).__name__, "advisory_notice": "Report generation failed; "
                            "review the stored evidence directly."}
            db.update_alert_llm_report(alert_id, error_report)
            db.log_audit("ALERT", alert_id, "AI_REPORT_FAILED", {"error": type(exc).__name__})
            self._task_status[alert_id] = {"status": "FAILED", "error": type(exc).__name__}
            return error_report

    def get_status(self, alert_id: str) -> Dict[str, Any]:
        return self._task_status.get(alert_id, {"status": "NOT_FOUND"})


# Singleton instance
task_manager = InvestigationTaskManager()
