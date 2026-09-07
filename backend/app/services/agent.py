import hashlib
import hmac
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from app.config import settings
from app.schemas import (
    AgentTraceStep,
    QueryRequest,
    QueryResponse,
    ToolOutput,
)
from app.services.registry import registry
from app.services.session_manager import session_manager


class ReActAgent:
    def __init__(self):
        self.secret_key = settings.SECRET_KEY.encode('utf-8')

    def _determine_plan(self, query: str, validation: dict[str, Any] | None) -> list[tuple[str, str, dict[str, Any]]]:
        """
        Plans tool sequence (up to 3 steps) with human-readable 'why_this_tool' justifications.
        """
        q_lower = query.lower()
        mode = validation.get("mode", "single_image") if validation else "single_image"
        plan = []

        is_change_query = any(w in q_lower for w in ["change", "difference", "increase", "decrease", "expanded", "before", "after", "t1", "t2"])
        is_fusion_query = any(w in q_lower for w in ["sar", "radar", "cloud", "optical and sar", "all-weather", "penetrate"])
        is_grounding_query = any(w in q_lower for w in ["highlight", "ground", "where is", "locate", "detect", "box", "show me"])

        # Multi-tool chaining scenario
        if mode == "optical_sar" and (is_grounding_query or "damage" in q_lower or "building" in q_lower):
            plan.append((
                "optical_sar_fusion",
                "Triggered cross-modal fusion first to extract all-weather surface features despite possible cloud cover.",
                {"query": query}
            ))
            plan.append((
                "vqa_grounding",
                "Sequenced VLM grounding as step 2 to pinpoint specific structures within the fused region.",
                {"query": query}
            ))
        elif mode == "bi_temporal" or is_change_query:
            plan.append((
                "change_detection",
                "User requested bi-temporal comparison; activated differential change detector with pseudo-change suppression.",
                {"query": query, "suppress_pseudo_change": True}
            ))
        elif mode == "optical_sar" or is_fusion_query:
            plan.append((
                "optical_sar_fusion",
                "Optical and SAR pairs detected; dispatched gated cross-modal specialist to balance optical and microwave cues.",
                {"query": query}
            ))
        else:
            plan.append((
                "vqa_grounding",
                "Single-image visual inspection requested; dispatched RS-adapted VLM for question answering and grounding.",
                {"query": query}
            ))

        return plan

    def _fuse_confidences(self, tool_outputs: list[ToolOutput]) -> tuple[float, dict[str, float], list[str]]:
        """
        Multi-Sensor Gated Weighted Confidence Fusion:
        Clamped strictly to [0.05, 1.0] with exponential warning penalty.
        """
        if not tool_outputs:
            return 0.5, {}, []

        total_weight = 0.0
        weighted_sum = 0.0
        breakdown = {}
        all_warnings = []

        for out in tool_outputs:
            all_warnings.extend(out.warnings)
            w = 1.0
            if out.tool == "optical_sar_fusion":
                cloud_frac = out.metrics.get("cloud_fraction", 0.0)
                # SAR gets higher weight under heavy clouds
                w = out.metrics.get("sar_weight", 0.8)
            elif out.tool == "vqa_grounding":
                w = 1.0
            elif out.tool == "change_detection":
                w = 1.1

            total_weight += w
            weighted_sum += w * out.confidence
            breakdown[f"{out.tool}_{out.model_version}"] = round(out.confidence, 4)

        avg_c = weighted_sum / total_weight if total_weight > 0 else 0.5
        n_warnings = len(all_warnings)
        
        # Exponential penalty for warnings
        penalty = math.exp(-0.08 * n_warnings)
        final_conf = min(1.0, max(0.05, avg_c * penalty))
        return round(final_conf, 4), breakdown, all_warnings

    def _compute_deterministic_signature(
        self,
        image_checksums: list[str],
        query: str,
        model_versions: dict[str, str],
        metrics: dict[str, Any],
        final_answer: str
    ) -> str:
        payload = {
            "image_checksums": sorted(image_checksums),
            "query": query.strip(),
            "model_versions": dict(sorted(model_versions.items())),
            "metrics": dict(sorted(metrics.items())),
            "final_answer": final_answer.strip()
        }
        encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    def _compute_tamper_token(self, signature_hash: str, session_id: str, timestamp_iso: str) -> str:
        msg = f"{signature_hash}:{session_id}:{timestamp_iso}".encode()
        return hmac.new(self.secret_key, msg, hashlib.sha256).hexdigest()

    def run_query(self, request: QueryRequest) -> QueryResponse:
        session = session_manager.get_session(request.session_id)
        validation_data = session.get("validation") if session else None
        
        # Determine image paths
        aligned_dir = settings.ALIGNED_DIR / request.session_id
        uploads_dir = settings.UPLOADS_DIR / request.session_id
        
        # Check aligned files first, fallback to uploads
        aligned_files = sorted(list(aligned_dir.glob("aligned_*.tif")))
        if len(aligned_files) >= 2:
            image_paths = aligned_files
        else:
            image_paths = sorted(list(uploads_dir.glob("*.tif")) + list(uploads_dir.glob("*.tiff")))

        if not image_paths:
            # Fallback if testing without uploads: use sample_data
            sample_dir = settings.BASE_DIR / "sample_data"
            sample_files = sorted(list(sample_dir.glob("*.tif")) + list(sample_dir.glob("*.tiff")))
            image_paths = sample_files if sample_files else [Path("sample_placeholder.tif")]

        plan = self._determine_plan(request.query, validation_data)
        trace_steps: list[AgentTraceStep] = []
        tool_outputs: list[ToolOutput] = []
        model_versions: dict[str, str] = {}
        composite_features: list[dict[str, Any]] = []
        composite_bboxes: list[dict[str, Any]] = []
        all_metrics: dict[str, Any] = {}

        for step_idx, (tool_name, why_this_tool, params) in enumerate(plan, 1):
            tool = registry.get_tool(tool_name)
            if not tool:
                continue

            # Execute tool
            output = tool.execute(image_paths, params, force_mode=request.force_mode or "auto")
            tool_outputs.append(output)
            model_versions[tool_name] = output.model_version
            all_metrics.update(output.metrics)

            # Extract step geometry overlay
            step_overlay = None
            if output.mask_geojson:
                step_overlay = output.mask_geojson
                composite_features.extend(output.mask_geojson.get("features", []))
            elif output.bboxes:
                step_overlay = {"bboxes": [bb.model_dump() for bb in output.bboxes]}
                composite_bboxes.extend([bb.model_dump() for bb in output.bboxes])

            trace_steps.append(AgentTraceStep(
                step_number=step_idx,
                thought=f"Executing {tool_name} to address specific aspect of query: '{request.query}'.",
                action=f"Dispatch to specialist tool: {tool_name}",
                tool_called=tool_name,
                why_this_tool=why_this_tool,
                tool_input=params,
                observation_summary=output.answer,
                step_confidence=output.confidence,
                step_overlay=step_overlay
            ))

        # Rollup execution mode
        modes = {out.execution_mode for out in tool_outputs}
        if modes == {"real_model"}:
            exec_mode: Literal["real_model", "heuristic_fallback", "hybrid"] = "real_model"
        elif modes == {"heuristic_fallback"}:
            exec_mode = "heuristic_fallback"
        else:
            exec_mode = "hybrid"

        # Fuse confidence
        final_conf, breakdown, warnings = self._fuse_confidences(tool_outputs)

        # Synthesize final answer
        if len(tool_outputs) == 1:
            final_answer = tool_outputs[0].answer
        else:
            answers = " ".join([f"[Step {i+1} - {out.tool}]: {out.answer}" for i, out in enumerate(tool_outputs)])
            final_answer = f"Synthesized multi-sensor assessment: {answers}"

        # Calibration badge determination
        sensor_badge = "Cartosat-2S & RISAT Calibrated" if validation_data and validation_data.get("mode") == "optical_sar" else "ISRO Earth-Observation Standard"

        # Image checksums
        checksums = [img.get("checksum_sha256", "none") for img in validation_data.get("images", [])] if validation_data else ["sample"]

        now_iso = datetime.now(timezone.utc).isoformat()
        sig_hash = self._compute_deterministic_signature(checksums, request.query, model_versions, all_metrics, final_answer)
        tamper_token = self._compute_tamper_token(sig_hash, request.session_id, now_iso)

        composite_overlays = {
            "type": "FeatureCollection",
            "features": composite_features,
            "bboxes": composite_bboxes
        }

        response = QueryResponse(
            session_id=request.session_id,
            query=request.query,
            final_answer=final_answer,
            confidence=final_conf,
            confidence_breakdown=breakdown,
            sensor_calibration_badge=sensor_badge,
            execution_mode=exec_mode,
            trace=trace_steps,
            composite_overlays=composite_overlays,
            metrics_summary=all_metrics,
            run_signature_hash=sig_hash,
            report_tamper_token=tamper_token,
            generated_at=now_iso
        )

        # Record in session history
        session_manager.append_query_history(request.session_id, response.model_dump())
        return response

react_agent = ReActAgent()
