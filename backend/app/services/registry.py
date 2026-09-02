
from app.schemas import ToolInfo
from app.tools.base import BaseSpecialistTool
from app.tools.tool1_vlm import vlm_tool
from app.tools.tool3_change import change_tool
from app.tools.tool4_fusion import fusion_tool


class ModelRegistry:
    def __init__(self):
        self._tools: dict[str, BaseSpecialistTool] = {
            "vqa_grounding": vlm_tool,
            "change_detection": change_tool,
            "optical_sar_fusion": fusion_tool,
        }

    def get_tool(self, tool_name: str) -> BaseSpecialistTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[ToolInfo]:
        infos = []
        for name, tool in self._tools.items():
            infos.append(ToolInfo(
                name=name,
                description=f"Specialist executor for {name.replace('_', ' ').title()}",
                tier_available=tool.tier_available,
                active_mode="real_model" if tool.tier_available in ("gpu", "quantized_cpu") else "heuristic_fallback",
                model_version=tool.default_version
            ))
        return infos

registry = ModelRegistry()
