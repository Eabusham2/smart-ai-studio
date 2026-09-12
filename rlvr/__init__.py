"""RLVR package runtime integrity wiring."""
from rlvr.master_curriculum import MasterCurriculumOrchestrator
from rlvr.runtime_hardening import install_master_curriculum_hardening

install_master_curriculum_hardening(MasterCurriculumOrchestrator)

__all__ = ["MasterCurriculumOrchestrator"]
