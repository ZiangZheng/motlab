"""Manager subsystem — IsaacLab-style declarative term registry.

A manager-based env owns several managers (action, observation, reward,
termination, command, event, curriculum). Each manager is configured via a
``*ManagerCfg`` dataclass whose fields are ``*TermCfg`` instances; the
manager loops over its terms each step.
"""

from motlab.managers.action_manager import ActionManager, ActionTerm
from motlab.managers.command_manager import CommandManager, CommandTerm
from motlab.managers.curriculum_manager import CurriculumManager
from motlab.managers.event_manager import EventManager
from motlab.managers.manager_base import ManagerBase
from motlab.managers.manager_term_cfg import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from motlab.managers.observation_manager import ObservationManager
from motlab.managers.reward_manager import RewardManager
from motlab.managers.termination_manager import TerminationManager

__all__ = [
    "ActionManager",
    "ActionTerm",
    "ActionTermCfg",
    "CommandManager",
    "CommandTerm",
    "CommandTermCfg",
    "CurriculumManager",
    "CurriculumTermCfg",
    "EventManager",
    "EventTermCfg",
    "ManagerBase",
    "ObservationGroupCfg",
    "ObservationManager",
    "ObservationTermCfg",
    "RewardManager",
    "RewardTermCfg",
    "SceneEntityCfg",
    "TerminationManager",
    "TerminationTermCfg",
]
