"""IDEA — Idea -> Deep research -> Experiment -> Analyze."""

from dataclasses import dataclass

__version__ = "0.2.0"


@dataclass
class Config:
    model: str = "claude-opus-4-6"
    max_cost: float = 50.0
    max_depth: int = 2
    max_branches: int = 3
    web_search: bool = True
