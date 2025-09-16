from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, Union


@dataclass(frozen=True, slots=True, kw_only=True)
class DiscreteSpace:
    n_actions: int

    type: Literal["discrete"] = "discrete"

    def __post_init__(self):
        if self.n_actions <= 1:
            raise ValueError("n_actions must be greater than 1 for discrete control.")

    def validate_action(self, action: int):
        if not (0 <= action < self.n_actions):
            raise ValueError(
                f"Action {action} out of bounds [0, {self.n_actions - 1}]."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuousSpace:
    lower_bound: float
    upper_bound: float

    type: Literal["continuous"] = "continuous"

    def __post_init__(self):
        if self.lower_bound >= self.upper_bound:
            raise ValueError(
                "lower_bound must be less than upper_bound for continuous control."
            )

    def validate_action(self, action: float):
        if not (self.lower_bound <= action <= self.upper_bound):
            raise ValueError(
                f"Action {action} out of bounds [{self.lower_bound}, {self.upper_bound}]."
            )


Space = Union[DiscreteSpace, ContinuousSpace]
