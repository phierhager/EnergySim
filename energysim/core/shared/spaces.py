from dataclasses import dataclass
from typing import ClassVar, Literal, Union
from abc import abstractmethod, ABC
import numpy as np


class Space(ABC):
    type: ClassVar[Literal["discrete", "continuous"]]

    @abstractmethod
    def validate_action(self, action: Union[int, float]):
        pass

    @abstractmethod
    def sample(self) -> Union[int, float]:
        pass

@dataclass(frozen=True, slots=True, kw_only=True)
class DictSpace:
    spaces: dict[str, Space]

    type: Literal["dict"] = "dict"

    def validate_action(self, action: dict[str, Union[int, float]]):
        for key, space in self.spaces.items():
            if key not in action:
                raise ValueError(f"Action missing key: {key}")
            space.validate_action(action[key])

    def sample(self) -> dict[str, Union[int, float]]:
        return {key: space.sample() for key, space in self.spaces.items()}

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
        
    def sample(self) -> int:
        return np.random.randint(0, self.n_actions)


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
    
    def sample(self) -> float:
        return np.random.uniform(self.lower_bound, self.upper_bound)