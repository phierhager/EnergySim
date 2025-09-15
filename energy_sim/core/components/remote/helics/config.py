from dataclasses import dataclass
from typing import Literal, ClassVar
from bems_simulation.core.components.remote.shared.config import (
    BaseRemoteComponentConfig,
)
from bems_simulation.core.components.shared.spaces import Space


@dataclass(frozen=True, slots=True, kw_only=True)
class HelicsConnectionConfig:
    """HELICS connection configuration."""

    federate_name: str
    pub_topic: str
    sub_topic: str
    broker_address: str = "127.0.0.1"
    core_type: str = "zmq"


@dataclass(frozen=True, slots=True, kw_only=True)
class HelicsComponentConfig(BaseRemoteComponentConfig):
    """HELICS remote component configuration."""

    action_space: dict[str, Space]
    connection: HelicsConnectionConfig

    type: Literal["helics"] = "helics"
