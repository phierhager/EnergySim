"""Example HELICS component demonstrating remote component pattern."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict, Union
from abc import ABC, abstractmethod

from bems_simulation.core.components.shared.spaces import Space
from bems_simulation.core.components.shared.component_base import (
    ComponentBase,
)
from bems_simulation.core.components.remote.helics.config import HelicsComponentConfig
from bems_simulation.core.components.remote.helics.connection import HelicsConnection
from bems_simulation.core.components.shared.component_outputs import (
    ComponentOutputs,
)
from dacite import from_dict, Config as DaciteConfig
from bems_simulation.core.components.registry import register_remote_component
from bems_simulation.core.components.remote.shared.component import RemoteComponent

import logging

from bems_simulation.core.utils.converter import numpy_to_python

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@register_remote_component(HelicsComponentConfig)
class HelicsComponent(RemoteComponent):
    """Remote component that communicates via HELICS."""

    def __init__(
        self,
        connection: HelicsConnection,
        action_space: Dict[str, Space],
    ):
        # Not very elegant as no Dependency Injection
        self.helics_connection = connection
        self._initialized = False
        self._action_space = action_space

    def initialize(self) -> ComponentOutputs:
        """Initialize HELICS connection."""

        if not self._initialized:
            logger.debug(
                "Initializing HELICS component with interface: %s",
                self.helics_connection,
            )
            self.helics_connection.connect()
            self._initialized = True
            logger.info("HELICS component initialized and connected.")

        return ComponentOutputs()

    def advance(self, input: dict, dt_seconds: float) -> ComponentOutputs:
        """Send inputs to HELICS and receive outputs."""
        if not self._initialized:
            raise RuntimeError("Component must be initialized before advance.")

        logger.debug("Advance called with input=%s, dt=%s", input, dt_seconds)
        serialized_input = numpy_to_python(input)

        # Send control inputs via HELICS
        logger.debug("Publishing control data to HELICS: %s", serialized_input)
        self.helics_connection.publish(serialized_input)
        # TODO! Blocker!
        # Receive state updates
        remote_data = self.helics_connection.subscribe()
        logger.debug("Received remote data from HELICS: %s", remote_data)

        if not isinstance(remote_data, dict):
            logger.error(
                "Expected dict from HELICS subscribe, got %s", type(remote_data)
            )
            raise TypeError("HELICS subscribe must return dict.")

        return from_dict(
            data_class=ComponentOutputs,
            data=remote_data,
            config=DaciteConfig(cast=[Enum]),
        )

    def cleanup(self):
        """Clean up HELICS connection."""
        if self._initialized:
            logger.info("Disconnecting HELICS component.")
            self.helics_connection.disconnect()
            self._initialized = False
            logger.debug("HELICS component disconnected.")

    @property
    def action_space(self) -> dict[str, Space]:
        """Return action configurations."""
        return self._action_space
