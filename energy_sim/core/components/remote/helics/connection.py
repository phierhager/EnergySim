from __future__ import annotations

import json
from typing import Any, Dict, Optional

import helics as h
from bems_simulation.core.components.remote.helics.config import HelicsConnectionConfig
from bems_simulation.core.components.registry import register_connection


@register_connection(HelicsConnectionConfig)
class HelicsConnection:
    """Real HELICS interface using pyhelics."""

    def __init__(self, connection_config: HelicsConnectionConfig) -> None:
        self.connection_config = connection_config
        self.fed: Optional[Any] = None  # HELICS Federate object
        self.pub: Optional[Any] = None  # HELICS Publication object
        self.sub: Optional[Any] = None  # HELICS Subscription object

    def connect(self) -> None:
        fedinfo: Any = h.helicsCreateFederateInfo()

        h.helicsFederateInfoSetCoreName(fedinfo, self.connection_config.federate_name)
        h.helicsFederateInfoSetCoreTypeFromString(
            fedinfo, self.connection_config.core_type
        )
        h.helicsFederateInfoSetCoreInitString(
            fedinfo,
            f"--federates=1 --broker_address={self.connection_config.broker_address}",
        )
        h.helicsFederateInfoSetTimeProperty(fedinfo, h.helics_property_time_delta, 1.0)

        self.fed = h.helicsCreateCombinationFederate(
            self.connection_config.federate_name, fedinfo
        )

        # Register pub/sub
        self.pub = h.helicsFederateRegisterGlobalPublication(
            self.fed, self.connection_config.pub_topic, h.HELICS_DATA_TYPE_STRING, ""
        )
        self.sub = h.helicsFederateRegisterSubscription(
            self.fed, self.connection_config.sub_topic, ""
        )

        h.helicsFederateEnterExecutingMode(self.fed)

    def publish(self, data: Dict[str, Any]) -> None:
        if self.pub is None:
            raise RuntimeError("HELICS publication not initialized.")
        h.helicsPublicationPublishString(self.pub, json.dumps(data))

    def subscribe(self) -> Dict[str, Any]:
        if self.sub is None:
            raise RuntimeError("HELICS subscription not initialized.")
        if h.helicsInputIsUpdated(self.sub):
            msg: str = h.helicsInputGetString(self.sub)
            try:
                return json.loads(msg)
            except json.JSONDecodeError:
                return {}
        return {}

    def disconnect(self) -> None:
        if self.fed is not None:
            h.helicsFederateFinalize(self.fed)
            h.helicsFederateFree(self.fed)
            h.helicsCloseLibrary()
            self.fed = None
            self.pub = None
            self.sub = None
