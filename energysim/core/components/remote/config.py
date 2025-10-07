"""Config definitions for local components. This is necessary to make dacite able to infer the correct types."""

from typing import Union

from .helics.config import HelicsComponentConfig

RemoteComponentConfig = Union[HelicsComponentConfig]
