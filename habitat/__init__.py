#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from habitat.config import Config, get_config
from habitat.core.agent import Agent
from habitat.core.benchmark import Benchmark
from habitat.core.challenge import Challenge
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import EmbodiedTask, Measure, Measurements
from habitat.core.env import Env, RLEnv
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorSuite, SensorTypes, Simulator
from habitat.core.vector_env import ThreadedVectorEnv, VectorEnv
from habitat.datasets import make_dataset
from habitat.version import VERSION as __version__  # noqa

__all__ = [
    "Agent",
    "Benchmark",
    "Challenge",
    "Config",
    "Dataset",
    "EmbodiedTask",
    "Env",
    "get_config",
    "logger",
    "make_dataset",
    "Measure",
    "Measurements",
    "RLEnv",
    "Sensor",
    "SensorSuite",
    "SensorTypes",
    "Simulator",
    "ThreadedVectorEnv",
    "VectorEnv",
]
