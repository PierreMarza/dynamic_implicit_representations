#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from habitat.core.registry import registry
from habitat.core.simulator import Simulator


def _try_register_pyrobot():
    try:
        import pyrobot

        has_pyrobot = True
    except ImportError as e:
        has_pyrobot = False
        pyrobot_import_error = e

    if has_pyrobot:
        from habitat.sims.pyrobot.pyrobot import PyRobot
    else:

        @registry.register_simulator(name="PyRobot-v0")
        class PyRobotImportError(Simulator):
            def __init__(self, *args, **kwargs):
                raise pyrobot_import_error
