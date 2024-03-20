#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.sims.habitat_simulator import _try_register_habitat_sim
from habitat.sims.pyrobot import _try_register_pyrobot


def make_sim(id_sim, **kwargs):
    logger.info("initializing sim {}".format(id_sim))
    _sim = registry.get_simulator(id_sim)
    assert _sim is not None, "Could not find simulator with name {}".format(id_sim)
    return _sim(**kwargs)


_try_register_habitat_sim()
_try_register_pyrobot()
