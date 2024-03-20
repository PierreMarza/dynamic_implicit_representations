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
from habitat.tasks.nav import _try_register_nav_task


def make_task(id_task, **kwargs):
    logger.info("Initializing task {}".format(id_task))
    _task = registry.get_task(id_task)
    assert _task is not None, "Could not find task with name {}".format(id_task)

    return _task(**kwargs)


_try_register_nav_task()
