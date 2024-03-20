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
from habitat.datasets.multi_nav import _try_register_multinavdatasetv1
from habitat.datasets.object_nav import _try_register_objectnavdatasetv1
from habitat.datasets.pointnav import _try_register_pointnavdatasetv1


def make_dataset(id_dataset, **kwargs):
    logger.info("Initializing dataset {}".format(id_dataset))
    _dataset = registry.get_dataset(id_dataset)
    assert _dataset is not None, "Could not find dataset {}".format(id_dataset)

    return _dataset(**kwargs)


_try_register_multinavdatasetv1()
_try_register_objectnavdatasetv1()
_try_register_pointnavdatasetv1()
