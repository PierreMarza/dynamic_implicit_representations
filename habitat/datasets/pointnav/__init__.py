#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_pointnavdatasetv1():
    try:
        from habitat.datasets.pointnav.pointnav_dataset import (
            PointNavDatasetV1,
        )

        has_pointnav = True
    except ImportError as e:
        has_pointnav = False
        pointnav_import_error = e

    if has_pointnav:
        from habitat.datasets.pointnav.pointnav_dataset import (
            PointNavDatasetV1,
        )
    else:

        @registry.register_dataset(name="PointNav-v1")
        class PointnavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error
