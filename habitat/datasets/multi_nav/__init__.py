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


def _try_register_multinavdatasetv1():
    try:
        from habitat.datasets.multi_nav.multi_nav_dataset import (
            MultiNavDatasetV1,
        )

        has_pointnav = True
    except ImportError as e:
        has_pointnav = False
        pointnav_import_error = e

    if has_pointnav:
        from habitat.datasets.multi_nav.multi_nav_dataset import (
            MultiNavDatasetV1,
        )
    else:

        @registry.register_dataset(name="MultiNav-v1")
        class MultiNavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise pointnav_import_error
