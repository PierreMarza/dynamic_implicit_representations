#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

from habitat.core.embodied_task import EmbodiedTask
from habitat.core.registry import registry


def _try_register_nav_task():
    try:
        from habitat.tasks.nav.nav import NavigationTask

        has_navtask = True
    except ImportError as e:
        has_navtask = False
        navtask_import_error = e

    if has_navtask:
        from habitat.tasks.nav.nav import NavigationTask
    else:

        @registry.register_task(name="Nav-v0")
        class NavigationTaskImportError(EmbodiedTask):
            def __init__(self, *args, **kwargs):
                raise navtask_import_error
