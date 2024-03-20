#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

import os

from habitat.core.benchmark import Benchmark
from habitat.core.logging import logger


class Challenge(Benchmark):
    def __init__(self):
        config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
        super().__init__(config_paths)

    def submit(self, agent):
        metrics = super().evaluate(agent)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))
