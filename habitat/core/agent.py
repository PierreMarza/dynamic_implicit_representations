#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################
r"""Base implementation of agent inside habitat. To build agents inside habitat
the user should subclass ``habitat.Agent`` and implement the ``act()``
and ``reset()`` methods.
"""

from typing import Any, Dict, Union

from habitat.core.simulator import Observations


class Agent:
    r"""Abstract class for defining agents which act inside `core.env.Env`.

    This abstract class standardizes agents to allow seamless benchmarking.
    """

    def reset(self) -> None:
        r"""Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(self, observations: Observations) -> Union[int, str, Dict[str, Any]]:
        r"""Called to produce an action to perform in an environment.

        :param observations: observations coming in from environment to be
            used by agent to decide action.
        :return: action to be taken inside the environment and optional action
            arguments.
        """
        raise NotImplementedError
