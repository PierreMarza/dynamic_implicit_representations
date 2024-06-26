#!/usr/bin/env python3

###############################################################################
# Base code from:                                                             #
# * https://github.com/facebookresearch/habitat-lab                           #
# * https://github.com/saimwani/multiON                                       #
# * https://github.com/PierreMarza/teaching_agents_how_to_map                 #
#                                                                             #
# Adapted by Pierre Marza (pierre.marza@insa-lyon.fr)                         #
###############################################################################

r"""Registry is central source of truth in Habitat.

Taken from Pythia, it is inspired from Redux's concept of global store.
Registry maintains mappings of various information to unique keys. Special
functions in registry can be used as decorators to register different kind of
classes.

Import the global registry object using

.. code:: py

    from habitat.core.registry import registry

Various decorators for registry different kind of classes with unique keys

-   Register a task: ``@registry.register_task``
-   Register a task action: ``@registry.register_task_action``
-   Register a simulator: ``@registry.register_simulator``
-   Register a sensor: ``@registry.register_sensor``
-   Register a measure: ``@registry.register_measure``
-   Register a dataset: ``@registry.register_dataset``
"""

import collections
from typing import Optional

from habitat.core.utils import Singleton


class Registry(metaclass=Singleton):
    mapping = collections.defaultdict(dict)

    @classmethod
    def _register_impl(cls, _type, to_register, name, assert_type=None):
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_task(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a task to registry with key :p:`name`

        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat.core.registry import registry
            from habitat.core.embodied_task import EmbodiedTask

            @registry.register_task
            class MyTask(EmbodiedTask):
                pass


            # or

            @registry.register_task(name="MyTaskName")
            class MyTask(EmbodiedTask):
                pass

        """
        from habitat.core.embodied_task import EmbodiedTask

        return cls._register_impl("task", to_register, name, assert_type=EmbodiedTask)

    @classmethod
    def register_simulator(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a simulator to registry with key :p:`name`

        :param name: Key with which the simulator will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat.core.registry import registry
            from habitat.core.simulator import Simulator

            @registry.register_simulator
            class MySimulator(Simulator):
                pass


            # or

            @registry.register_simulator(name="MySimName")
            class MySimulator(Simulator):
                pass

        """
        from habitat.core.simulator import Simulator

        return cls._register_impl("sim", to_register, name, assert_type=Simulator)

    @classmethod
    def register_sensor(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a sensor to registry with key :p:`name`

        :param name: Key with which the sensor will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.simulator import Sensor

        return cls._register_impl("sensor", to_register, name, assert_type=Sensor)

    @classmethod
    def register_measure(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a measure to registry with key :p:`name`

        :param name: Key with which the measure will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.embodied_task import Measure

        return cls._register_impl("measure", to_register, name, assert_type=Measure)

    @classmethod
    def register_task_action(cls, to_register=None, *, name: Optional[str] = None):
        r"""Add a task action in this registry under key 'name'

        :param action_space: An action space that describes parameters to the
            task action's method. If :py:`None` then the task action's method
            takes no parameters.
        :param name: Key with which the task action will be registered. If
            :py:`None` will use the name of the task action's method.
        """
        from habitat.core.embodied_task import Action

        return cls._register_impl("task_action", to_register, name, assert_type=Action)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a dataset to registry with key :p:`name`

        :param name: Key with which the dataset will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.dataset import Dataset

        return cls._register_impl("dataset", to_register, name, assert_type=Dataset)

    @classmethod
    def register_action_space_configuration(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a action space configuration to registry with key :p:`name`

        :param name: Key with which the action space will be registered.
            If :py:`None` will use the name of the class
        """
        from habitat.core.simulator import ActionSpaceConfiguration

        return cls._register_impl(
            "action_space_config",
            to_register,
            name,
            assert_type=ActionSpaceConfiguration,
        )

    @classmethod
    def _get_impl(cls, _type, name):
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_task(cls, name):
        return cls._get_impl("task", name)

    @classmethod
    def get_task_action(cls, name):
        return cls._get_impl("task_action", name)

    @classmethod
    def get_simulator(cls, name):
        return cls._get_impl("sim", name)

    @classmethod
    def get_sensor(cls, name):
        return cls._get_impl("sensor", name)

    @classmethod
    def get_measure(cls, name):
        return cls._get_impl("measure", name)

    @classmethod
    def get_dataset(cls, name):
        return cls._get_impl("dataset", name)

    @classmethod
    def get_action_space_configuration(cls, name):
        return cls._get_impl("action_space_config", name)


registry = Registry()
