# todo: this module is where the model class should be defined; the logic for the solution search, etc
# todo: should be implemented in the `internal` package and imported

from typing import Union, List, Generator, Literal, Any
from .constraints import Constraint
from .callbacks import Callback
from .instructions import Instruction
from .exceptions import *


class Function:
    # A model is a function that is defined by its input and output domain/ranges, and its behaviour. These
    # aspects of a model are not 'mutable' in the sense that changing these means creating a new model, as
    # two models with a different description are not the same model.
    def __init__(self, behaviour: str, output_schema: dict, input_schema: dict, constraints: List[Constraint] = None):
        self.behaviour = behaviour
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.constraints = constraints or []

        # todo: this is obviously just an example
        if behaviour == "":
            raise InsufficientSpecificationError("The model's behaviour is not specified.")

    # Runs the solution search and model training to construct a model
    # Note: we do not consider the dataset to be a property of the model itself
    # Note: there would also be an async version of this method
    def build(
        self,
        dataset: Union[str, Generator] = None,
        instructions: List[Instruction] = None,  # these are "hints" that can be followed during training
        callbacks: List[Callback] = None,  # these are functions that are called during training
        isolation: Literal["local", "subprocess", "docker"] = "local",  # this controls the sandboxing of training
    ) -> None:
        pass

    # Returns the output of the model given an input
    # todo: consider whether we should also support __call__
    def call(self, x: Any) -> Any:
        pass

    # Returns an assessment of the model's definition, which can be helpful when first defining the model
    def review(self) -> str:
        pass

    # Returns a human-readable description of the model, its performance, etc
    def describe(self) -> str:
        pass

    # Returns a list of "instruction" objects that could be followed to further improve the model
    def suggest(self) -> List[Instruction]:
        pass
