# smolmodels/models.py
import random
from typing import Union, List, Generator, Literal, Any

from smolmodels.callbacks import Callback
from smolmodels.constraints import Constraint
from smolmodels.directives import Directive
from smolmodels.internal.solution_generation.generator import generate


class Model:
    """
    Represents a model that transforms inputs to outputs according to a specified behaviour.

    A `Model` is defined by a human-readable description of its expected behaviour, as well as structured
    definitions of its input schema, output schema, and any constraints that must be satisfied by the model.

    Attributes:
        behaviour (str): A human-readable, natural language description of the model's expected behaviour.
        output_schema (dict): A mapping of output key names to their types.
        input_schema (dict): A mapping of input key names to their types.
        constraints (List[Constraint]): A list of Constraint objects that represent rules which must be
            satisfied by every input/output pair for the model.

    Example:
        model = Model(
            behaviour="Given a dataset of house features, predict the house price.",
            output_schema={"price": float},
            input_schema={
                "bedrooms": int,
                "bathrooms": int,
                "square_footage": float,
                "lot_size": float,
                "year_built": int,
                "location": str,
            }
        )
    """

    def __init__(self, behaviour: str, output_schema: dict, input_schema: dict, constraints: List[Constraint] = None):
        """
        Initialise a model with a natural language description of its behaviour, as well as
        structured definitions of its input schema, output schema, and any constraints.

        :param [str] behaviour: A human-readable, natural language description of the model's expected behaviour.
        :param [dict] output_schema: A mapping of output key names to their types.
        :param [dict] input_schema: A mapping of input key names to their types.
        :param List[Constraint] constraints: A list of Constraint objects that represent rules which must be
            satisfied by every input/output pair for the model.
        """
        # todo: analyse natural language inputs and raise errors where applicable

        # The model's identity is defined by these fields
        self.behaviour = behaviour
        self.output_schema = output_schema
        self.input_schema = input_schema
        self.constraints = constraints or []

        # The model's mutable state is defined by these fields
        # todo: this is WIP, trying to flesh out what the model's internal state might look like
        self.state = "draft"  # todo: this probably needs to be represented by a class, maybe an Enum
        self.trainer = None  # todo: this is the object that was used to train the model, but does it need to exist?
        self.predictor = None  # todo: this is an object that loads the model and makes predictions
        self.metrics = dict()  # todo: this is a dictionary of metrics that the model has achieved
        self.metadata = dict()  # todo: this is a dictionary of metadata about the model

    def build(
        self,
        dataset: Union[str, Generator],
        directives: List[Directive] = None,
        callbacks: List[Callback] = None,
        isolation: Literal["local", "subprocess", "docker"] = "local",
    ) -> None:
        # todo: implement properly, this is a placeholder
        generate(
            self.behaviour,
            self.input_schema,
            self.output_schema,
            self.constraints,
            dataset,
            directives,
            callbacks,
            isolation,
        )

    def predict(self, x: Any) -> Any:
        """
        Call the model with input x and return the output.
        :param x: input to the model
        :return: output of the model
        """
        # todo: this is a placeholder, implement the actual model prediction logic
        return random.randint(0, max(len(self.behaviour), x))

    def __call__(self, x: Any) -> Any:
        """
        Call the model with input x and return the output.
        :param x: input to the model
        :return: output of the model
        """
        return self.predict(x)
