# todo: the concept of a constraint should be implemented here; I think a constraint should be thought of
# as a function that every input/output pair of the model must satisfy, so I would define a constraint as
# an object that has a method that takes an input/output pair and returns a boolean


# A constraint defines a condition that must be satisfied by the input/output behavior of a model.
class Constraint:
    def __init__(self, description: str):
        pass
