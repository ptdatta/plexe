# todo: here we define the errors that can be thrown by the smolmodels package
# users can then do 'from smolmodels.errors import XyzError' to catch these errors


# This is an example of an error that can be thrown if the natural language behaviour
# specification is not sufficient to define the model
# There may be other such errors that may be thrown
class InsufficientSpecificationError(RuntimeError):
    pass
