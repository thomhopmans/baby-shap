class DimensionError(Exception):
    """
    Used for instances where dimensions are either
    not supported or cause errors.
    """

    pass


class InvalidFeaturePerturbationError(ValueError):
    pass


class InvalidModelError(ValueError):
    pass


class InvalidAlgorithmError(ValueError):
    pass


class InvalidClusteringError(ValueError):
    pass
