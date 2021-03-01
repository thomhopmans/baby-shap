from .._serializable import Serializable, Serializer, Deserializer


class Model(Serializable):
    """ This is the superclass of all models.
    """

    def __init__(self, model=None):
        """ Wrap a callable model as a SHAP Model object.
        """
        if isinstance(model, Model):
            self.model = model.model
        else:
            self.model = model

    def __call__(self, *args):
        return self.model(*args)

    def save(self, out_file):
        """ Save the model to the given file stream.
        """
        super().save(out_file)
        with Serializer(out_file, "shap.Model", version=0) as s:
            s.save("model", self.model)

    @classmethod
    def load(cls, in_file, instantiate=True):
        if instantiate:
            return cls._instantiated_load(in_file)

        kwargs = super().load(in_file, instantiate=False)
        with Deserializer(in_file, "shap.Model", min_version=0, max_version=0) as s:
            kwargs["model"] = s.load("model")
        return kwargs
