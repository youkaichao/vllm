import dataclasses

class EfficientPickleDataclass:
    def __getstate__(self):
        # state is a tuple, sorted by field names.
        # value of default is replaced with None
        fields = dataclasses.fields(self)
        fields = sorted(fields, key=lambda f: f.name)
        state = []
        for field in fields:
            value = getattr(self, field.name)
            if value == field.default:
                value = None
            state.append(value)
        return state

    def __setstate__(self, state):
        fields = dataclasses.fields(self)
        fields = sorted(fields, key=lambda f: f.name)
        for field, value in zip(fields, state):
            if value is not None:
                setattr(self, field.name, value)
            else:
                setattr(self, field.name, field.default)
