class StrEnum:
    """
    A custom enumeration class for string values.
    """

    def __init__(self, *args):
        self._valid_strings = set()
        for arg in args:
            if arg in self._valid_strings:
                raise ValueError(f"Duplicate value found: {arg}")
            setattr(self, arg, arg)
            self._valid_strings.add(arg)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            raise AttributeError("Cannot reassign values in StrEnum")
        super().__setattr__(key, value)

    def __iter__(self):
        return iter(self._valid_strings)

    def __repr__(self):
        return f"<StrEnum {', '.join(self._valid_strings)}>"

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(self._valid_strings)})"
