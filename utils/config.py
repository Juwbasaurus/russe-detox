class Config:
    """Read-only dictionary-like object that replaces keys with attributes.
    """

    def __init__(self, dictionary):
        self.__data = dict(dictionary)

    def __getattr__(self, name):
        try:
            return self.__data[name]
        except KeyError:
            raise KeyError(f'No attribute {name}')
