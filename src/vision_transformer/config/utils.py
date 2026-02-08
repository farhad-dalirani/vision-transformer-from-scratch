from dataclasses import fields, is_dataclass


def dataclass_from_dict(cls, data: dict):
    """Recursively construct a dataclass instance from a dictionary.

    This function rebuilds a dataclass and its nested dataclass fields from
    a dictionary representation, such as one produced by
    ``dataclasses.asdict``. Fields whose types are themselves dataclasses
    are reconstructed recursively.

    Args:
        cls: Dataclass type to instantiate.
        data: Dictionary containing field values for the dataclass.

    Returns:
        An instance of ``cls`` populated with values from ``data``.

    Raises:
        TypeError: If ``cls`` is not a dataclass type.
    """
    kwargs = {}
    for f in fields(cls):
        value = data.get(f.name)
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = dataclass_from_dict(f.type, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
