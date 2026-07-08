from pathlib import Path


def validate_path(path_input):
    """
    Check if the input is a str or Path.

    Parameters
    ----------
    path_input : str or Path
        Input path

    Returns
    -------
    path_obj : Path
        Returns the input as a Path

    Raises
    ------
    TypeError
        If the input is not a str or Path, then this error is thrown.
    """
    if not isinstance(path_input, (str, Path)):
        raise TypeError(
            f"Expected input to be a str or Path. Instead it is a {type(path_input)}."
        )

    path_obj = Path(path_input)

    return path_obj
