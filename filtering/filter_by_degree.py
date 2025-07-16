import numpy as np


def filter_by_degree(
    X: np.ndarray, y: np.ndarray, degrees: np.ndarray, filter: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter vertices based on their degree in the graph.

    The threshold is determined by the filter criterion:

    - 'class-average': Filter vertices with degree below the average degree of their class.
    - 'interclass-average': Filter vertices with degree below the average degree of all vertices of interclass edges.
    - 'zero': Filter vertices with degree close to zero.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    y : np.ndarray
        The labels corresponding to the data points.
    degrees : np.ndarray
        The degrees of the vertices in the graph.
    filter : str
        The filter criterion to apply ('class-average', 'interclass-average', 'zero').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The filtered data points and their corresponding labels.
    """

    if filter == "class-average":
        return class_average_filter(X, y, degrees)
    elif filter == "interclass-average":
        return interclass_average_filter(X, y, degrees)
    elif filter == "zero":
        return zero_degree_filter(X, y, degrees)
    else:
        raise ValueError(f"Unknown filter method: {filter}")


def class_average_filter(
    X: np.ndarray, y: np.ndarray, degrees: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter vertices based on the average degree of their class.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    y : np.ndarray
        The labels corresponding to the data points.
    degrees : np.ndarray
        The degrees of the vertices in the graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The filtered data points and their corresponding labels.
    """
    mask = np.zeros(len(X), dtype=bool)

    # Get unique classes
    unique_classes = np.unique(y)

    # For each class, calculate average degree and filter
    for class_label in unique_classes:
        class_mask = y == class_label
        class_degrees = degrees[class_mask]
        class_avg_degree = np.mean(class_degrees)

        # Keep vertices with degree >= class average
        mask[class_mask] = class_degrees >= class_avg_degree

    return X[mask], y[mask]


def interclass_average_filter(
    X: np.ndarray, y: np.ndarray, degrees: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter vertices based on the average degree of interclass edges.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    y : np.ndarray
        The labels corresponding to the data points.
    degrees : np.ndarray
        The degrees of the vertices in the graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The filtered data points and their corresponding labels.
    """
    mask = degrees >= np.mean(degrees)

    return X[mask], y[mask]


def zero_degree_filter(
    X: np.ndarray, y: np.ndarray, degrees: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter vertices with degree close to zero.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    y : np.ndarray
        The labels corresponding to the data points.
    degrees : np.ndarray
        The degrees of the vertices in the graph.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The filtered data points and their corresponding labels.
    """
    mask = degrees > 1e-6

    return X[mask], y[mask]
