import numpy as np


def filter_by_degree(
    X: np.ndarray, y: np.ndarray, degrees: np.ndarray, filter: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter vertices based on their degree in the graph.

    The threshold is determined by the filter criterion:

    - 'class-average': Below the average degree of their class.
    - 'interclass-average': Below the average degree of all vertices
        of the same class that are connected to interclass edges.
    - 'zero': Close to zero.

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
    Drop vertices with degree below the average degree of the vertices
    of the same class that are connected to interclass edges.

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

    # For each class, calculate average degree of vertices connected to interclass edges
    for class_label in unique_classes:
        class_mask = y == class_label
        class_degrees = degrees[class_mask]
        
        # Find vertices connected to interclass edges (degree < 1.0)
        interclass_mask = class_degrees < 1.0
        
        if np.any(interclass_mask):
            # Calculate average degree of vertices with interclass connections
            interclass_avg_degree = np.mean(class_degrees[interclass_mask])
            
            # Keep vertices with degree >= interclass average
            mask[class_mask] = class_degrees >= interclass_avg_degree
        else:
            # If no interclass connections, keep all vertices of this class
            mask[class_mask] = True

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
