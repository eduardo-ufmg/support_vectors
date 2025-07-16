import numpy as np

def get_interclass_vertices(X: np.ndarray, ADJ: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the vertices of edges that connect samples from different classes in a graph.

    Parameters
    ----------
    X : np.ndarray
        The input data points.
    ADJ : np.ndarray
        The adjacency matrix representing the graph.
    y : np.ndarray
        The labels corresponding to the data points.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - vertices: The vertices that connect different classes.
        - class_labels: The labels of the classes corresponding to the vertices.
        - degrees: The degrees of the vertices in the graph.

    The degree of a vertex is defined as the number of same-class edges connected to it divided by the total number of edges connected to it.
    """

    X = np.asarray(X)
    ADJ = np.asarray(ADJ)
    y = np.asarray(y)

    n = X.shape[0]
    degrees = -1 * np.ones(n, dtype=float) # Initialize degrees to -1 to indicate uncalculated
    vertices = []
    class_labels = []

    for i in range(n):
        neighbors = np.where(ADJ[i])[0]
        if len(neighbors) == 0:
            continue
        
        same_class_neighbors = neighbors[y[neighbors] == y[i]]
        different_class_neighbors = neighbors[y[neighbors] != y[i]]

        if len(different_class_neighbors) > 0:
            vertices.append(X[i])
            class_labels.append(y[i])
            degrees[i] = len(same_class_neighbors) / len(neighbors)

    degrees = degrees[degrees != -1]

    return np.array(vertices), np.array(class_labels), degrees