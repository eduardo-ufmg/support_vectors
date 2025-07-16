import numpy as np

from numpy.typing import ArrayLike

from gabriel_graph.gabriel_graph import gabriel_graph
from relative_neighborhood_graph.relative_neighborhood_graph import relative_neighborhood_graph
from urquhart_graph.urquhart_graph import urquhart_graph

def support_vectors(X: ArrayLike, y: ArrayLike,
                    graph_method: str, filter_method: str)-> tuple[np.ndarray, np.ndarray]:
    """
    Compute the support vectors based on the specified graph method and filter method.

    Parameters
    ----------
    X : ArrayLike
        The input data points.
    y : ArrayLike
        The labels corresponding to the data points.
    graph_method : str
        The method to compute the graph ('gabriel', 'relative_neighborhood', or 'urquhart').
    filter_method : str
        The method to filter the support vectors ('two-step', 'one-step')

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The support vectors and their corresponding labels.
    """

    raise NotImplementedError("This function is not implemented yet.")

