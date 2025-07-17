import numpy as np
from numpy.typing import ArrayLike

from filtering.filter_by_degree import filter_by_degree
from filtering.get_interclass_vertices import get_interclass_vertices
from gabriel_graph.gabriel_graph import gabriel_graph
from relative_neighborhood_graph.relative_neighborhood_graph import \
    relative_neighborhood_graph
from urquhart_graph.urquhart_graph import urquhart_graph


def support_vectors(
    X: ArrayLike,
    y: ArrayLike,
    graph_method: str,
    filter_method: str,
    one_step_filter_criterion: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the support vectors based on the specified graph method and filter method.

    Parameters
    ----------
    X : ArrayLike
        The input data points.
    y : ArrayLike
        The labels corresponding to the data points.
    graph_method : str
        The method to compute the graph
        ('gabriel', 'relative_neighborhood', or 'urquhart').
    filter_method : str
        The method to filter the support vectors ('two-pass', 'one-pass')
    one_step_filter_criterion : str
        The criterion for one-step filtering ('interclass-average', or 'zero').


    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The support vectors and their corresponding labels.

    Raises
    ------
    ValueError
        If an unknown graph method or filter method is provided;
        If the one-step filter criterion is not recognized;
        If the support vectors do not cover all classes in the original data.
    """

    method_dict = {
        "gabriel": gabriel_graph,
        "relative_neighborhood": relative_neighborhood_graph,
        "urquhart": urquhart_graph,
    }

    if graph_method in method_dict:
        build_graph = method_dict[graph_method]
    else:
        raise ValueError(f"Unknown graph method: {graph_method}")

    X = np.asarray(X)
    y = np.asarray(y)

    ADJ = build_graph(X)

    Xinter, yinter, degreeinter = get_interclass_vertices(X, ADJ, y)

    if filter_method == "two-pass":

        Xfiltered, yfiltered = filter_by_degree(
            Xinter, yinter, degreeinter, "class-average"
        )

        ADJfiltered = build_graph(Xfiltered)

        Xsupport, ysupport, _ = get_interclass_vertices(
            Xfiltered, ADJfiltered, yfiltered
        )

    elif filter_method == "one-pass":
        if one_step_filter_criterion not in ["interclass-average", "zero"]:
            raise ValueError(
                f"Unknown one-step filter criterion: {one_step_filter_criterion}"
            )

        Xsupport, ysupport = filter_by_degree(
            Xinter, yinter, degreeinter, one_step_filter_criterion
        )

    else:
        raise ValueError(f"Unknown filter method: {filter_method}")

    if len(np.unique(y)) != len(np.unique(ysupport)):
        raise ValueError(
            "The support vectors do not cover all classes in the original data."
        )

    return Xsupport, ysupport
