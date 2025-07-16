import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from support_vectors import support_vectors

def test_support_vectors():
    """
    Test all combinations of graph methods and support vector selectors.
    Plot all results in the same figure with the complete set and highlighted support vectors.
    """
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=200, n_features=2, cluster_std=2)[0:2]
    
    # Define test parameters
    graph_methods = ['gabriel', 'relative_neighborhood', 'urquhart']
    filter_methods = ['two-pass', 'one-pass', 'one-pass']
    one_step_criteria = [None, 'interclass-average', 'zero']
    
    # Create subplot layout
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Support Vector Selection: All Methods Comparison')
    
    # Test all combinations
    for i, graph_method in enumerate(graph_methods):
        for j, (filter_method, one_step_criterion) in enumerate(zip(filter_methods, one_step_criteria)):
            ax = axes[i, j]

            title = ""
            
            try:
                # Get support vectors
                if filter_method == 'two-pass':
                    X_support, y_support = support_vectors(
                        X, y, graph_method, filter_method, ''
                    )
                    title = f'{graph_method.title()}\nTwo-Pass'
                else:
                    X_support, y_support = support_vectors(
                        X, y, graph_method, filter_method, one_step_criterion
                    )
                    title = f'{graph_method.title()}\nOne-Pass ({one_step_criterion})'
                
                # Plot complete dataset
                scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
                
                # Highlight support vectors
                if len(X_support) > 0:
                    ax.scatter(X_support[:, 0], X_support[:, 1], 
                             c=y_support, marker='x', s=100)
                
                ax.set_title(title)
                
                print(f"{title.replace(chr(10), ' ')}: {len(X_support)} support vectors")
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)}', 
                       transform=ax.transAxes, ha='center', va='center',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
                ax.set_title(title)
                print(f"{title.replace(chr(10), ' ')}: Error - {str(e)}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    for i, graph_method in enumerate(graph_methods):
        print(f"\n{graph_method.upper()} GRAPH:")
        print("-" * 30)
        
        for j, (filter_method, one_step_criterion) in enumerate(zip(filter_methods, one_step_criteria)):
            try:
                if filter_method == 'two-pass':
                    X_support, y_support = support_vectors(
                        X, y, graph_method, filter_method, 'interclass-average'
                    )
                    method_name = "Two-Pass"
                else:
                    X_support, y_support = support_vectors(
                        X, y, graph_method, filter_method, one_step_criterion
                    )
                    method_name = f"One-Pass ({one_step_criterion})"
                
                reduction_rate = (1 - len(X_support) / len(X)) * 100
                print(f"  {method_name:25}: {len(X_support):3d} vectors ({reduction_rate:5.1f}% reduction)")
                
            except Exception as e:
                print(f"  {method_name:25}: ERROR - {str(e)}")

def test_individual_methods():
    """
    Test individual methods to ensure they work correctly.
    """
    print("\n" + "="*50)
    print("INDIVIDUAL METHOD VALIDATION")
    print("="*50)
    
    # Generate simple test data
    X_simple, y_simple = make_blobs(n_samples=50, n_features=2, cluster_std=1.0)[0:2]
    
    # Test each graph method individually
    from gabriel_graph.gabriel_graph import gabriel_graph
    from relative_neighborhood_graph.relative_neighborhood_graph import relative_neighborhood_graph
    from urquhart_graph.urquhart_graph import urquhart_graph
    
    print("\nTesting individual graph methods:")
    
    try:
        adj_gabriel = gabriel_graph(X_simple)
        print(f"  Gabriel graph: {adj_gabriel.sum()} edges")
    except Exception as e:
        print(f"  Gabriel graph: ERROR - {str(e)}")
    
    try:
        adj_rng = relative_neighborhood_graph(X_simple)
        print(f"  RNG graph: {adj_rng.sum()} edges")
    except Exception as e:
        print(f"  RNG graph: ERROR - {str(e)}")
    
    try:
        adj_urquhart = urquhart_graph(X_simple)
        print(f"  Urquhart graph: {adj_urquhart.sum()} edges")
    except Exception as e:
        print(f"  Urquhart graph: ERROR - {str(e)}")
    
    # Test filtering functions
    from filtering.get_interclass_vertices import get_interclass_vertices
    from filtering.filter_by_degree import filter_by_degree

    print("\nTesting interclass vertex selection:")

    try:
        vertices, labels, degrees = get_interclass_vertices(X_simple, adj_gabriel, y_simple)
        print(f"  Interclass vertices: {len(vertices)} found")
    except Exception as e:
        print(f"  Interclass vertex selection: ERROR - {str(e)}")
    
    print("\nTesting filtering functions:")

    try:
        X_filtered, y_filtered = filter_by_degree(vertices, labels, degrees, 'class-average')
        print(f"  After class-average degree filtering: {len(X_filtered)} remaining")
    except Exception as e:
        print(f"  Class Average filtering: ERROR - {str(e)}")

    try:
        X_filtered, y_filtered = filter_by_degree(vertices, labels, degrees, 'interclass-average')
        print(f"  After interclass-average degree filtering: {len(X_filtered)} remaining")
    except Exception as e:
        print(f"  Interclass Average filtering: ERROR - {str(e)}")

    try:
        X_filtered, y_filtered = filter_by_degree(vertices, labels, degrees, 'zero')
        print(f"  After zero degree filtering: {len(X_filtered)} remaining")
    except Exception as e:
        print(f"  Zero degree filtering: ERROR - {str(e)}")

if __name__ == "__main__":
    print("Starting comprehensive support vector validation...")
    test_support_vectors()
    test_individual_methods()
    print("\nValidation complete!")