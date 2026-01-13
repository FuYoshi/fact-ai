"""
Quick unit tests for FCGrad implementation.
Run with: python test_fcgrad.py
"""
import sys
sys.path.insert(0, '/home/scur0012/simon/fact_fcgrad/external/SocialJax/algorithms/IPPO')

import jax.numpy as jnp
import numpy as np
from fcgrad_utils import project_gradient, fcgrad_adjust, flatten_grads, unflatten_grads

def test_projection():
    """Test that projection onto normal plane works correctly."""
    # g perpendicular to n should remain unchanged
    g = jnp.array([1.0, 0.0])
    n = jnp.array([0.0, 1.0])
    proj = project_gradient(g, n)
    assert jnp.allclose(proj, g), f"Perpendicular case failed: {proj}"
    
    # g parallel to n should become zero
    g = jnp.array([1.0, 0.0])
    n = jnp.array([1.0, 0.0])
    proj = project_gradient(g, n)
    assert jnp.allclose(proj, jnp.zeros(2), atol=1e-6), f"Parallel case failed: {proj}"
    
    # General case: project [1,1] onto plane normal to [1,0] -> should be [0,1]
    g = jnp.array([1.0, 1.0])
    n = jnp.array([1.0, 0.0])
    proj = project_gradient(g, n)
    assert jnp.allclose(proj, jnp.array([0.0, 1.0]), atol=1e-6), f"General case failed: {proj}"
    
    print("✅ Projection tests passed")


def test_conflict_detection():
    """Test that conflicts are detected correctly."""
    # Conflicting gradients (dot < 0)
    g_ind = {'w': jnp.array([1.0, 0.0])}
    g_col = {'w': jnp.array([-1.0, 0.0])}
    
    flat_ind = flatten_grads(g_ind)
    flat_col = flatten_grads(g_col)
    dot = jnp.dot(flat_ind, flat_col)
    assert dot < 0, f"Should detect conflict, got dot={dot}"
    
    # Non-conflicting gradients (dot >= 0)
    g_ind = {'w': jnp.array([1.0, 0.0])}
    g_col = {'w': jnp.array([1.0, 1.0])}
    
    flat_ind = flatten_grads(g_ind)
    flat_col = flatten_grads(g_col)
    dot = jnp.dot(flat_ind, flat_col)
    assert dot >= 0, f"Should not detect conflict, got dot={dot}"
    
    print("✅ Conflict detection tests passed")


def test_prioritization():
    """Test that the correct objective is prioritized based on return value."""
    # Create conflicting gradients
    g_ind = {'w': jnp.array([1.0, 0.0])}
    g_col = {'w': jnp.array([-0.5, 0.5])}
    
    # Case 1: Individual value < collective value (individual disadvantaged)
    # Should project INDIVIDUAL onto COLLECTIVE's normal plane
    result = fcgrad_adjust(g_ind, g_col, value_individual=0.5, value_collective=1.0)
    
    flat_result = flatten_grads(result)
    flat_ind = flatten_grads(g_ind)
    flat_col = flatten_grads(g_col)
    
    # Result should have non-negative dot with collective (no longer conflicts)
    dot_with_col = jnp.dot(flat_result, flat_col)
    assert dot_with_col >= -1e-6, f"Projected grad should not conflict with collective: {dot_with_col}"
    print(f"  Case 1 (ind disadvantaged, lower value): result·col = {dot_with_col:.4f} ✓")
    
    # Case 2: Collective value < individual value (collective disadvantaged)
    # Should project COLLECTIVE onto INDIVIDUAL's normal plane
    result = fcgrad_adjust(g_ind, g_col, value_individual=1.0, value_collective=0.5)
    
    flat_result = flatten_grads(result)
    
    # Result should have non-negative dot with individual (no longer conflicts)
    dot_with_ind = jnp.dot(flat_result, flat_ind)
    assert dot_with_ind >= -1e-6, f"Projected grad should not conflict with individual: {dot_with_ind}"
    print(f"  Case 2 (col disadvantaged, lower value): result·ind = {dot_with_ind:.4f} ✓")
    
    print("✅ Prioritization tests passed")


def test_no_conflict_passthrough():
    """Test that non-conflicting gradients pass through unchanged."""
    g_ind = {'w': jnp.array([1.0, 1.0])}
    g_col = {'w': jnp.array([0.5, 0.5])}
    
    result = fcgrad_adjust(g_ind, g_col, value_individual=1.0, value_collective=0.5)
    
    # Should return individual gradient unchanged
    assert jnp.allclose(result['w'], g_ind['w']), "No-conflict case should return individual gradient"
    
    print("✅ No-conflict passthrough test passed")


def test_nested_pytree():
    """Test with nested parameter structure like real neural networks."""
    g_ind = {
        'params': {
            'Dense_0': {'kernel': jnp.array([[1.0, -1.0], [0.5, 0.5]]), 'bias': jnp.array([0.1, -0.1])},
            'Dense_1': {'kernel': jnp.array([[0.2]]), 'bias': jnp.array([0.3])}
        }
    }
    g_col = {
        'params': {
            'Dense_0': {'kernel': jnp.array([[-0.5, 0.5], [-0.2, 0.2]]), 'bias': jnp.array([-0.05, 0.05])},
            'Dense_1': {'kernel': jnp.array([[0.1]]), 'bias': jnp.array([0.15])}
        }
    }
    
    # Just check it doesn't crash and returns correct structure
    result = fcgrad_adjust(g_ind, g_col, value_individual=1.0, value_collective=2.0)
    
    assert 'params' in result
    assert 'Dense_0' in result['params']
    assert 'kernel' in result['params']['Dense_0']
    assert result['params']['Dense_0']['kernel'].shape == (2, 2)
    
    print("✅ Nested pytree test passed")


def test_numerical_example():
    """Concrete numerical example matching paper's description."""
    print("\n📊 Numerical Example (paper's logic: lower return = disadvantaged):")
    
    # Scenario: Agent doing worse (cleaning waste), collective doing better
    g_ind = {'w': jnp.array([2.0, 1.0])}   # Individual wants to go this direction
    g_col = {'w': jnp.array([-1.0, 1.0])}  # Collective wants opposite in x
    
    print(f"  g_individual = {g_ind['w']}")
    print(f"  g_collective = {g_col['w']}")
    print(f"  dot product  = {jnp.dot(g_ind['w'], g_col['w']):.2f} (CONFLICT)")
    
    # Individual is disadvantaged (lower return)
    result = fcgrad_adjust(g_ind, g_col, value_individual=1.0, value_collective=2.0)
    print(f"\n  Individual disadvantaged (value_ind=1.0 < value_col=2.0):")
    print(f"  → Result = {result['w']} (individual projected onto collective's normal)")
    
    # Collective is disadvantaged (lower return)  
    result = fcgrad_adjust(g_ind, g_col, value_individual=2.0, value_collective=1.0)
    print(f"\n  Collective disadvantaged (value_col=1.0 < value_ind=2.0):")
    print(f"  → Result = {result['w']} (collective projected onto individual's normal)")
    
    print("\n✅ Numerical example completed")


if __name__ == "__main__":
    print("=" * 50)
    print("FCGrad Unit Tests")
    print("=" * 50 + "\n")
    
    test_projection()
    test_conflict_detection()
    test_prioritization()
    test_no_conflict_passthrough()
    test_nested_pytree()
    test_numerical_example()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
