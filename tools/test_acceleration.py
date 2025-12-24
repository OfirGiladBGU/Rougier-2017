"""
Test script to validate acceleration options and compare performance.
"""
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import voronoi

def create_test_density(size=100):
    """Create a simple test density map (gradient)"""
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    density = (X + Y) / 2.0
    return density.astype(np.float32)

def test_acceleration():
    """Test all acceleration modes"""
    print("=" * 70)
    print("Weighted Voronoi Stippler - Acceleration Test")
    print("=" * 70)
    
    # Test parameters
    n_points = 500
    n_iters = 10
    size = 100
    
    print(f"\nTest Configuration:")
    print(f"  Points: {n_points}")
    print(f"  Iterations: {n_iters}")
    print(f"  Density map size: {size}x{size}")
    
    # Check availability
    print(f"\nAcceleration Availability:")
    print(f"  Numba (CPU JIT): {'✓ Available' if voronoi.NUMBA_AVAILABLE else '✗ Not available'}")
    print(f"  CUDA (GPU):      {'✓ Available' if voronoi.CUDA_AVAILABLE else '✗ Not available'}")
    
    if voronoi.CUDA_AVAILABLE:
        import torch
        print(f"    GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Create test data
    print("\nPreparing test data...")
    np.random.seed(42)
    points = np.random.rand(n_points, 2) * size
    density = create_test_density(size)
    density_P = density.cumsum(axis=1)
    density_Q = density_P.cumsum(axis=1)
    bbox = np.array([0, size, 0, size])
    
    # Test each acceleration mode
    modes = ['none']
    if voronoi.NUMBA_AVAILABLE:
        modes.append('numba')
    if voronoi.CUDA_AVAILABLE:
        modes.append('cuda')
    
    results = {}
    
    print("\n" + "=" * 70)
    print("Performance Benchmarks")
    print("=" * 70)
    
    for mode in modes:
        print(f"\nTesting --accelerator {mode}...")
        
        # Warm-up run (especially important for numba and cuda)
        if mode in ['numba', 'cuda']:
            print("  Warming up...")
            voronoi.centroids(points, density, bbox, density_P, density_Q, accelerator=mode)
        
        # Timed runs
        current_points = points.copy()
        start = time.time()
        
        for i in range(n_iters):
            regions, current_points = voronoi.centroids(
                current_points, density, bbox, density_P, density_Q, 
                accelerator=mode
            )
        
        elapsed = time.time() - start
        results[mode] = {
            'time': elapsed,
            'regions': len(regions),
            'final_points': current_points.copy()
        }
        
        print(f"  Time: {elapsed:.3f}s ({elapsed/n_iters:.3f}s per iteration)")
        print(f"  Regions: {len(regions)}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    
    baseline_time = results['none']['time']
    
    print(f"\n{'Mode':<15} {'Time (s)':<12} {'Speedup':<10} {'Status'}")
    print("-" * 70)
    
    for mode in modes:
        time_val = results[mode]['time']
        speedup = baseline_time / time_val
        status = "✓" if speedup >= 1.0 else "?"
        print(f"{mode:<15} {time_val:<12.3f} {speedup:<10.2f}x {status}")
    
    # Verify consistency
    print("\n" + "=" * 70)
    print("Numerical Consistency Check")
    print("=" * 70)
    
    baseline_points = results['none']['final_points']
    
    for mode in modes:
        if mode == 'none':
            continue
        
        points_diff = np.abs(results[mode]['final_points'] - baseline_points)
        max_diff = np.max(points_diff)
        mean_diff = np.mean(points_diff)
        
        print(f"\n{mode} vs baseline:")
        print(f"  Max difference:  {max_diff:.6f} pixels")
        print(f"  Mean difference: {mean_diff:.6f} pixels")
        
        if max_diff < 0.01:
            print(f"  Status: ✓ Excellent agreement")
        elif max_diff < 0.1:
            print(f"  Status: ✓ Good agreement")
        else:
            print(f"  Status: ⚠ Some differences detected")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    return results

if __name__ == '__main__':
    try:
        test_acceleration()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
