# Dask Configuration Optimization Guide

This guide helps you determine optimal values for `n_workers`, `threads_per_worker`, and `memory_limit` in your MIMIC-IV analysis application.

## Quick Start: System Resource Detection

### 1. Check Your System Resources

```bash
# Check CPU cores
sysctl -n hw.ncpu  # macOS
nproc              # Linux

# Check total memory
sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}'  # macOS
free -h            # Linux

# Check available memory
vm_stat | grep "Pages free" | awk '{print $3*4096/1024/1024/1024 " GB"}'  # macOS
```

### 2. Python Script for Automatic Detection

```python
import psutil
import os

def get_optimal_dask_config():
    """Get recommended Dask configuration based on system resources."""
    
    # System resources
    cpu_count = os.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    print(f"System Resources:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Total Memory: {total_memory_gb:.1f} GB")
    print(f"  Available Memory: {available_memory_gb:.1f} GB")
    
    # Recommendations
    recommendations = {
        'conservative': {
            'n_workers': max(1, cpu_count // 4),
            'threads_per_worker': 4,
            'memory_limit': f"{max(2, int(available_memory_gb * 0.2))}GB"
        },
        'balanced': {
            'n_workers': max(1, cpu_count // 2),
            'threads_per_worker': 8,
            'memory_limit': f"{max(4, int(available_memory_gb * 0.4))}GB"
        },
        'aggressive': {
            'n_workers': max(1, cpu_count - 1),
            'threads_per_worker': 2,
            'memory_limit': f"{max(8, int(available_memory_gb * 0.6))}GB"
        }
    }
    
    return recommendations

# Run the function
if __name__ == "__main__":
    configs = get_optimal_dask_config()
    for profile, config in configs.items():
        print(f"\n{profile.upper()} Profile:")
        print(f"  n_workers: {config['n_workers']}")
        print(f"  threads_per_worker: {config['threads_per_worker']}")
        print(f"  memory_limit: {config['memory_limit']}")
```

## Parameter Guidelines

### 1. Number of Workers (`n_workers`)

**Rule of Thumb:**
- **Conservative**: `CPU_CORES / 4` (minimum 1)
- **Balanced**: `CPU_CORES / 2` (minimum 1)
- **Aggressive**: `CPU_CORES - 1` (minimum 1)

**Considerations:**
- More workers = better parallelization but higher memory overhead
- Each worker is a separate process with its own memory space
- For MIMIC-IV data: Start with 2-4 workers for most systems

**Examples:**
- 4-core system: 1-2 workers
- 8-core system: 2-4 workers
- 16-core system: 4-8 workers

### 2. Threads per Worker (`threads_per_worker`)

**Rule of Thumb:**
- **I/O Heavy Tasks**: 8-16 threads (reading large CSV/Parquet files)
- **CPU Heavy Tasks**: 2-4 threads (complex computations, ML)
- **Mixed Workload**: 4-8 threads (typical MIMIC-IV analysis)

**Considerations:**
- Higher threads = better for I/O operations
- Lower threads = better for CPU-intensive tasks
- Total threads = `n_workers × threads_per_worker`
- Keep total threads ≤ 2 × CPU cores

### 3. Memory Limit (`memory_limit`)

**Rule of Thumb:**
- **Per Worker**: `AVAILABLE_MEMORY / n_workers × 0.8`
- **Safety Buffer**: Always leave 20-30% system memory free
- **Minimum**: 2GB per worker
- **Maximum**: 80% of available memory total

**Calculations:**
```
Total Dask Memory = n_workers × memory_limit_per_worker
Recommended: Total Dask Memory ≤ 60-70% of system memory
```

## Configuration Profiles

### Profile 1: Development/Testing (Small Datasets)
```python
n_workers = 1
threads_per_worker = 4
memory_limit = "4GB"
```
**Use Case:** Testing, small MIMIC-IV subsets, development

### Profile 2: Standard Analysis (Medium Datasets)
```python
n_workers = 2
threads_per_worker = 8
memory_limit = "8GB"
```
**Use Case:** Full MIMIC-IV tables, standard analysis workflows

### Profile 3: Heavy Processing (Large Datasets)
```python
n_workers = 4
threads_per_worker = 4
memory_limit = "16GB"
```
**Use Case:** Multiple merged tables, complex feature engineering

### Profile 4: Maximum Performance (High-End Systems)
```python
n_workers = 6
threads_per_worker = 2
memory_limit = "20GB"
```
**Use Case:** Full MIMIC-IV dataset, intensive clustering analysis

## Monitoring and Optimization

### 1. Dask Dashboard
Access at `http://localhost:8787` (or your configured port)

**Key Metrics to Watch:**
- **Memory Usage**: Should stay below 80% per worker
- **Task Duration**: Look for bottlenecks
- **Network Traffic**: High = potential optimization opportunity
- **CPU Utilization**: Should be balanced across workers

### 2. Performance Indicators

**Good Performance:**
- Memory usage: 60-80% per worker
- CPU utilization: 70-90%
- No frequent garbage collection
- Minimal task failures

**Poor Performance Signs:**
- Memory usage > 90% (risk of OOM)
- CPU utilization < 50% (underutilized)
- Frequent worker restarts
- Long task queues

### 3. Optimization Strategies

**If Memory Issues:**
- Reduce `memory_limit` per worker
- Increase `n_workers` (distribute load)
- Use chunking for large operations

**If CPU Bottlenecks:**
- Increase `threads_per_worker`
- Reduce `n_workers` (less overhead)
- Optimize computation algorithms

**If I/O Bottlenecks:**
- Increase `threads_per_worker`
- Use faster storage (SSD)
- Optimize data formats (Parquet vs CSV)

## MIMIC-IV Specific Recommendations

### Dataset Size Considerations

| Dataset Size | n_workers | threads_per_worker | memory_limit |
|--------------|-----------|-------------------|---------------|
| Small (< 1GB) | 1 | 4 | 4GB |
| Medium (1-10GB) | 2 | 8 | 8GB |
| Large (10-50GB) | 4 | 4 | 16GB |
| Very Large (> 50GB) | 6+ | 2 | 20GB+ |

### Common MIMIC-IV Operations

**Data Loading:**
- Higher `threads_per_worker` (8-16)
- Moderate `n_workers` (2-4)

**Table Merging:**
- Balanced configuration
- Monitor shuffle operations

**Feature Engineering:**
- Lower `threads_per_worker` (2-4)
- Higher `n_workers` (4-6)

**Clustering Analysis:**
- CPU-optimized configuration
- Maximum available memory

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `memory_limit`
   - Increase `n_workers`
   - Use data chunking

2. **Slow Performance**
   - Check Dask dashboard
   - Adjust thread/worker balance
   - Optimize data types

3. **Worker Crashes**
   - Reduce memory pressure
   - Check system resources
   - Monitor error logs

### Testing Configuration

```python
# Test script to validate configuration
import dask
from dask.distributed import Client, LocalCluster
import time

def test_dask_config(n_workers, threads_per_worker, memory_limit):
    """Test Dask configuration with sample workload."""
    
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    
    client = Client(cluster)
    
    try:
        # Simple test computation
        import dask.array as da
        x = da.random.random((10000, 10000), chunks=(1000, 1000))
        
        start_time = time.time()
        result = x.sum().compute()
        end_time = time.time()
        
        print(f"Configuration Test Results:")
        print(f"  Workers: {n_workers}")
        print(f"  Threads per worker: {threads_per_worker}")
        print(f"  Memory limit: {memory_limit}")
        print(f"  Computation time: {end_time - start_time:.2f} seconds")
        print(f"  Result: {result:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Configuration failed: {e}")
        return False
        
    finally:
        client.close()
        cluster.close()

# Test different configurations
configs = [
    (1, 4, "4GB"),
    (2, 8, "8GB"),
    (4, 4, "16GB")
]

for config in configs:
    test_dask_config(*config)
    print("-" * 50)
```

## Summary

1. **Start Conservative**: Begin with lower values and scale up
2. **Monitor Performance**: Use Dask dashboard actively
3. **Iterate and Optimize**: Adjust based on actual workload
4. **Consider Workload**: Different operations need different configurations
5. **System Limits**: Never exceed 80% of system memory

For most MIMIC-IV analysis workflows, start with:
- `n_workers = 2`
- `threads_per_worker = 8`
- `memory_limit = "8GB"`

Then adjust based on your system performance and specific use case requirements.