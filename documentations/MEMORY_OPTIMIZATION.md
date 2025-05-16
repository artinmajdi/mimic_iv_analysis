# Memory Optimization Guide for MIMIC-IV Dashboard

This document describes the memory optimizations implemented to prevent SIGKILL errors when running the Streamlit dashboard with large MIMIC-IV datasets.

## Problem

The Streamlit app was being killed with SIGKILL (signal 9) due to excessive memory usage when loading large MIMIC-IV tables.

## Solutions Implemented

### 1. Lazy Component Initialization

Instead of initializing all components at startup, we now use lazy initialization:

```python
@property
def data_handler(self):
    """Lazy initialization of data handler"""
    if self._data_handler is None:
        self._data_handler = DataLoader(mimic_path=Path(st.session_state.mimic_path))
    return self._data_handler
```

### 2. Dask Client with Memory Limits

Implemented a Dask client with strict memory limits to prevent OOM:

```python
def setup_dask_client(self):
    """Setup Dask client with memory limits to prevent OOM kills"""
    dask.config.set({
        'distributed.worker.memory.target': 0.5,  # Target 50% memory usage
        'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
        'distributed.worker.memory.pause': 0.8,   # Pause at 80%
        'distributed.worker.memory.terminate': 0.95,  # Terminate at 95%
    })
    
    self._dask_client = Client(
        n_workers=2,                # Reduced workers
        threads_per_worker=2,       # Reduced threads
        processes=True,             # Use processes for better memory isolation
        memory_limit='4GB',         # Limit per worker
    )
```

### 3. Garbage Collection

Added explicit garbage collection after clearing analysis states:

```python
def _clear_analysis_states(self):
    """Clears session state and forces garbage collection"""
    # ... clear session states ...
    gc.collect()  # Force garbage collection
```

### 4. Memory Usage Monitoring

Added real-time memory usage display in the sidebar:

```python
import psutil
memory = psutil.virtual_memory()
st.sidebar.caption(f"Memory: {memory.percent:.1f}% used ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
```

### 5. Safe Runner Script

Created `run_dashboard_safe.py` that monitors memory usage and restarts the app if needed:

```python
MAX_MEMORY_PERCENT = 70  # Restart if memory usage exceeds this percentage
CHECK_INTERVAL = 30      # Check memory every N seconds
MAX_RESTARTS = 3         # Maximum restarts before giving up
```

### 6. Streamlit Configuration

Updated `.streamlit/config.toml` with memory-optimized settings:

```toml
[server]
maxUploadSize = 100
maxMessageSize = 200
enableWebsocketCompression = true

[runner]
postScriptGC = true
fastReruns = false
magicEnabled = false
```

## Usage Recommendations

1. **Use Dask for Large Files**: Always enable the "Use Dask" option when working with large tables.

2. **Sample Data**: Use subject-based or row-based sampling to limit memory usage:
   - Subject-based: Loads data for a specific number of patients
   - Row-based: Loads a fixed number of rows

3. **Monitor Memory**: Keep an eye on the memory usage displayed in the sidebar.

4. **Check Dask Dashboard**: Access the Dask dashboard (link shown in sidebar) to monitor distributed computing resources.

5. **Use Safe Runner**: For production use, run the app with:
   ```bash
   python scripts/run_dashboard_safe.py
   ```

## Troubleshooting

If you still experience memory issues:

1. Reduce the number of Dask workers in `setup_dask_client()`
2. Decrease the memory limit per worker
3. Use more aggressive sampling
4. Close other memory-intensive applications
5. Consider upgrading system RAM

## Future Improvements

1. Implement streaming data processing for very large files
2. Add automatic data partitioning based on available memory
3. Implement caching strategies for frequently accessed data
4. Add memory profiling tools to identify bottlenecks