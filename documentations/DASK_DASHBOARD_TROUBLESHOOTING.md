# Dask Dashboard Troubleshooting Guide

## Expected Behavior

When you check "Use Dask" in the sidebar, you should see:
1. **Dask Status** section in the sidebar
2. A clickable dashboard link
3. Worker and thread count information
4. The dashboard should open at http://localhost:8787

## Common Issues and Solutions

### Dashboard Not Showing

1. **Check if Dask client is initialized**
   - Look at the console/terminal for log messages about Dask client creation
   - Should see: "Dask client created successfully"

2. **Verify port availability**
   - Make sure port 8787 is not already in use
   - Check with: `lsof -i :8787` on macOS/Linux

3. **Check Streamlit session state**
   - The Dask client is stored in Streamlit session state
   - If you refresh the page, it should reuse the existing client

### Dashboard Link Issues

The dashboard link might appear in different formats:
- `http://localhost:8787`
- `http://127.0.0.1:8787`
- `http://[::1]:8787` (IPv6)

All should work the same. Try accessing each directly in your browser.

### Debugging Steps

1. **Check Dask client status in Python:**
   ```python
   import streamlit as st
   if 'dask_client' in st.session_state:
       client = st.session_state.dask_client
       print(client)
       print(client.dashboard_link)
       print(client.scheduler_info())
   ```

2. **Test dashboard directly:**
   ```python
   from dask.distributed import Client
   client = Client()
   print(client.dashboard_link)
   # Open link in browser
   ```

3. **Check firewall/security settings**
   - Some corporate networks block local ports
   - Try disabling firewall temporarily

### Alternative Access Methods

If the dashboard link doesn't work:

1. **Direct browser access:**
   - Open browser and go to: http://localhost:8787

2. **Use different dashboard pages:**
   - Status: http://localhost:8787/status
   - Workers: http://localhost:8787/workers
   - Tasks: http://localhost:8787/tasks

3. **Check with curl:**
   ```bash
   curl http://localhost:8787
   ```

### Configuration Options

The app creates a Dask client with:
- 2 workers
- 2 threads per worker
- 4GB memory limit per worker
- Process-based isolation

You can modify these in `setup_dask_client()` method if needed.

### Still Not Working?

1. Check the logs for errors
2. Try running the test script: `python test_dask_dashboard.py`
3. Verify Dask is installed: `pip install dask[distributed]`
4. Check for conflicting Dask installations
5. Try a different port by modifying the client creation code