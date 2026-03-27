# Load Testing with Locust

This folder stress-tests the FastAPI model endpoints:

- `GET /health`
- `POST /predict` (weighted heavier than health checks)

## 1) Install Locust

```powershell
python -m pip install locust
```

## 2) Run matrix test against your Render service

```powershell
powershell -ExecutionPolicy Bypass -File .\loadtest\run_locust_matrix.ps1 `
  -HostUrl "https://<your-render-service>.onrender.com" `
  -UsersMatrix "5,10,20,40" `
  -SpawnRate 5 `
  -RunTime "90s"
```

Outputs:

- Per-run files: `loadtest/results/u{users}_stats.csv` and related Locust CSV files
- Combined summary: `loadtest/results/summary_predict_matrix.csv`

## 3) Compare different container counts

Run the same command above for each deployment size:

- 1 container (baseline)
- 2 containers
- 4 containers

Save each run in a separate output folder:

```powershell
# 1 container
powershell -ExecutionPolicy Bypass -File .\loadtest\run_locust_matrix.ps1 `
  -HostUrl "https://<service-1>.onrender.com" -OutDir "loadtest/results/c1"

# 2 containers
powershell -ExecutionPolicy Bypass -File .\loadtest\run_locust_matrix.ps1 `
  -HostUrl "https://<service-2>.onrender.com" -OutDir "loadtest/results/c2"

# 4 containers
powershell -ExecutionPolicy Bypass -File .\loadtest\run_locust_matrix.ps1 `
  -HostUrl "https://<service-4>.onrender.com" -OutDir "loadtest/results/c4"
```

Then compare `summary_predict_matrix.csv` across `c1/c2/c4` for:

- `median_ms` (typical latency)
- `p95_ms` / `p99_ms` (tail latency)
- `failures` (stability)
- `rps` (throughput)
