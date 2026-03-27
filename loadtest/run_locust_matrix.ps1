param(
    [Parameter(Mandatory = $true)]
    [string]$HostUrl,

    [string]$TargetImage = "data/viz_01_class_distribution.png",

    # Comma-separated list of user counts to test.
    [string]$UsersMatrix = "5,10,20,40",

    [int]$SpawnRate = 5,
    [string]$RunTime = "90s",
    [string]$OutDir = "loadtest/results"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

$usersList = $UsersMatrix.Split(",") | ForEach-Object { [int]($_.Trim()) }
$summaryRows = @()

foreach ($users in $usersList) {
    $prefix = Join-Path $OutDir ("u" + $users)
    Write-Host "Running Locust test users=$users spawnRate=$SpawnRate runTime=$RunTime host=$HostUrl"

    $env:TARGET_IMAGE = $TargetImage
    python -m locust `
        -f loadtest/locustfile.py `
        --host $HostUrl `
        --headless `
        --users $users `
        --spawn-rate $SpawnRate `
        --run-time $RunTime `
        --csv $prefix `
        --only-summary

    $statsPath = "${prefix}_stats.csv"
    if (!(Test-Path $statsPath)) {
        throw "Expected stats file not found: $statsPath"
    }

    $rows = Import-Csv $statsPath
    $predictRow = $rows | Where-Object { $_.Name -eq "POST /predict" } | Select-Object -First 1
    if ($null -eq $predictRow) {
        throw "No 'POST /predict' row found in $statsPath"
    }

    $summaryRows += [PSCustomObject]@{
        users            = $users
        requests         = [int]$predictRow."Request Count"
        failures         = [int]$predictRow."Failure Count"
        median_ms        = [double]$predictRow."Median Response Time"
        p95_ms           = [double]$predictRow."95%"
        p99_ms           = [double]$predictRow."99%"
        avg_ms           = [double]$predictRow."Average Response Time"
        rps              = [double]$predictRow."Requests/s"
    }
}

$summaryPath = Join-Path $OutDir "summary_predict_matrix.csv"
$summaryRows | Export-Csv -NoTypeInformation -Path $summaryPath

Write-Host ""
Write-Host "Saved summary to: $summaryPath"
$summaryRows | Format-Table -AutoSize
