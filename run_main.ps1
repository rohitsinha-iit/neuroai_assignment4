param(
    [int]$TrainingImages = 2000,
    [int]$ValidationImages = 500,
    [int]$TestImages = 2000,
    [string]$OutputName = 'main'
)
$ErrorActionPreference = 'Stop'
if ($OutputName -notmatch '^[a-zA-Z0-9_-]+$') {
    throw 'OutputName must contain only letters, numbers, underscores, and hyphens.'
}
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskResult = Join-Path $taskRoot "results\$OutputName"
New-Item -ItemType Directory -Force -Path $taskResult | Out-Null
$taskArguments = @('-u', '-X', 'utf8', '-m', 'snn_experiment.execute',
    '--output', "results/$OutputName", '--train', "$TrainingImages",
    '--validation', "$ValidationImages", '--test', "$TestImages",
    '--seeds', '42', '43', '44', '--targets', '20', '40', '60',
    '--adapt-size', '1000', '--adapt-epochs', '2', '--tau-sweep')
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments `
    -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $taskResult 'run.log') `
    -RedirectStandardError (Join-Path $taskResult 'errors.log')
@{
    process_id = $taskProcess.Id
    started = (Get-Date).ToString('o')
    arguments = $taskArguments
    working_directory = $taskRoot
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $taskResult 'launch.json')
Write-Output "Started experiment supervisor PID $($taskProcess.Id)."
Write-Output "Progress: $taskResult\run.log"
Write-Output "Status: $taskResult\status.json"
# Keep the supervising shell alive. Codex returns a managed running session;
# an unattached child can otherwise be reaped when the launching shell exits.
Wait-Process -Id $taskProcess.Id
