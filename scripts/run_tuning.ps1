param(
  [int]$TotalSteps = 300000,
  [int]$Agents = 30,
  [int]$Horizon = 5000,
  [int]$RequestQueueSize = 10,
  [int]$TrainingLayoutCount = 4,
  [string]$Seeds = "0,1,2,3,4",
  [string]$RunsRoot = "artifacts/agv_runs",
  [switch]$NoGif
)

$ErrorActionPreference = "Stop"

$experiments = @(
  @{ name = "tune_base";          lr = "2.5e-4"; ent = "0.035"; wp = "0.03"; wait = "-0.02" },
  @{ name = "tune_lr_low";        lr = "1.0e-4"; ent = "0.035"; wp = "0.03"; wait = "-0.02" },
  @{ name = "tune_lr_high";       lr = "5.0e-4"; ent = "0.035"; wp = "0.03"; wait = "-0.02" },
  @{ name = "tune_entropy_low";   lr = "2.5e-4"; ent = "0.015"; wp = "0.03"; wait = "-0.02" },
  @{ name = "tune_entropy_high";  lr = "2.5e-4"; ent = "0.060"; wp = "0.03"; wait = "-0.02" },
  @{ name = "tune_reward_mild";   lr = "2.5e-4"; ent = "0.035"; wp = "0.02"; wait = "-0.01" },
  @{ name = "tune_reward_strong"; lr = "2.5e-4"; ent = "0.035"; wp = "0.05"; wait = "-0.03" }
)

foreach ($e in $experiments) {
  Write-Host "==== Training $($e.name) ===="

  python -m agv_drl.learning.train_ltf_ppo `
    --total-steps $TotalSteps `
    --agents $Agents `
    --horizon $Horizon `
    --request-queue-size $RequestQueueSize `
    --training-layout-count $TrainingLayoutCount `
    --regenerate-training-layout `
    --learning-rate $e.lr `
    --entropy-coef $e.ent `
    --waypoint-reward $e.wp `
    --active-wait-penalty $e.wait `
    --rollout-steps 128 `
    --minibatch-size 1024 `
    --runs-root $RunsRoot `
    --run-name $e.name

  $runDir = Get-ChildItem $RunsRoot -Directory |
    Where-Object { $_.Name -like "$($e.name)_*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  if ($null -eq $runDir) {
    throw "Cannot find training directory for $($e.name)"
  }

  $checkpoint = Join-Path $runDir.FullName "checkpoints/ltfp_latest.pt"
  if (-not (Test-Path $checkpoint)) {
    throw "Missing checkpoint: $checkpoint"
  }

  Write-Host "==== Evaluating $($e.name) with seeds $Seeds ===="

  $evalArgs = @(
    "-m", "agv_drl.experiments.evaluate_ltf_ppo",
    "--controller", "ltfp",
    "--checkpoint", $checkpoint,
    "--horizon", "$Horizon",
    "--agents", "$Agents",
    "--request-queue-size", "$RequestQueueSize",
    "--seeds", $Seeds,
    "--run-name", "eval_$($e.name)_5seeds_gif"
  )

  if (-not $NoGif) {
    $evalArgs += @("--render", "--render-every", "1", "--save-gif", "--gif-frame-ms", "80")
  }

  python @evalArgs
}

Write-Host "==== All tuning runs finished ===="
