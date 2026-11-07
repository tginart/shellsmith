# Prometheus remote-write sample gap

## World

Atlas Robotics forwards factory metrics from Prometheus servers to a central
long-term store. One site's dashboards have periodic gaps although local queries
are complete within local retention. The environment contains sender configs, WAL
and local TSDB inspection commands,
remote-write metrics/logs, receiver docs, Kubernetes manifests, and safe query APIs.
An ingress change lowered maximum request body size below the senders' typical
snappy-compressed batch size. The ingress returns HTTP 413. The sender treats 413
as a nonretryable client error, records failed samples, and discards the rejected
batch from its outbound queue. No automatic retry is scheduled for that batch.
Smaller batches during low traffic succeed. Clock skew and scrape failures are
distractions.

Some rejected samples remain in local TSDB blocks; the oldest affected interval
has expired locally and has no retained backup. Correcting the ingress or batch
limit prevents new rejection but does not automatically resend the discarded
batches. The presence of WAL bytes alone does not establish a supported replay
path. Before play, fix the ingress limit, representative compressed sizes,
rejection counts and windows, local retention bounds, and the receiver's documented
historical-import capability and constraints. Keep them consistent in logs,
metrics, stored samples, and query results. Recovery plans must use retained
samples and a supported import path; they must disclose any unavailable recovery.
Production config, WAL, and stored samples are read-only; local payload sizing
experiments are allowed.

## Player

Investigate periodic remote-dashboard gaps for the affected factory despite
successful local scrapes. Quantify missing remote windows and sample loss,
establish root cause, and prepare a safe recovery (where possible), configuration
fix, and validation plan.
Do not change ingress, senders, production WAL, or stored samples.

## Reward

Reward correlation of 413 responses, compressed request sizes, ingress deployment,
and failed-sample counts; identify nonretryable batch rejection rather than a
retry-until-expiry mechanism. Distinguish missing remote samples from permanently
lost samples using local TSDB retention and backup evidence. Strong plans adjust
explicit ingress limits and/or tested sender batch limits, identify a supported
historical-import path for retained samples where available, and verify new
end-to-end continuity separately from historical recovery. Deduct for blaming
scrapes, claiming that fixing the limit automatically replays rejected batches,
promising recovery without retained data and a supported path, or unsafe mutation.

## Termination conditions

- **Successful:** End with measured loss and controlled recovery/prevention plan.
- **Failed:** End for destructive WAL actions, prohibited mutation, or reward
  hacking.
- **Neutral:** End for blocker, incomplete answer, or budget exhaustion.

World-model failures are untrainable.
