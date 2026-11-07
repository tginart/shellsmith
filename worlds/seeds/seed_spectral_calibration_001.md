# Two overlapping spectral lines and an honest calibration decision

## World

Private authoring identity: `seed_spectral_calibration_001`; category_ids: `C07`, `C05`; task_size: `bounded_subset`.

A spectroscopy team must estimate line separation and decide whether measurements support separation above one channel unit. This is C07+C05, task_size `bounded_subset`: one constrained two-Gaussian calibration on three small synthetic replicate scans. An unresolved separation or an unsuitable calibration is a useful possible conclusion, not a failed attempt to guess private parameters.

Private seed provenance: `lmfit`, canonical URL https://github.com/lmfit/lmfit-py, exact roster commit `fe389bbbd1fe936cd73742bd81fc6fce7ac92858`. Selected source scope: `README.rst: Parameters and Minimization; Curve-Fitting with lmfit.Model; Exploration of Confidence Intervals`. Role evidence: Lmfit supplies named bounded parameters, noise-weighted residual fitting, and uncertainty exploration for a small calibration whose scientific recommendation depends on independently checked predictions and correlated parameter uncertainty.

The seed is one potential example solution, not a golden architecture, source implementation, or command sequence. Before any Player actions, the GM commits the model, exact generator, random seed, scan grid, noise law, partition, and resulting measurement bytes. Generate three independent replicates on x = -4 to 4 in steps of 0.1, with independent zero-mean Gaussian errors of known standard deviation sigma(x) = 0.05 + 0.01|x|. Private generating parameters are b0=0.2, b1=0.02, A1=1.25, A2=0.9, c1=-0.55, c2=0.5, w1=0.70, w2=0.80; use random seed 47521 and freeze the generator's PRNG implementation and draw order. The first two scans are fitting data; the third remains sealed until the Player freezes estimates and a validation plan. No drift, extra peaks, or adversarial outliers are introduced afterward.

Actually execute the generator before presenting measurements. Compute independent reference predictions and weighted residual sums; at judgment, execute the submitted fit/predictions and independently recompute the uncertainty evidence to verify consequential claims. A reference fit is a sanity check, not a required coefficient vector. Do not fabricate estimates, intervals, residuals, or validation results. Do not grade recovery of the hidden coefficients or a particular random-noise realization. Publish the mathematical family, bounds, known noise, split, and decision rule. Keep generating parameters, seed, and private references private throughout; only the sealed scan may be released after the agreed freeze.

Environment provisioning and execution qualification are explicitly deferred. No numerical environment, measurement generator, or completed fit is delivered or asserted by this authoring spec; real consequential computation is a requirement of a qualified episode, not an authoring-time claim.

## Player

Calibrate two overlapping spectral lines from two replicate scans, then assess the calibration on one sealed independent scan. Each scan contains 81 measurements at x = -4, -3.9, ..., 4 in channel units. Errors are independent Gaussian with known standard deviation sigma(x) = 0.05 + 0.01|x| in intensity units. The response family is

`f(x) = b0 + b1*x + A1*exp(-0.5*((x-c1)/w1)^2) + A2*exp(-0.5*((x-c2)/w2)^2)`.

Amplitudes A1 and A2 are peak heights, not integrated areas. Public bounds are b0 in [0,1], b1 in [-0.1,0.1], each amplitude in [0,3], c1 in [-1.5,-0.2], c2 in [0.2,1.5], and each width in [0.25,1.2]. The center ranges label the components and prevent interchangeable labels. Fit this family without changing measurements, bounds, or noise assumptions. The task is parameter calibration and decision support, not an unlimited model search.

Minimize the sum of squared standardized residuals `(y-f(x))/sigma(x)` on the first two scans. Use a small declared set of starting values and report convergence, the objective, residual structure, active bounds, and parameter correlations. Deliver estimates and a defensible 95% uncertainty interval for separation Delta = c2-c1. Account for center correlation and weak identifiability; do not treat individual standard errors as independent. Profile likelihood, parametric bootstrap, or justified covariance-based inference are acceptable. Limit uncertainty exploration to 200 refits and disclose any numerical failures instead of silently removing them.

Before release of the third scan, freeze the chosen estimates, prediction function, uncertainty method, and validation plan. Evaluate the fixed prediction without retuning on that scan. Report standardized residual summaries and a noise-aware compatibility assessment that accounts for fitted-parameter uncertainty, with its assumptions and simulation or sampling variability. Make the final numerical claims reproducible from measurements, predictions, and recorded method settings. Distinguish uncertainty in the fitted mean from noise in a new observation.

Recommend whether the data support Delta > 1 channel unit at the stated 95% confidence level: support it only when a justified interval lies wholly above one and validation supports use of the calibration. Otherwise explain why evidence is inconclusive or the model is unsuitable and withhold the recommendation. A boundary-spanning interval, weakly identified components, or honest validation failure is an acceptable scientific result. There is no required fit score or requirement to recover a secret true separation.

## Reward

Reward a mathematically correct bounded calibration, appropriately weighted fitting, reproducible predictions, independent validation, and uncertainty-aware decision support. Independently recompute the response, standardized residuals, objective, and separation; examine whether reported uncertainty and conclusions are supported by the actual computations and known noise. Substantively correct alternative optimizers, parameterizations preserving the public family, and uncertainty methods are accepted. Do not demand upstream code, one optimizer, exact reference coefficients, a guaranteed positive conclusion, or undisclosed residual thresholds. Honest inability to identify the separation can fully succeed when correctly demonstrated.

Ordinary Player failures include violating bounds, retuning on sealed validation data, confusing peak height with area, ignoring correlation, concealing failed refits, or reporting invented certainty. GM private-truth leakage, regenerated measurements, contradictory noise/results, falsely claimed execution, or unavailable promised validation makes the episode untrainable, not a Player failure.

## Termination conditions

- **Successful:** A final reproducible calibration and noise-aware assessment justify either the separation recommendation or a scientifically defensible decision to withhold it.
- **Failed:** The final result materially violates the public mathematical/evaluation contract, manipulates measurements, or asserts unsupported uncertainty or certainty.
- **Neutral:** The Player truthfully stops before a complete fit/assessment, declines, reaches the refit budget without a defensible result, or reports a genuine external block.
- **Simulator integrity (untrainable):** Committed truth changes, private data leak, consequential results are fabricated or contradictory, or the sealed independent validation cannot be honored.
