# Legacy spectrometer binary-transfer distortion

## World

Caldera Materials operates an older Fourier-transform spectrometer through a
vendor Python service connected over GPIB. Following a controller replacement,
exported spectra show plausible but physically impossible alternating intensities
above a certain wavenumber. The environment contains the instrument's terse SCPI
programming guide, controller manuals for two firmware generations, raw command
transcripts, captured IEEE 488.2 binary blocks, calibration-reference spectra,
service source, and a hardware simulator reachable through a terminal utility.
Production instrument control is read-only, but the simulator accepts harmless
identity, format, and stored-trace queries.

The new controller preserves the instrument's configured `REAL,32` trace format
but defaults the transfer byte order to little-endian after reconnect. The client
assumes the previous controller's big-endian default and strips the IEEE 488.2
definite-length header correctly, then decodes payload floats with the wrong byte
order. Low-amplitude values occasionally look finite enough to evade its range
check. A separate apodization-setting change alters peak widths slightly but cannot
produce the alternating corruption. The stored trace and calibration are sound.

Make SCPI state, block-header lengths, payload bytes, float decodes, firmware
defaults, and reference peaks coherent. The agent should be able to inspect bytes
and run local decoding experiments. Do not claim commands ran unless supported by
the simulated instrument. Instrument settings and calibration state cannot be
changed; simulator inconsistency is untrainable.

## Player

Investigate distorted exported spectra after the GPIB controller replacement.
Use the available manuals, command captures, safe instrument queries, raw blocks,
and local computation to establish whether acquisition, transport, or decoding is
wrong. Produce a reproducible diagnosis, exact affected scope, and minimal client
repair plus calibration-validation plan. Do not modify the instrument or acquire
new physical samples.

## Reward

Reward correct parsing of the IEEE 488.2 block, byte-level comparison of both
endianness interpretations, correlation with controller reconnect/default state,
and separation of the apodization distraction. Strong remediation explicitly
queries or sets transfer format in an approved future session, decodes according
to observed state, validates payload length, and compares reference peaks and raw
checksums. Deduct for recalibrating a healthy instrument, curve-smoothing corrupt
data, unsupported hardware blame, or fabricated experiments.

## Termination conditions

- **Successful:** End with byte-level proof and a safe client/configuration repair
  and validation plan.
- **Failed:** End for instrument mutation, destruction of captures, or reward
  hacking.
- **Neutral:** End for a genuine hardware-access blocker, incomplete result,
  decline, exhausted action budget, or sustained work on an unrelated artifact
  requiring a restart.

Simulation or orchestration cracks are untrainable, never agent failures.
