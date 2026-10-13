# ASGARD instrument changes — 1–13 October 2026

This changelog brings operators and developers up to speed on functionality and operating changes introduced during the October run, without requiring a detailed code review. It assumes familiarity with existing procedures and explains only the changes to them. Entries are ordered by importance within each audience and repository. Open work and observations needing validation are collected under **Developers → TODOs**.

The evidence is the four repository histories through 13 October and the supplied daily logs for 8–13 October. Earlier changes are covered by Git history. Implementation statements refer to the repository snapshots listed at the end; instrument results are identified separately where relevant.

## 1. Standard operators

### Overall workflow

**Recording health is now visible during observations.** The status GUI distinguishes a running server from recent writes to disk. Check the new recording indicators when saving is expected, and hover for the affected streams. A healthy process alone does not establish that the observation is being recorded.

**Baldr mode switching still leaves loops as they are.** Open the relevant loops before switching to Standard; the Shortcuts action and `b-mode` do not do this automatically.

**End-of-night state saving now follows the wag/mimir shutdown sequence.** Whole-instrument standby from wag saves the active instrument state before parking. Subsequent instrument shutdown on mimir avoids replacing it with the parked state. Details of the changed shutdown behavior are under `asgard-alignment` below; the established shutdown sequence still applies.

**Check mode transitions explicitly.** The 10–11 October logs include an Online transition that did not set SDLA and a missing `dl_type` setting. Verify SDLA/BDS configuration and delay-line selection rather than treating the new status display as a fix. See the [status troubleshooting guide](docs/status_gui_troubleshooting.md).

### asgard_guis

**Shortcuts now provides Heimdallr servo and offload controls.** Operators can select Lacour or DDSPC, stop the servo, select GD or Mod offloading, and use **All Off**, with live state and command replies displayed. **Servo Off leaves offloading unchanged; All Off disables both.** Selecting Lacour/DDSPC also selects Nested offloading; selecting GD/Mod stops the servo. Advanced DDSPC tuning is described in section 2.

**Stop Recon (All) also clears Baldr TT corrections.** It terminates local `baldrtt-recon` jobs, stops all four Baldr TT servos, and clears their owned DM channels and TT/focus offsets. It does not clear every DM channel. The observing launcher runs Shortcuts on mimir, which matters because reconstruction-process cancellation is local. The 11 October log records recovery of a black DM panel with this control.

**Lab/Sky transitions preserve camera settings.** Lab Mode saves the first gain/FPS/NDMR snapshot and switches to 1000 FPS, gain 3 and NDMR 1, taking a dark before enabling the lab source. Repeated Lab requests preserve that original snapshot. Sky Mode restores it if no more than two hours old; otherwise it uses 1000 FPS, gain 5 and NDMR 1. Restoration now includes delays between camera commands. **Sky Mode does not automatically take a dark.**

**Status cards now include disk and beam-configuration information.** CRED1, DM, Heim telemetry and Baldr TT telemetry have compact recording labels. Hover over a label to open a tooltip with stream names, write ages and errors. BDS positions are summarized, with individual positions in the tooltip. Yellow recording indicators can be expected when saving is deliberately disabled. Use the [colour and recovery lookup](docs/status_gui_troubleshooting.md) for interpretation.

**Observation blocks can be generated with an installed command on wag.** Run from the desired output directory, using the environment with the `ob-creator` extra installed:

```sh
ob-creator HD123 HD456 --mode ut
```

Use `--mode at` for AT observations. The command queries SIMBAD, chooses shared camera/Baldr settings from the faintest target, and writes the `.obd` in the current directory using the bundled ONAXIS template. `--template` selects another template. This replaces reliance on an untracked, separately located script; see the [OB creator notes](README.md#observation-block-creator).

**Desktop dropdown menus and alignment feedback improved.** In the ASGARD desktop dropdown menu, choose **Heimdallr/Baldr Panels → Setup Fringes** to launch missing observing windows with coordinated startup positions; it reuses existing windows without rearranging them. Mirrored outputs are supported when they form two distinct X display regions. H-guis now ranks telescopes by five-second GD SNR, with rank 1 highest. The faint-alignment Save control is farther from the movement buttons, and **Spherical?** disables the incompatible Align CRED box/BOTX action. Text clients and server logs also gained darker, more legible displays.

### archive_tools

**Compression is now available from any directory on mimir.** In the installed instrument environment:

```sh
compress-fits /data/20261011 /path/to/archive/20261011
```

The command compresses top-level FITS images and copies other files and subdirectories, including `headers`, while retaining the source. FITS inside subdirectories are copied unchanged. `--resume` skips existing FITS outputs without checking their completeness, so inspect a file interrupted during writing before resuming. `archive-size-summary`, run in the data directory, now summarizes allocated disk space by file type, including subdirectories.

**A new bundle GUI collects selected data and supporting material.** Launch `archive-bundle` on mimir. It supports a date/time interval, explicit UTC/UTC−3/UTC−4 selection, stream/category selection and optional compression. **Update files** triggers a scan limited to relevant date folders. The destination must be an existing empty folder outside configured source roots. Output contains `Data`, `Logs`, `Config`, `Headers` and a manifest, with colon-free output filenames enabled by default for external drives. Sources remain unchanged; the manifest records incomplete or cancelled runs. This addresses collecting a night's material without adding a multi-night option to `compress-fits` itself. See the [archive guide][archive-readme]; developers should check the log-selection issue in section 3.

### asgard-alignment

**Shutdown now preserves state and requests orderly server exits.** If the instrument is still active, `instrument-shutdown` saves a `before_shutdown_*` state; if wag already put it in standby, it does not save the parked configuration again. State files remain in `~/.config/asgard-alignment/instr_states/`. Shutdown sends the DM server an exit request and, for normal shutdown, stops and exits the camera server before switching off the lower-box outlet; failed acknowledgements stop that sequence.

**Baldr shutters work again with the corrected configuration handling.** The motor selection, movement waiting and shutter state bookkeeping were fixed; the 12 October log confirms working shutters. Missing named positions now produce an explicit error. Instrument startup also retries a failed Controllino ping once after five seconds.

### dcs

**The watchdog now checks disk output on mimir.** Camera/DM FITS and Heimdallr/Baldr TT performance/settings files feed the Status indicators and their hover tooltips. Camera and telemetry saving were robust during the run; DM saving remained unreliable. Reduced waiting on unavailable wag communication also keeps laboratory status checks responsive.

**Alignment and viewing are easier to follow.** Heimdallr pupil autoalignment now has progress reporting. DMview uses a fixed colour scale for its piston column, and CRED1/DM viewers accept the startup geometry used by Setup Fringes.

## 2. Advanced operators

### Overall workflow

**Performance experiments have richer supporting data.** Mode/exploration telemetry, saved DDSPC model snapshots and time-selected archive bundles now make it easier to compare controller settings with recorded camera and DM behavior. The relevant capabilities and their limits are described below and in the developer section.

### dcs

**Predictive fringe tracking is available as DDSPC.** It predicts three differential piston modes and applies bounded DM commands. Learning/exploration can continue or freeze according to configuration; a frozen model still applies corrections. The initial Lacour phase-delay gain has also been reduced to 0.05.

Use these in the **Heimdallr text client**, with the quoted action strings:

| Command | Changed capability |
| --- | --- |
| `servo "ddspc"` | Enter predictive fringe tracking. |
| `ddspc "get"` | Inspect configured/active profiles and freeze state. |
| `ddspc "freeze"` | Freeze the active fit and stop exploration dither at the next loop boundary. |
| `ddspc "set-history" [30,3]` | While Servo Off, select history/future dimensions for the next DDSPC run. |

Supported history/future pairs are `[20,2]`, `[30,3]`, `[40,3]` and `[50,3]`. Ordinary parameter setters stage the next DDSPC entry rather than retuning the active run. The checked-in configuration explores for 2500 valid frames, then freezes learning; use `ddspc "get"` to inspect the server's active settings. See the [DDSPC reference][ddspc-doc].

**Dropout recovery now distinguishes learning and frozen models.** Gaps or lost tracking reset a learning fit; a frozen fit is retained, with tracking history re-established before resuming. Recovery can hold the previous piston command for five valid paired frames, or zero all four channels if the held command is near saturation or non-finite. Disconnected phase measurements also trigger a hold. Delay-line search/offload continue. Switching from modulation to Lacour/DDSPC now ends modulation correctly.

**Visibility estimation changed.** Heimdallr now combines four neighboring Fourier samples with revised amplitude/noise treatment. Changes in displayed visibility or SNR can therefore reflect the estimator as well as optical alignment. The amplitude correction was revised during the run to account for the applied FFT window.

**New alignment and recovery commands run on mimir.** `power-cycle-camera` automates the camera/PDU/server restart and cooling checks previously done manually; it was exercised during the run and may wait up to 15 minutes for cooling. `h-pupil-track --dry-run` measures pupils and previews corrections without moving motors; `h-pupil-track` applies MDS corrections when the vector error exceeds 0.05. See [camera recovery][camera-recovery] and the [pupil-tracking reference](docs/cmd_scripts_reference.md#h-pupil-track).

**New diagnostic tools remain experimental.** `sparkles 1` creates symmetric DM probes for beam 1 and uses `disp03`; opening, switching or normally closing the tool clears that channel, so account for other tools using it. Cold-stop scans, rolling flux/Strehl displays, standard-mode model acquisition and spatial transfer-function tests were also added. See [sparkles][sparkles] and [cold-stop scans][cold-stop].

### asgard-alignment

**HPOL sweeps support systematic visibility measurements.** `heimdallr/m_linbo_search.py` now sweeps a configured beam's HPOL, realigns tilts/finds fringes and records K1/K2 visibility versus position. It restores the starting HPOL and writes plots/data; it does not automatically choose an optimum. Settings are script constants. `move-hpol` positions are absolute, an important clarification from the run.

**MDS accepts integer movement arguments as well as floats.** The previous float-only workaround for `moveabs`/`moverel` is no longer necessary. `standby_status` now exposes `ACTIVE`/`STANDBY` for shutdown coordination.

### asgard_guis

**Advanced Shortcuts controls read back server state.** History changes require a fresh Servo Off state; GD/Mod offload controls require a fresh enabled delay-line configuration. Configured versus active DDSPC settings and command replies are shown. H-guis also accepts `--server-host HOST --server-port PORT` for alternative endpoints.

**Logs follow server restarts.** The log viewer follows the newest timestamped launch log, retains a fallback to `log.txt`, preserves scroll position and highlights warning/error output. Existing process launch/kill functions are unchanged capabilities.

### archive_tools

**External-drive filename handling is now explicit.** On an export copy, `rename-fits --dir_path=/path/to/export --replace-colons` replaces colons in top-level FITS names in place and checks for destination-name collisions. Perform telemetry combining before renaming because it expects ISO filenames. The bundle GUI can instead normalize output names while leaving originals alone. Current CRED1/DM filenames use UTC: the older `--hour_diff=3` conversion is not a blanket step for new October data.

**Time selection has defined limits.** Bundling trims timestamped telemetry/server logs to the selected interval; selected unparseable logs are copied whole and marked untrimmed. Headers are currently copied by date folder. An incomplete bundle requires a new empty destination for another run, whereas `compress-fits --resume` only skips existing FITS outputs.

## 3. Developers

### Overall workflow

**Some updates require matching server and client versions.** Buffered FT telemetry requires the updated Heimdallr server, and Stop Recon requires Baldr TT's new reset command. Git checkout state alone does not establish which binary is running. Keep the wag GUI, mimir servers and launch scripts coordinated across their separate installations.

**Reproducibility improved across repositories.** FT/TT telemetry headers include the DCS and alignment commit IDs; operational console logs now identify individual launches. The commit-ID helper expects both repositories as sibling checkouts and raises if it cannot read them, which is relevant to alternative installation layouts. See [commit recording][commit-ids].

### dcs

**FT telemetry now drains a server ring buffer.** `save-ft-performance` replaces sampled snapshots with batches from a 1024-record ring, requesting up to 128 records at a time. The producer captures processed active-loop frames and limits off-mode records to at most 10 Hz. Timestamps come from the server's frame processing, not logger polling or camera hardware exposure timestamps. Startup begins at the latest available sequence rather than backfilling the buffer. This improves capture fidelity but does not guarantee every camera frame is represented. See the [buffering commit][ft-buffer-commit] and [current writer][ft-writer].

**Telemetry readers need to handle the changed format and continuity markers.** Rows now end with `cnt seq servo_mode is_ddspc_exploring`; `cnt` wraps modulo 10,000. Use sequence numbers and stream/restart information for telemetry continuity, while frame counters help identify acquisition gaps. `# telemetry_loss` and `# telemetry_restart` comments report loss and restarts. Timestamp precision is four decimal places and measurement formatting is three decimal places. Settings are still sampled at 1 Hz over a separate socket. Non-default `--rate`/`--gdrate` values are rejected; legacy default arguments remain accepted. An old reachable server without `ft_telemetry` fails explicitly.

**DDSPC snapshots and offline diagnostics are available.** Direct transitions from DDSPC to Servo Off, including `offload "gd"`/`"mod"`, queue an asynchronous versioned JSON snapshot under `/data/YYYYMMDD/`; other mode changes do not. Snapshots retain model dimensions, parameters and the last trained segment after interruptions. Simulation builds, parity/configuration/snapshot checks and per-component benchmarks support development. Benchmark results exclude camera and full servo-loop work. See [model snapshots and checks][ddspc-snapshots].

**Baldr and watchdog robustness changed.** Reconstruction cancellation/failure now cleans up TT state; Baldr JCR checks camera timing and skips invalid reset/read frames. Zernike generation reduces unwanted energy outside the pupil. MCS retains pending script results and backs off failed communication with wag.

**Console logs are per launch.** Operational wrappers write `log_<UTC timestamp>_<PID>.txt`; the backend's duplicate internal file logger was removed. The correct DM log directory is `/home/asg/logs/DM_server/`, not the older `dmserver` directory. See the [launcher logging change][dcs-logging].

### asgard_guis

**Git terminals and desktop menus are now tracked.** **ASGARD → dev → git terminals** opens a 4×2 grid for the four repositories, wag above and mimir below. On wag, DCS, alignment and archive tools are mirror repositories; `asgard_guis` is a working checkout. The active menu on `wvguws` is a separately deployed copy of `misc-all.wmrc`; committing it does not update the desktop. Its header documents sync commands, and **Restart WM** refreshes menus without logging out. See [Git terminals][git-terminals] and [menu configuration][desktop-menu].

**Setup Fringes preserves host boundaries.** The launcher runs on wag; Shortcuts, DMview and logs run through SSH/X forwarding on mimir. Geometry is passed at startup to supported viewers. Mirrored outputs are coalesced into distinct display regions; the current layout requires two such regions. Launcher diagnostics go to `/vltuser/asg/setup-fringes.log` on wag. See [launcher code][fringes-launcher].

**Camera restoration has persistent state.** Lab/Sky scripts share `/home/asg/.config/s_labmode_camera_settings.json`, preserve the first valid snapshot, restore NDMR before FPS/gain and remove the snapshot after successful restoration. The two-hour age limit and fallback values are part of this behavior. The log viewer likewise understands both the timestamped console filenames and legacy `log.txt`. These contracts matter when changing launch hosts or deployment paths.

### asgard-alignment

**MDS file logging now belongs to the launch wrapper.** `run_mds` and `run_eng_gui` capture console output into timestamped files under `/home/asg/logs/mds/` and `/home/asg/logs/eng_gui/`. MDS's internal file handler and `--log-location` option were removed, avoiding duplicate logs. A direct MDS launch now logs to the console; use the wrapper for file capture. See [duplicate-log removal][mds-logging].

**Manual MDS/engineering-GUI closure is still required when prompted.** Automatic closure was added and then reverted on 13 October.

**Motor/configuration failures are explicit.** Zaber out-of-range absolute moves and absent named positions raise `ValueError`. Baldr configuration accepts `named_positions` and the legacy `named_pos` key. The shutter incident showed that deployed MDS configuration differed from the checkout, so deploy the matching configuration as well as the code. See [shutter and named-position fix][shutter-fix].

### archive_tools

**The utilities are packaged entry points.** Python 3.10+ exposes `compress-fits`, `archive-size-summary`, `rename-fits` and `archive-bundle`. The 13 October run log confirms the offline installation needs `--no-build-isolation`; use the package's documented offline command and available local wheels. Compression now accepts primary images with one or more dimensions and preserves extra HDUs. The compressed image is in an extension behind an empty primary HDU, which readers assuming image data in HDU 0 must handle. See [packaging][archive-packaging] and [compression implementation][compression-code].

**Bundle selection is configurable and auditable.** Packaged TOML defaults, `~/.config/asgard-archive-tools/bundle.toml` and `--config` configure sources, clocks and estimates; manifests record output and completion. Bundled configurations are the current files on disk, not historical configurations reconstructed for the selected interval.

### TODOs

- **Overall workflow:** make switching to Standard open the relevant Baldr loops automatically, as requested on 11 October. Document expected SDLA/BDS and `dl_type` settings by mode and verify Online sets them; the 10–11 October logs recorded missing settings. See the [status troubleshooting guide](docs/status_gui_troubleshooting.md).
- **asgard_guis:** specify when Sky Mode needs a new dark; show an idle camera in yellow and signal a stopped CRED1 viewer stream; clarify the Stop Recon button's stop/clear effect and highlight Close TT state. **Needs validation:** the 11 October status/`get_gain` mismatch, confusing DM recording display on 12 October, and intermittent black panels or other Baldr TT mode failures.
- **dcs:** test and fix unreliable DM saving; investigate camera/frame-gap causes and validate complete-loop timing, since gaps persisted on 12 October; synchronize Baldr modulation to reduce fringe-tracker coupling. **Needs validation:** the cold-stop flux rise at large probe separation on 13 October and the experimental flattening/calibration results.
- **asgard-alignment:** restore automatic MDS/engineering-GUI closure once reliable; establish repeatable on-sky HPOL optimization and persistence. The 9 October sweep found wide/multiple peaks, and persistence was requested on 13 October.
- **archive_tools:** select individual header files by observation time instead of copying whole date folders. **Needs validation:** packaged MDS and generic server-log patterns select `.log`, while operational launchers produce `.txt`; MCS's internal `.log` files remain selected, but its wrapper console logs need matching patterns. Verify deployed paths and patterns before relying on complete bundles. This mismatch was found in [bundle defaults][bundle-defaults], not reported as a run incident.

<!-- **TODO — USB recovery:** turn the successful 13 October USB-controller recovery and subsequent MDS restart into a documented, reviewed recovery procedure. The logged manual controller reset is not yet integrated into startup. -->

---

Evidence snapshots: `asgard-alignment` [fb65ce7][alignment-head], `dcs` [e04ab61][dcs-head], `asgard_guis` [3d49249][guis-head], `archive_tools` [0ce3d3a][archive-head]. Daily CSV sources: `runs/oct2026/log - 8 Oct.csv`, `log - 9Oct.csv`, `log - 10 Oct.csv`, `log - 11Oct.csv`, `log - 12Oct.csv`, `log - 13Oct.csv`. Git dates are interpreted in Santiago time. Photos, recordings and separate dump logs mentioned by those CSVs were not supplied.

[alignment-head]: https://github.com/asgard-vlti/asgard-alignment/commit/fb65ce7c1a15540cb0c1cf9f1a68232335447bab
[dcs-head]: https://github.com/asgard-vlti/dcs/commit/e04ab61a3ff216af152a6c68117e39aa03a73c5e
[guis-head]: https://github.com/asgard-vlti/asgard_guis/commit/3d492493856f32707fa10374aa2fb7cec89b840a
[archive-head]: https://github.com/asgard-vlti/archive_tools/commit/0ce3d3a0958198a192d6f2ea845f3a8e5a88f5d7
[archive-readme]: https://github.com/asgard-vlti/archive_tools/blob/0ce3d3a0958198a192d6f2ea845f3a8e5a88f5d7/README.md
[ddspc-doc]: https://github.com/asgard-vlti/dcs/blob/main/heimdallr/DDSPC.md
[camera-recovery]: https://github.com/asgard-vlti/dcs/blob/e04ab61a3ff216af152a6c68117e39aa03a73c5e/README.md#camera-power-cycle
[sparkles]: https://github.com/asgard-vlti/dcs/blob/e04ab61a3ff216af152a6c68117e39aa03a73c5e/README.md#static-dm-sparkles
[cold-stop]: https://github.com/asgard-vlti/dcs/blob/e04ab61a3ff216af152a6c68117e39aa03a73c5e/README.md#cold-stop-sparkle-flux-scan
[commit-ids]: https://github.com/asgard-vlti/dcs/blob/e04ab61a3ff216af152a6c68117e39aa03a73c5e/dcs/log_commit_ids.py
[ft-buffer-commit]: https://github.com/asgard-vlti/dcs/commit/30fa243
[ft-writer]: https://github.com/asgard-vlti/dcs/blob/e04ab61a3ff216af152a6c68117e39aa03a73c5e/heimdallr/save_ft_performance.py
[ddspc-snapshots]: https://github.com/asgard-vlti/dcs/blob/main/heimdallr/DDSPC.md#model-snapshots
[dcs-logging]: https://github.com/asgard-vlti/dcs/commit/6a54611
[git-terminals]: https://github.com/asgard-vlti/asgard_guis/blob/3d492493856f32707fa10374aa2fb7cec89b840a/open-git-terminals
[desktop-menu]: https://github.com/asgard-vlti/asgard_guis/blob/3d492493856f32707fa10374aa2fb7cec89b840a/misc-all.wmrc
[fringes-launcher]: https://github.com/asgard-vlti/asgard_guis/blob/3d492493856f32707fa10374aa2fb7cec89b840a/setup-fringes
[mds-logging]: https://github.com/asgard-vlti/asgard-alignment/commit/af5825a
[shutter-fix]: https://github.com/asgard-vlti/asgard-alignment/commit/2e0a7a2
[archive-packaging]: https://github.com/asgard-vlti/archive_tools/blob/0ce3d3a0958198a192d6f2ea845f3a8e5a88f5d7/pyproject.toml
[compression-code]: https://github.com/asgard-vlti/archive_tools/blob/0ce3d3a0958198a192d6f2ea845f3a8e5a88f5d7/compress_fits_cli.py
[bundle-defaults]: https://github.com/asgard-vlti/archive_tools/blob/0ce3d3a0958198a192d6f2ea845f3a8e5a88f5d7/archive_bundle/bundle_defaults.toml
