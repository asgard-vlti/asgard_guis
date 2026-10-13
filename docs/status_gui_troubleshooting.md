# Status GUI: colour and recovery lookup

Use this on the instrument system running the status GUI. The GUI polls the `mcs-client` watchdog at `tcp://mimir:7019` every 5 seconds; it reads SDLA and BDS positions separately from MDS at `tcp://mimir:5555`. `status-mimir --no-gui` shows the same watchdog and disk checks in a terminal. Check the displayed field and its source before starting a server or moving an axis. A colour applies to that field or card condition, not to the health of the whole instrument.

| Appearance | Meaning | First check |
| --- | --- | --- |
| Green value | Text contains `open`, `running`, or `true`; for `shm_error`, `false` is green. A green `disk` label means all monitored files have recent writes. | Confirm the relevant task is doing the intended work. |
| Red value | Text contains `closed`, `error`, or `false`; for `shm_error`, `true` is red. | Read the field and check the server below. |
| Yellow value | MDS `SDLA` returned `Error (Standby?)`; the GUI's `disk` label turns yellow when saving is incomplete or unverifiable. | Check MDS communication or the disk tooltip. |
| Grey value | No colour rule matched, including `cnt`, `locked`, `N/A`, `ready`, and `no-reply`. Grey does **not** establish that a value is healthy. | Interpret the value using the relevant server status. |
| Red card border | At least one card entry is red. Red takes precedence over yellow. | Find the red field first. |
| Yellow card border | A `cnt` did not change between fresh watchdog replies, MDS `SDLA` is zero or `Error (Standby?)`, or a disk-saving check is incomplete/unavailable. | Identify which of these applies; use the table below. |
| Red “last updated” header and dimmed cards | No valid watchdog reply for more than 10 seconds after a previous reply. Initial “waiting” is not yet a stale reading. | Check `mcs-client` on `mimir`, port 7019, and `/home/asg/logs/mcs_client/`. Treat card values as old. |

`process: running` on a server card can mean its TCP port accepts connections even when its `status` request did not reply; `zmq: no-reply` then stays grey. The Heim and Baldr telemetry `process` fields instead come from the watchdog's process-list check. A neutral card border only means none of the GUI's red or yellow rules fired.

## Card lookup

Run the installed `run_*` scripts on **mimir** only after confirming the corresponding process is absent. The scripts launch another process; they do not replace an existing one. Check the newest file in the listed log directory when a server is present but does not respond.

| Card | If red, yellow, or unexpected | Check and recovery |
| --- | --- | --- |
| BTT1–4 | Red `process`/`zmq`: Baldr tip-tilt port check failed. Grey `zmq: no-reply`: port opened but the status request timed out. Yellow border with otherwise normal fields: `cnt` was unchanged at the last poll. | Check `tcp://mimir:6671`–`6674` and `/home/asg/logs/baldr_tt/N/`. If beam `N` is confirmed down, use `run_baldr_tt N`. Check the camera feed and server log for an unchanged `cnt`. |
| BAO1–4 | Same watchdog rules as BTT; these are the **`dcs/baldr_jcr`** RTC servers. | For beam `N`, run `./supervisor.py N status` from the deployed `baldr_jcr` directory. Ports are `6662`, `6663`, `6664`, `6665` for beams 1–4; logs are in `/home/asg/logs/baldr/N/`. If confirmed down, use `run_baldr N`. The returned `cnt` is modulo 10,000, so repeated equal values alone do not prove a stopped loop. |
| CRED1 | Red `process`/`zmq`, or red `shm_error: True`. `cam_status` uses only the generic text rules; numeric `fps` and `gain` values stay grey. | Check port 6667 and `/home/asg/logs/cam_server/`; if confirmed down, use `run_cam_server`. For a live server, inspect its `status` and camera/SHM state using the [CRED1 command reference](dcs_command_reference.md#cred1-camera-server). `disk` is a separate check below. |
| DM | Red `process`/`zmq`, or a red status value. | Check port 6666 and `/home/asg/logs/DM_server/`; if confirmed down, use `run_DM_server`. Inspect the server `status` with the [DM command reference](dcs_command_reference.md#deformable-mirror-server). `disk` is separate. |
| HDLR | Red `process`/`zmq`, or yellow border from an unchanged `cnt`. `locked` is deliberately grey. | Check port 6660 and `/home/asg/logs/heimdallr/`; if confirmed down, use `run_heimdallr`. Confirm the intended lock state separately. |
| MDS | Red `process`/`zmq`: server check failed. Yellow `SDLA: Error (Standby?)`: no numeric reading (possibly standby); `SDLA: 0.0` gives a yellow border. Red `BDS`: beams are not all at `BIF_H` or all at `BIF_YJ`; hover for each position. | Check port 5555 and `/home/asg/logs/mds/`; if confirmed down, use `run_mds`. Check the MDS axis state before responding to SDLA/BDS. `align`, `empty`, or mixed BDS positions can be intentional during alignment. **TODO:** document the expected SDLA position and approved BDS/SDLA movement procedure for each operating mode. |
| Eng gui | Red `process`/`zmq`: watchdog cannot connect to the engineering GUI's port 8501. | Check `/home/asg/logs/eng_gui/`; if confirmed down, use `run_eng_gui`. This port check does not test GUI functions. |
| back_end | Red `process`/`zmq`: back-end server check failed. | Check port 7002 and `/home/asg/logs/back_end_server/`; if confirmed down, use `run_back_end_server`. |
| Heim Telem | Red `process`: `save-ft-performance` is absent. Yellow `disk`: performance or settings writes are missing/stale. | Check `/home/asg/logs/heim_telem/` and the disk tooltip. If both telemetry writers are confirmed down and needed, use `run_telem`. **TODO:** document the approved single-writer restart. |
| Baldr TT Telem | Red `process`: `save-tt-performance` is absent. Yellow `disk`: one or more beam performance/settings writes are missing/stale. | Check `/home/asg/logs/baldr_tt_telem/` and the disk tooltip. If both telemetry writers are confirmed down and needed, use `run_telem`. **TODO:** document the approved single-writer restart. |

Both BTT and BAO server `status` replies expose `cnt` modulo 10,000. A single yellow poll can therefore be a sampling coincidence; compare further status replies with the server log and camera activity before deciding the loop has stalled. **TODO:** document the approved recovery sequence for a live RTC with a confirmed stopped frame counter.

## Disk-saving label

The small `disk: X/Y streams saved` label is independent of the server's `process` and `zmq` fields. Green means every expected stream has a recent nonempty file or log write; yellow means at least one is stale or the disk reply cannot be verified (`?/Y`). The GUI uses only green or yellow for this label; `status-mimir --no-gui` also shows red disk summaries when no stream is fresh or the reply is invalid. Hover over the label for the stream names, write ages, and failure details. If recording is intentionally off, a yellow label can be expected.

| Card | Expected writes and freshness limit | If saving is expected |
| --- | --- | --- |
| CRED1 | Six FITS streams: `baldr1`–`baldr4`, `hei_k1`, `hei_k2`; each under 10 seconds old. | Check camera `save_mode`, `skip_save_baldr`, and today's `/data/YYYYMMDD/` files on `mimir`; use the [CRED1 commands](dcs_command_reference.md#cred1-camera-server). |
| DM | One `DM commands` FITS stream under 10 seconds old. | Check `get_save_status`; mode 0 means saving is off. If recording is required, enable `save 1` through the DM command client, then check `/data/YYYYMMDD/dms_*.fits`. See the [DM server README](../../dcs/asgard-dm-server/README.md). |
| Heim Telem | One performance log under 2 seconds and one settings log under 3 seconds. | Check the `save-ft-performance` process, `/home/asg/logs/heim_telem/`, and today's `/data/YYYYMMDD/` logs. |
| Baldr TT Telem | Four beam performance logs under 2 seconds and four settings logs under 3 seconds. | Check `save-tt-performance`, `/home/asg/logs/baldr_tt_telem/`, and the named beam files in today's `/data/YYYYMMDD/`. |

If all disk labels become unverifiable together, check the watchdog's `disk_status` reply and `/home/asg/logs/mcs_client/` before changing individual writers. **TODO:** document the local storage recovery procedure for full, unavailable, or unwritable `/data`.

Sources: [GUI colour and polling rules](../asgard_guis/cmd_scripts/status.py), [watchdog fields and ports](../../dcs/mcs_client/mcs_client.py), [disk freshness checks](../../dcs/mcs_client/disk_status.py), [Baldr JCR status command](../../dcs/baldr_jcr/README.md), [Baldr runner](../../dcs/run_scripts/run_baldr), [telemetry runner](../../dcs/run_scripts/run_telem), [MDS runner](../../asgard-alignment/asgard_alignment/run_scripts/run_mds), and [status-mimir invocation](cmd_scripts_reference.md#status-mimir).
