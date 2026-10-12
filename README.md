# asgard_guis
GUIs to be run on wag for asgard observations at VLTI

## Log timestamps

`temperature-watchdog` writes `~/logs/templogs/tempWD_YYYYMMDDTHHMMSSZ.log`
with UTC CSV timestamps such as `2026-10-12T01:02:03.456Z`.
`temperature-plotting` displays UTC and also accepts numeric ISO offsets and
historical zone-less `YYYY-MM-DD HH:MM:SS` rows, interpreted as UTC because wag
uses UTC. Historical temperature files from another timezone must have their
offset supplied in each timestamp before using this plotter. The standalone
`asgard-alignment/temp_monitoring` scripts are legacy and are not these installed
commands.

`logs-client` accepts explicit UTC/offset timestamps and historical zone-less
timestamps when highlighting server messages.

## Observation block creator

Install on wag with `pip install '.[ob-creator]'` from this directory, then run
`ob-creator HD123 HD456 --mode ut`. The resulting `.obd` file is written to the
directory where you run the command. The command uses the included
`templates/no_coords_dynamical.ONAXIS.obd` by default. Use `--template` to
select another template.
