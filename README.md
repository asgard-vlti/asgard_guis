# asgard_guis
GUIs to be run on wag for asgard observations at VLTI

## Observation block creator

Install on wag with `pip install '.[ob-creator]'` from this directory, then run
`ob-creator HD123 HD456 --mode ut`. The resulting `.obd` file is written to the
directory where you run the command. The command uses the included
`templates/no_coords_dynamical.ONAXIS.obd` by default. Use `--template` to
select another template.
