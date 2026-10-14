# asgard_guis
GUIs to be run on wag for asgard observations at VLTI

## Observation block creator

Install on wag with `pip install '.[ob-creator]'` from this directory, then run
`ob-creator HD123 HD456 --mode ut`. The resulting `.obd` file is written to the
directory where you run the command. The command uses the included
`templates/no_coords_dynamical.ONAXIS.obd` by default. Use `--template` to
select another template.

## PDU configuration

The `mimir-startup`, `mimir-shutdown`, `read-all-pdu`, and `temperature-watchdog`
commands read the PDU IP and outlet assignments from the packaged
[`asgard_guis/pdu.toml`](asgard_guis/pdu.toml). Each outlet has a list of devices,
so a device may use multiple outlets and an outlet may supply multiple devices.
The current assignments are Mimir on outlets 1 and 8, MDS on 5, and C-RED on 6.
Outlets without a confirmed assignment are omitted.

Edit that TOML in the source checkout and reinstall `asgard_guis` on wag for a
change to take effect. Invalid or missing assignments stop the affected command
before it connects to the PDU or starts a shutdown. Temperature logs keep the
outlet names recorded when they were created, so older logs still plot correctly.

Run `read-all-pdu` on wag to print the status of each configured outlet as
`outlet: status`, ordered by outlet number.
