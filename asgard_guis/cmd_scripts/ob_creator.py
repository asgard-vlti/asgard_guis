# Tool to create OBs for ASGARD
# Revised by Yi Lu v7 on 15/04/2026
# syntax = ob-creator <name of target 1> <name of target 2>... --mode at|ut
# This version assign DIT/GAIN/NWORESET/BLDMODE based on the faintest target
# This version allows different numbers of targets
# This verison allows --help; --mode for UT or at; --label for comment; --bldmode for manually entering bldmode
#
# required file = obd file in which target characteristics (= template-parameter
#                 values) have been replaced by markers ("__*__")
#                 name of that file: no_coords_dynamical.ONAXIS.obd
#


import sys
from pathlib import Path

TEMPLATE_NAME = "no_coords_dynamical.ONAXIS.obd"
PACKAGE_TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / TEMPLATE_NAME
SOURCE_TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / TEMPLATE_NAME
DEFAULT_TEMPLATE = PACKAGE_TEMPLATE if PACKAGE_TEMPLATE.is_file() else SOURCE_TEMPLATE

def create_ob(targets_data, output_filename, template_file=DEFAULT_TEMPLATE):
    try:
        with open(template_file, "r", encoding="utf-8") as infile:
            full_template = infile.read()
    except FileNotFoundError:
        print(f"Error: Could not find the template file '{template_file}'.")
        sys.exit(1)

    try:
        header, rest = full_template.split("__TARGET_BLOCK_START__\n")
        target_block_template, footer = rest.split("__TARGET_BLOCK_END__\n")
    except ValueError:
        print("Error: Template is missing the __TARGET_BLOCK_START__ or __TARGET_BLOCK_END__ tags.")
        sys.exit(1)
        
    final_content = header
    
    for target in targets_data:
        current_block = target_block_template
        
        for key, value in target.items():
            current_block = current_block.replace(f"__{key}__", str(value))
            
        final_content += current_block
    
    final_content += footer

    # Write the resulting OB
    with open(output_filename, "w", encoding="utf-8") as outfile:
        outfile.write(final_content)
    print(f"Success! Created observation block: {output_filename}")
    
    
def extract_simbad_data(target_name):
    import numpy as onp
    from astroquery.simbad import Simbad
    from astropy.coordinates import Angle

    print(f"Querying Simbad for {target_name}...")
    Simbad.add_votable_fields("pmra", "pmdec", "V", "G", "J", "H", "K", "plx_value", "rvz_radvel")
    result = Simbad.query_object(target_name)
    print("Got the info !")

    ra0 = result[0]['ra']
    ra_str = f'{ra0}d'
    ra_angle = Angle(ra_str)
    ras = ra_angle.hms
    ra_out = f'{ras.h:02.0f}{ras.m:02.0f}{ras.s:06.3f}'
    
    dec0 = result[0]['dec']
    dec_str = f'{dec0}d'
    dec_angle = Angle(dec_str)
    decs = dec_angle.dms
    dec_out = f'{decs.d:02.0f}{abs(decs.m):02.0f}{abs(decs.s):06.3f}'
    
    val_pmra = result[0]['pmra']
    if onp.ma.is_masked(val_pmra) or onp.isnan(val_pmra):
        pma_out = '0.'
    else:
        pma_out = f"{(float(val_pmra)/1000):.3f}"
        
    val_pmdec = result[0]['pmdec']
    if onp.ma.is_masked(val_pmdec) or onp.isnan(val_pmdec):
        pmd_out = '0.'
    else:
        pmd_out = f"{(float(val_pmdec)/1000):.3f}"
        
    val_h = result[0]['H']
    if onp.ma.is_masked(val_h) or onp.isnan(val_h):
        hmag_out = '99.'
    else:
        hmag_out = f"{float(val_h):.2f}"
    
    val_k = result[0]['K']
    if onp.ma.is_masked(val_k) or onp.isnan(val_k):
        kmag_out = '99.'
        k_float = 99.0
    else:
        kmag_out = f"{float(val_k):.2f}"
        k_float = float(val_k)

        
    val_parallax = result[0]['plx_value']
    if onp.ma.is_masked(val_parallax) or onp.isnan(val_parallax):
        plx_out = '0.'
    else:
        plx_out = f"{(float(val_parallax)/1000):.3f}"
    
    #val_radvel = result[0]['rvz_radvel']
    #if np.ma.is_masked(val_radvel) or np.isnan(val_radvel):
    # radvel_out = '0.'
    #else:
    #radvel_out = f"{(float(val_radvel)):.3f}"
        
    val_g = result[0]['G']
    if onp.ma.is_masked(val_g) or onp.isnan(val_g):
        gmag_out = '99.'
    else:
        gmag_out = f"{float(val_g):.2f}"
        


    print(f"{ras.h:02.0f} h {ras.m:02.0f} m {ras.s:06.3f} s")
    print(f"{decs.d:02.0f} h {abs(decs.m):02.0f} m {abs(decs.s):06.3f} s")
    print(f"ra = {ra_out} ; dec = {dec_out} ; Gmag = {gmag_out} ; Kmag = {kmag_out} ; Hmag = {hmag_out}")
    print(f"pma = {pma_out} arcsec/yr ; pmd = {pmd_out} arcsec/yr ; plx = {plx_out} arcsec ")
    print("------------------------------")
    
    return {
        "NAME": target_name,
        "RA": ra_out,
        "DEC": dec_out,
        "PMA": pma_out,
        "PMD": pmd_out,
        "H": hmag_out,
        "K": kmag_out,
        "K_FLOAT": k_float,
        "PARALLAX": plx_out,
        "G": gmag_out
    }
       

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ASGARD Observation Blocks for single or multiple targets.",
        epilog="Example: ob-creator HD123 HD456 --mode ut --label Commu --bldmode FAINT"
    )
    
    parser.add_argument("targets", nargs='+', help="List of target star names (e.g., HD123 HD98922)")
    parser.add_argument("-m", "--mode", choices=['at','ut'], required=True, help="Telescope mode: 'at' or 'ut'")
    parser.add_argument("-l", "--label", type=str, default="", help="Optional label to prepend to the filename")
    parser.add_argument("-b","--bldmode", type=str.upper,
        choices=['STANDARD','FAINT'], help="Optional: Override automatic BLDMODE setting (STANDARD or FAINT)")
    parser.add_argument("-t", "--template", type=Path, default=DEFAULT_TEMPLATE,
        help="OB template path (default: the template included with asgard_guis)")
        
    args = parser.parse_args()
    if not args.template.is_file():
        parser.error(f"Template file not found: {args.template}. Use --template to select one.")
    
    target_names = args.targets
    mode = args.mode.lower()
    label = args.label
    manual_bldmode = args.bldmode
    

    all_targets_data = []
    
    for name in target_names:
        target_dict = extract_simbad_data(name)
        all_targets_data.append(target_dict)
    
    faintest_k = max([target["K_FLOAT"] for target in all_targets_data])
    
    print(f"\nEvaluating System: Mode is '{mode.upper()}' | Faintest K-mag is {faintest_k}")
    #AT settings
    if mode == 'at':
            
    # Detector settings based on faintest star
    
        if faintest_k < 5.5: #Bright
            global_dit = "0.0008"
            global_gain = "20"
            global_nworeset = "1"
            global_bldmode = "FAINT"
    
        elif 5.5 <= faintest_k <= 9.0:
            global_dit = "0.002"
            global_gain = "30"
            global_nworeset = "1"
            global_bldmode = "FAINT"
#            global_dit = "0.0008"
#            global_gain = "30"
#            global_nworeset = "21"
#            global_bldmode = "FAINT"
        
        else:               #Faint
            global_dit = "0.0008" #should be 0.0004
            global_gain = "30"
            global_nworeset = "51" #Should be 51
            global_bldmode = "FAINT"
    #UT settings
    elif mode == 'ut':
    
    # Detector settings based on faintest star
    
        if faintest_k < 5: #Bright
            global_dit = "0.0008"
            global_gain = "20"
            global_nworeset = "1"
            global_bldmode = "STANDARD"
    
        elif 5.0 <= faintest_k <= 8.0:
            global_dit = "0.0004"
            global_gain = "30"
            global_nworeset = "21"
            global_bldmode = "FAINT"
        
        else:               #Faint
            global_dit = "0.0004"
            global_gain = "30"
            global_nworeset = "51"
            global_bldmode = "FAINT"
        
    if manual_bldmode:
        print(f"** Manual Override Triggered: BLDMODE changed from '{global_bldmode}' to '{manual_bldmode}' **")
        global_bldmode = manual_bldmode
        
    print(f"Applying Global Detector Settings  -> DIT: {global_dit}, GAIN: {global_gain}, NWORESET: {global_nworeset}, BLDMODE: {global_bldmode}\n")
    
    for target in all_targets_data:
        target["DIT"] = global_dit
        target["GAIN"] = global_gain
        target["NWORESET"] = global_nworeset
        target["BLDMODE"] = global_bldmode
    
    filename_targets = [name.replace("/", "_").replace("\\", "_") for name in target_names]
    base_filename = "__".join(filename_targets) + f"_{mode.upper()}.obd"
    
    if label:
        filename_label = label.replace("/", "_").replace("\\", "_")
        output_filename = Path.cwd() / f"{filename_label}__{base_filename}"
    else:
        output_filename = Path.cwd() / base_filename
        
    create_ob(all_targets_data, output_filename, args.template)


if __name__ == "__main__":
    main()
     
