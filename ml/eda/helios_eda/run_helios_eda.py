# ml/eda/helios_eda/run_helios_eda.py 
""" 
ml/eda/helios_eda/run_helios_eda.py 
===================================== 
Backward-compatible wrapper around the unified run_eda pipeline. 

All HEL1OS EDA logic now lives in ml/eda/run_eda.py. 
This module delegates to it so that existing commands continue to work: 

    python -m ml.eda.helios_eda.run_helios_eda --date 20240611 
    python -m ml.eda.helios_eda.run_helios_eda --date 20240611 --detector ALL 
    python -m ml.eda.helios_eda.run_helios_eda --date 20240611 --all-detectors 
""" 

from __future__ import annotations 

import argparse 
import sys 

_ALL_DETECTORS = ["CdTe1", "CdTe2", "CZT1", "CZT2"] 

def _build_arg_parser() -> argparse.ArgumentParser: 
    p = argparse.ArgumentParser( 
        description="Aditya-L1 HEL1OS EDA pipeline (delegates to run_eda)", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
    ) 
    p.add_argument( 
        "--date", required=True, 
        help="Observation date YYYYMMDD, e.g. 20240611", 
    ) 
    p.add_argument( 
        "--detector", default="ALL", 
        choices=_ALL_DETECTORS + ["ALL"], 
        help="HEL1OS detector. ALL runs every detector.", 
    ) 
    p.add_argument( 
        "--all-detectors", action="store_true", 
        help="Alias for --detector ALL.", 
    ) 
    p.add_argument("--config",      default=None) 
    p.add_argument("--show",        action="store_true") 
    p.add_argument("--onset-sigma", type=float, default=3.0) 
    return p 

def main() -> None: 
    """ 
    Translate the HEL1OS-specific CLI into the unified run_eda CLI and delegate. 
    No EDA logic is duplicated here. 
    """ 
    from ml.eda.run_eda import run_eda, _build_arg_parser as _unified_parser 

    local_args = _build_arg_parser().parse_args() 

    # Build a Namespace that run_eda expects 
    unified = argparse.Namespace( 
        date            = local_args.date, 
        all_days        = False, 
        detector        = "SDD2",            # SoLEXS not run from this entry point 
        helios_detector = ( 
            "ALL" 
            if (local_args.all_detectors or local_args.detector == "ALL") 
            else local_args.detector 
        ), 
        with_helios     = True,              # HEL1OS is always the point here 
        config          = local_args.config, 
        show            = local_args.show, 
        onset_sigma     = local_args.onset_sigma, 
        debug           = False, 
    ) 

    run_eda(unified) 

if __name__ == "__main__": 
    main()