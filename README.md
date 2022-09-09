# harvey_scaling


This repository contains scripts and data to reproduce results published in "Middelanis et al. (2022) Economic losses from hurricanes cannot be nationally offset under unabated warming. Environmental Research Letters".

1. affected counties and resulting initial forcing parameters for Hurricane Harvey are calculated using ./scripts/calc_initial_forcing_intensity_HARVEY.py
2. main simulation runs are generated using ./scripts/generate_forcing_ensemble.py and started using the scripts ./data/acclimate_forcing/main_analysis/orchestrate_ensemble.sh
3. output from main simulations is aggregated for further processing using ./scripts/aggregate_acclimate_outputs.py
4. sensitivity analysis simulations are generated and started using ./scripts/generate_sensitivity_ensemble.py
5. figures are generated using ./scripts/plotting.py
