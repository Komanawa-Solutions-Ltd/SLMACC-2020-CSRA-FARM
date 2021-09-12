# SLMACC-2020-CSRA
Repository for the climate components of the 2020 Climate-Shock Resilience and Adaptation

This branch (Normal Dist) marks the transition from a farmer defined event definitions to a 
normal distribution event definitions, which as of 2021-09-13 was the main branch.

# overall script for re-running definitions plus
* Climate_Shocks/make_new_event_definition_data.py

# scripts for:
* **IID**: BS_work/IID/IID.py 
* **SWG**: Storylines/storyline_runs/run_SWG_for_all_months.py
* **SIRG**: Climate_Shocks/Stochastic_Weather_Generator/irrigation_generator.py
* **Pasture growth model**: Pasture_Growth_Modelling/full_model_implementation.py
* **pasture growth model multi processing**: Pasture_Growth_Modelling/full_pgr_model_mp.py
* **pasture growth calculation**: Pasture_Growth_Modelling/calculate_pasture_growth.py

# scripts for basgra parameter sets:
* Pasture_Growth_Modelling/basgra_parameter_sets.py
* Pasture_Growth_Modelling/storage_parameter_sets.py

# scripts for dryland calibration:
* Pasture_Growth_Modelling/initialisation_support/validate_dryland_v1.py
* Pasture_Growth_Modelling/initialisation_support/validate_dryland_v2.py
* Pasture_Growth_Modelling/initialisation_support/validate_dryland_v3.py

# scrips for monthly bias correction for Farmmax (correction values provided by WS):
* Storylines/storyline_evaluation/transition_to_fraction.py

# scripts for cumulative probability processing
* Storylines/storyline_evaluation/storyline_eval_support.py

# scripts for running the PGR model suites:
* Storylines/storyline_runs/historical_quantified_1yr_detrend.py
* Storylines\storyline_runs\historical_quantified_1yr_trend.py
* Storylines/storyline_runs/run_random_suite.py
* Storylines/storyline_runs/run_nyr_suite.py

# script for 'most probable data'
* Storylines/storyline_evaluation/transition_to_fraction.py

# scripts for exporting plots and model data
* Storylines/storyline_evaluation/plot_historical_detrended.py
* Storylines/storyline_evaluation/export_cum_percentile.py
* Storylines/storyline_evaluation/plot_cumulative_historical_v_modelled.py
* Storylines/storyline_evaluation/plot_historical_trended.py
* Storylines\storyline_evaluation\plot_site_v_site.py
* Storylines/storyline_evaluation/plots.py
  
# script for exporting final storylines to Water Strategies
* Storylines/storyline_evaluation/storyline_slection/stories_to_ws_pg_threshold.py
