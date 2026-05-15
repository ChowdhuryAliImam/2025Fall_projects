Title: Monte Carlo Analysis of Occupancy-Driven Energy and CO₂ Variability in a Public Building
-----------------------------------------------------------------------------------------------

Brief:
Under the uncertainty of occupant behavior, its effect on energy demand and indoor carbon dioxide levels are simulated using Monte Carlo Simulation.
For simulation, I didn't use any simulation engine, physics based simulation was used using formulas.

------------------------------------------------------------------------------------------------
Data:
Data Source: 
[1]Jens Hjort Schwee, Aslak Johansen, Bo Nørregaard Jørgensen, Mikkel Baun Kjærgaard, Claudio Giovanni Mattera, Fisayo Caleb Sangogboye, and Christian Veje. 2019. Room-level occupant counts using heterogeneous sensing modalities from a teaching and office building. DOI:https://doi.org/10.6084/m9.figshare.c.4505813.v1

[https://doi.org/10.1038/s41597-019-0274-4](https://doi.org/10.6084/m9.figshare.c.4505813)
--------------------------------------------------------------------------------------------------
Validation:
The sampling is validated by comparing the sampled occupancy distribution against the original dataset.
The measured carbon-di-oxide concentration is also compared with the original dataset

----------------------------------------------------------------------------------------------------
Hypothesis:
1. As average occupancy increases, the per-capita energy consumption decreases

2. Compared to fixed ventilation, occupancy-based ventilation will reduce CO₂ exceedance hours and energy use.

-----------------------------------------------------------------------------------------------------
RUN the SIMULATION
download Data, params, all_functions and main. run main.

------------------------------------------------------------------------------------------------------
result: 
Hypothesis 1: Increased mean occupancy decreased the per capita energy consumption:
<img width="1000" height="750" alt="image" src="https://github.com/user-attachments/assets/9d35e1fc-871e-403e-9113-e7d95b40770b" />
Hypothesis 2:Neither of the strategy increase carbon-di-oxide exceedance hours but occupancy based ventilation reduced energy use
<img width="833" height="625" alt="image" src="https://github.com/user-attachments/assets/2a672c0d-46b7-4f07-9a85-1b57e08eeaea" />
<img width="833" height="625" alt="image" src="https://github.com/user-attachments/assets/e1fa067f-a365-4d6c-9d3d-8fb2c1b2859b" />

