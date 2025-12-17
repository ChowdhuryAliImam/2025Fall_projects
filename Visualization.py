import pandas as pd
import matplotlib.pyplot as plt

# H1: baseline vs treatment

baseline = pd.read_csv("H1_baseline_building.csv")
treatment = pd.read_csv("H1_treatment_building.csv")

# per-capita energy
baseline = baseline[baseline["mean_occupancy"] > 0].copy()
treatment = treatment[treatment["mean_occupancy"] > 0].copy()


h1 = baseline['E_per_capita'].copy()
h12 = treatment['E_per_capita'].copy()


plt.figure()
plt.plot(h1, marker="o", linestyle="-", label="baseline")
plt.plot(h12, marker="o", linestyle="-", label="treatment")
plt.xlabel("Monte Carlo run")
plt.ylabel("Per-capita energy [kWh/person]")
plt.title("H1: Per-capita energy (Baseline vs Increased-Mean Occupancy)")
plt.legend()


# H2: fixed vs occ_based

h2 = pd.read_csv("H2_building_totals.csv")

fixed = h2[h2["strategy"] == "fixed"].copy()
occ = h2[h2["strategy"] == "occ_based"].copy()

# align by run
h2m = fixed[["run", "E_total_kWh", "CO2_exceed_hours"]].merge(
    occ[["run", "E_total_kWh", "CO2_exceed_hours"]],
    on="run",
    suffixes=("_fixed", "_occ_based"),
    how="inner",
)

plt.figure()
plt.plot(h2m["run"], h2m["E_total_kWh_fixed"], marker="o", linestyle="-", label="fixed")
plt.plot(h2m["run"], h2m["E_total_kWh_occ_based"], marker="o", linestyle="-", label="occ_based")
plt.xlabel("Monte Carlo run")
plt.ylabel("Total energy [kWh]")
plt.title("H2: Total energy (Fixed vs Occupancy-based ventilation)")
plt.legend()

plt.figure()
plt.plot(h2m["run"], h2m["CO2_exceed_hours_fixed"], marker="o", linestyle="-", label="fixed")
plt.plot(h2m["run"], h2m["CO2_exceed_hours_occ_based"], marker="o", linestyle="-", label="occ_based")
plt.xlabel("Monte Carlo run")
plt.ylabel("CO₂ exceed hours [h]")
plt.title("H2: CO₂ exceedance (Fixed vs Occupancy-based ventilation)")
plt.legend()

plt.show()

if __name__ == "__main__":
    main()

#AI Disclosure: ChatGPT assistance was used for debugging and code suggestions in this project. Google AI studio was used for doctest help.