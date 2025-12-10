import matplotlib.pyplot as plt
from plots import (  # or just paste the functions directly
    plot_validation_occupancy_bootstrap,
    plot_validation_co2_timeseries,
    plot_h1_per_capita,
    plot_h2_ventilation,
)

if __name__ == "__main__":
    plot_validation_occupancy_bootstrap()
    plot_validation_co2_timeseries(room="Room1", day_id=23)
    plot_h1_per_capita()
    plot_h2_ventilation()
