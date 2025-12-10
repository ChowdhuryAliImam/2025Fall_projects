import pandas as pd
import matplotlib.pyplot as plt



def plot_validation_occupancy_bootstrap(csv_path="validation_occupancy_bootstrap_all_rooms.csv"):
    df = pd.read_csv(csv_path)

    rooms = df["room"].unique()
    n_rooms = len(rooms)

    fig, axes = plt.subplots(
        n_rooms, 1,
        figsize=(7, 4 * n_rooms),
        sharex=True
    )

    if n_rooms == 1:
        axes = [axes]

    for ax, room in zip(axes, rooms):
        sub = df[df["room"] == room]

        orig = sub[sub["source"] == "original"]["daily_mean_occupancy"]
        boot = sub[sub["source"] == "bootstrap"]["daily_mean_occupancy"]

        bins = 20

        ax.hist(orig, bins=bins, alpha=0.6, label="Original days")
        ax.hist(boot, bins=bins, alpha=0.6, label="Bootstrapped days")

        ax.set_title(f"Occupancy validation – {room}")
        ax.set_xlabel("Daily mean occupancy")
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.tight_layout()
    plt.show()

def plot_validation_co2_timeseries(
    csv_path="validation_co2_timeseries_all_rooms.csv",
    room="Room1",
    day_id=None,
):
    """
    Plot measured vs simulated CO2 for a given room/day.

    If day_id is None, use the first available DayID for that room.
    """
    df = pd.read_csv(csv_path)

    sub_room = df[df["room"] == room]
    if sub_room.empty:
        raise ValueError(f"No CO2 validation data found for room={room!r}")

    if day_id is None:
        day_id = int(sub_room["DayID"].iloc[0])

    sub = sub_room[sub_room["DayID"] == day_id].copy()
    if sub.empty:
        raise ValueError(f"No CO2 validation data for room={room}, DayID={day_id}")

    sub = sub.sort_values("slot")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot simulated CO2
    ax.plot(sub["slot"], sub["CO2_sim"], label="Simulated CO₂", linewidth=2)

    # Plot measured CO2 if present
    if "CO2" in sub.columns and not sub["CO2"].isna().all():
        ax.plot(sub["slot"], sub["CO2"], label="Measured CO₂", linestyle="--", linewidth=1.5)

    ax.set_title(f"CO₂ validation – {room}, DayID={day_id}")
    ax.set_xlabel("Time slot (15-min index)")
    ax.set_ylabel("CO₂ [ppm]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

def plot_h1_per_capita(
    fixed_csv="h1_per_capita_fixed.csv",
    occ_csv="h1_per_capita_occ_based.csv",
):
    df_fixed = pd.read_csv(fixed_csv)
    df_fixed["strategy"] = "fixed"  # control group

    df_occ = pd.read_csv(occ_csv)
    df_occ["strategy"] = "occ_based"  # uncontrolled / adaptive group

    df = pd.concat([df_fixed, df_occ], ignore_index=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for strategy, sub in df.groupby("strategy"):
        label = "Control (Fixed)" if strategy == "fixed" else "Uncontrolled (Occ-based)"
        ax.scatter(
            sub["mean_occupancy"],
            sub["E_total_per_capita"],
            label=label,
            alpha=0.7,
        )

    ax.set_title("H1 – Per-capita energy vs mean occupancy")
    ax.set_xlabel("Mean occupancy")
    ax.set_ylabel("Per-capita energy [kWh/person]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

def plot_h2_ventilation(
    csv_path="h2_ventilation_comparison.csv",
):
    df = pd.read_csv(csv_path)

    # Make sure strategy is a nice string
    df["strategy"] = df["strategy"].astype(str)

    # Map labels for clarity
    strategy_labels = {
        "fixed": "Control (Fixed)",
        "occ_based": "Uncontrolled (Occ-based)",
    }

    # Reorder for consistent plotting
    strategies = ["fixed", "occ_based"]
    df["strategy_plot"] = df["strategy"].map(strategy_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Boxplot: CO2_exceed_hours ---
    data_co2 = [df[df["strategy"] == s]["CO2_exceed_hours"].dropna() for s in strategies]
    axes[0].boxplot(data_co2, labels=[strategy_labels[s] for s in strategies])
    axes[0].set_title("H2 – CO₂ exceed hours by strategy")
    axes[0].set_ylabel("CO₂ exceed hours [h]")

    # --- Boxplot: E_total_kWh ---
    data_E = [df[df["strategy"] == s]["E_total_kWh"].dropna() for s in strategies]
    axes[1].boxplot(data_E, labels=[strategy_labels[s] for s in strategies])
    axes[1].set_title("H2 – Total energy by strategy")
    axes[1].set_ylabel("Total energy [kWh]")

    fig.tight_layout()
    plt.show()

    # --- Scatter: CO2_exceed_hours vs E_total_kWh ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    for s in strategies:
        sub = df[df["strategy"] == s]
        label = strategy_labels[s]
        ax2.scatter(
            sub["E_total_kWh"],
            sub["CO2_exceed_hours"],
            alpha=0.7,
            label=label,
        )

    ax2.set_title("H2 – CO₂ exceed vs energy (control vs uncontrolled)")
    ax2.set_xlabel("Total energy [kWh]")
    ax2.set_ylabel("CO₂ exceed hours [h]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    plt.show()
