

# Simulation config (shared for all rooms) 
SIM_CONFIG = {
    "dt_seconds": 900,        
    "co2_threshold": 1000.0,  
    "n_days_per_run": 44,
    "n_runs": 100,
}

# Building parameters per room (as dicts)
BUILDING_PARAMS = {
    "Room1": {
        "V": 200.0,          
        "UA": 150.0,         
        "T_set": 22.0,       
        "eta_heat": 0.95,
        "COP_cool": 3.0,
        "q_person": 75.0,    
        "G_CO2": 0.000005,   
        "C_out": 420.0,      
        "P_plug": 500.0,     
        "P_light": 400.0,    
    },
    "Room2": {
        "V": 150.0,
        "UA": 120.0,
        "T_set": 22.0,
        "eta_heat": 0.95,
        "COP_cool": 3.0,
        "q_person": 75.0,
        "G_CO2": 0.000005,
        "C_out": 420.0,
        "P_plug": 400.0,
        "P_light": 300.0,
    },
    "Room3": {
        "V": 180.0,
        "UA": 130.0,
        "T_set": 22.0,
        "eta_heat": 0.95,
        "COP_cool": 3.0,
        "q_person": 75.0,
        "G_CO2": 0.000005,
        "C_out": 420.0,
        "P_plug": 450.0,
        "P_light": 350.0,
    },
}

# Ventilation parameters per room
VENTILATION_PARAMS = {
    "Room1": {
        "Vdot_fixed": 0.2,        
        "Vdot_base": 0.05,        
        "Vdot_per_person": 0.01,   
    },
    "Room2": {
        "Vdot_fixed": 0.18,
        "Vdot_base": 0.05,
        "Vdot_per_person": 0.01,
    },
    "Room3": {
        "Vdot_fixed": 0.22,
        "Vdot_base": 0.06,
        "Vdot_per_person": 0.012,
    },
}

ROOM_FILE_GROUPS = {
    "Room1": {
        "occupancy": r"Data\occupant_count_room_1.csv", 
        "co2":       r"Data\co2_room_1.csv",       
        "damper":    r"Data\vav_room_1.csv",    
    },
    "Room2": {
        "occupancy": r"Data\occupant_count_room_2.csv",
        "co2":       r"Data\co2_room_2.csv",
        "damper":    r"Data\vav_room_2.csv",
    },
    "Room3": {
        "occupancy": r"Data\occupant_count_room_3.csv",
        "co2":       r"Data\co2_room_3.csv",
        "damper":    r"Data\vav_room_3.csv",
    },
}

# Weather file
WEATHER_FILE = r"Data\copenhagen_parsed.csv"
