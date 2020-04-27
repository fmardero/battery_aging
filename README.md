# NASA Li-ion Battery Aging Datasets
Dataset [here](https://c3.nasa.gov/dashlink/resources/133/)

**NOTE**:  
datasets can be transformed, once a time, to `pd.DataFrame` using `to_df` function coded in `mat_2_DataFrame.ipynb`

## Features:
* type: 	operation  type, can be charge, discharge or impedance
* ambient_temperature:	ambient temperature (degree C)
* time: 	the date and time of the start of the cycle, in MATLAB  date vector format
* for charge the fields are:
    1. Voltage_measured: 	Battery terminal voltage (Volts)
    2. Current_measured:	Battery output current (Amps)
    3. Temperature_measured: 	Battery temperature (degree C)
    4. Current_charge:		Current measured at charger (Amps)
    5. Voltage_charge:		Voltage measured at charger (Volts)
    6. Time:			Time vector for the cycle (secs)
* for discharge the fields are:
    1. Voltage_measured: 	Battery terminal voltage (Volts)
    2. Current_measured:	Battery output current (Amps)
    3. Temperature_measured: 	Battery temperature (degree C)
    4. Current_charge:		Current measured at load (Amps)
    5. Voltage_charge:		Voltage measured at load (Volts)
    6. Time:			Time vector for the cycle (secs)
    7. Capacity:		Battery capacity (Ahr) for discharge till 2.7V 
* for impedance the fields are:
    1. Sense_current:		Current in sense branch (Amps)
    2. Battery_current:	Current in battery branch (Amps)
    3. Current_ratio:		Ratio of the above currents 
    4. Battery_impedance:	Battery impedance (Ohms) computed from raw data
    5. Rectified_impedance:	Calibrated and smoothed battery impedance (Ohms) 
    6. Re:			Estimated electrolyte resistance (Ohms)
    7. Rct:			Estimated charge transfer resistance (Ohms)
