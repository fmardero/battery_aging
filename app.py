import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import altair as alt
import numpy as np
import time

plt.rcParams.update({'font.size': 22, 'lines.linewidth': 3})


@st.cache
def load_data():
    filename = 'discharge'
    df = pd.read_csv(f'{filename}.csv')
    return df

@st.cache(allow_output_mutation=True)
def load_model():
    filename = 'model'
    model = load(f'{filename}.joblib')
    return model

@st.cache
def load_features(df_discharge):
    df_new = pd.DataFrame()
    max_volt = {'B0005':2.7,'B0006':2.5,'B0007':2.2, 'B0018':2.5}

    for idcycle, battery in df_discharge[['id_cycle','Battery']].drop_duplicates().apply(tuple, axis=1):
        mask1 = df_discharge.id_cycle == idcycle
        mask2 = df_discharge.Battery == battery
        df_tmp = df_discharge[mask1 & mask2].sort_values('Time')
        t_0 = df_tmp['Time'].min()
        t_volt = df_tmp.loc[df_tmp['Voltage_measured'] <= max_volt[battery],'Time'].min()
        t_tmax = df_tmp.loc[df_tmp['Temperature_measured'] == df_tmp['Temperature_measured'].max(),'Time'].min()
        capacity = df_tmp['Capacity'].max()
        df_new = df_new.append(
            pd.DataFrame(
                {
                    'Battery': battery,
                    'id_cycle': idcycle,
                    'time_volt': t_volt - t_0,
                    'time_temp': t_tmax - t_0,
                    'capacity': capacity
                }, index = [0])
        )
    return df_new

# Data Loader
df = load_data()
batteries = sorted(df.Battery.unique())
palette = {battery: sns.color_palette('bright')[i] for i, battery in enumerate(batteries)}

# Dashboard title
st.title('NASA Li-ion Battery Aging Dataset')
st.write(
    'Dataset can be found <a href="https://c3.nasa.gov/dashlink/resources/133/">here</a>.', 
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title('Navigation Bar')
view = st.sidebar.radio(
    'Select one',
    ('Data Analysis', 'ML Model'),
    index=1
)
st.header(view)

# Display Page
if view == 'Data Analysis':
    st.subheader('Battery **State of Health**')
    st.latex(r'SOH = \frac{Q_{aged}}{Q_{rate}} \times 100')
    st.markdown(r'where $Q_{rate}$ is the rated capacity of the battery when it leaves the factory.')

    # Plot #1
    st.subheader('Capacity degradation curve')
    st.text('Acceptable Performance Threshold (APT) is set as 70% of rated capacity.')
    bat_selected = st.multiselect(
        'Battery',
        batteries,
        default=batteries
    )
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    for battery in bat_selected:
        mask = df.Battery == battery
        tmp_palette = palette[battery]
        df_tmp = df[mask]
        ax.plot(df_tmp['id_cycle'], df_tmp['Capacity'], color=tmp_palette)
    plt.axhline(y=1.4, color='k', lw=2, linestyle='dashed', label='APT')
    plt.xlabel('Charge and discharge cycles')
    plt.ylabel('Actual Capacity in Ah')
    plt.legend()
    #plt.title('Lithium-ion battery actual capacity degradation curve')
    st.pyplot(fig)

    # Plot #2
    st.subheader('Time necessary to reach the minimum voltage')
    option = st.selectbox(
        'Battery',
        batteries, 
        key='plot#2'
    )
    mask = df.Battery == option
    df_tmp = pd.pivot_table(
        data=df[mask],
        index='id_cycle',
        values=['Voltage_measured'],
        aggfunc='min'
    ).reset_index()
    df_tmp = df.merge(df_tmp, on=['id_cycle', 'Voltage_measured'])
    df_tmp = pd.pivot_table(
        data=df_tmp,
        index=['id_cycle', 'Voltage_measured'],
        values=['Time'],
        aggfunc='min'
    ).reset_index()

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    ax.plot(df_tmp['id_cycle'], df_tmp['Time'], color=palette[option])
    plt.xlabel('Cycles')
    plt.ylabel('Time/seconds')
    #plt.title('Time necessary to reach the min Voltage_measured')
    st.pyplot(fig)

    # Plot #3
    st.subheader('Voltage-Current trend during discharge')
    bat_opt = st.selectbox(
        'Battery',
        batteries,
        key='plot#3'
    )
    cycles = np.sort(df.loc[df.Battery == bat_opt, 'id_cycle'].unique())
    cycles = [int(v) for v in cycles]
    cycle_opt = {}
    cycle_opt[0] = st.slider(
        'Cycle 1',
        min(cycles),
        max(cycles),
        value=min(cycles),
        key='cycle#1'
    )
    cycle_opt[1] = st.slider(
        'Cycle 2',
        min(cycles),
        max(cycles),
        value=max(cycles),
        key='cycle#2'
    )
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.set_size_inches(12, 9)
    mask0 = (df.Battery == bat_opt) 
    for key, val in cycle_opt.items():
        mask1 = (df.id_cycle == val)
        ls = '-' if key == 0 else '--'
        ax[0].plot('Time', 'Voltage_charge', data=df[mask0 & mask1], color=palette[bat_opt], ls=ls, label=f'Cycle {val}')
        ax[1].plot('Time', 'Current_charge', data=df[mask0 & mask1], color=palette[bat_opt], ls=ls, label=f'Cycle {val}')
    plt.legend()

    ax[0].set_ylabel('Voltage')
    ax[1].set_ylabel('Current')
    ax[1].set_xlabel('Cycle')
    ax[1].set_ylim([1.88, 2.12])
    st.pyplot(fig)
elif view == 'ML Model':
    st.subheader('Random Forest Predictions')
    st.markdown('The machine learning model predicts the battery capacity (ageing) after the discarge cycle.')
    # Load model
    model = load_model()

    # Load Dataset
    df_new = load_features(df)

    option = st.selectbox(
        'Battery',
        batteries, 
        key='ML-model'
    )
    mask = df_new.Battery == option
    dataset = df_new[mask].sort_values('id_cycle')
    assert len(dataset) == df_new[mask].id_cycle.unique().shape[0], 'Wrong number of elements in the dataset.'

    # Feature pre-processing
    X_col = ['time_volt', 'time_temp']
    y_col = ['capacity']
    X = dataset[X_col]
    y_true = dataset[y_col]

    # Model predictions
    y_pred = model.predict(X)
    
    # Build results DataFrame
    results = pd.DataFrame()
    results['true'] = np.ravel(y_true)
    results['predicted'] = y_pred
    results.index = dataset.id_cycle

    # Display speed
    speed = st.slider(
        'Discharge cycles per second',
        1.0,
        20.0,
        value=10.0,
        step=1.0
    )
    sleeping_time = 1.0 / speed

    # Prapare to plot
    chart = st.empty()

    # Plot cycle by cycle
    for i in range(X.shape[0]):
        tmp = results.iloc[:i+1].reset_index().rename(columns={'id_cycle': 'Cycle'})
        print(tmp)
        tmp_true = tmp[['Cycle', 'true']].rename(columns={'true': 'Capacity'})
        tmp_true['type'] = 'Ground Truth'
        tmp_pred = tmp[['Cycle', 'predicted']].rename(columns={'predicted': 'Capacity'})
        tmp_pred['type'] = 'Predicted'

        data_to_be_added = pd.DataFrame()
        data_to_be_added = data_to_be_added.append(tmp_true)
        data_to_be_added = data_to_be_added.append(tmp_pred)
        
        x = alt.Chart(data_to_be_added).mark_line(size=5).encode(
            x='Cycle',
            y=alt.Y('Capacity', scale=alt.Scale(domain=(1.2, 2.2))),
            color='type'
        ).properties(
            width=800,
            height=400
        ).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        ) .interactive()

        chart.altair_chart(x)
        time.sleep(sleeping_time)
