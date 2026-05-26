# Tardis

Simple app to see SNCF train lateness

## Quick Start

### 1 install requirements

Tardis requires python3

You then need to install the requirements:

```bash
pip install -r requirements.txt
```

>[!TIP]
> If that doesn't work, it probably means you are in a controlled environement.
> In that case, you need to install venv, and then do `python3 -m venv .venv`

### 2 run the App

the app is run using streamlit

```bash
streamlit run tardis_dashboard.py
```

## Features

Tardis has a lot of features, separated by it's screens

>[!NOTE]
> If we talk about delay, it will always be about delay at **arival**

### Delay distribution screen

This screen contains a distribution of the delay of trains

It may contain other charts in the future

### Summary statistics screen

This screen contains some statistics about the trains.

Current statistics shown:

- Number of train trips gotten for our predictions
- Ponctuality rate of the trains (not like the real data from the SNCF, it counts 14 minutes delay as "on time")
- Average delay for all trains

### Prediction interface screen

This screen contains three selection menus, in which you may select a departure station, an arrival station, and a month of trip, and we will calculate some data for you

Current data predicted:

- Average delay of the train

## Bonus

There is a bonus for Tardis, located in the bonus directory

The bonus fetches the data directly from the SNCF, and retrains the model using this new data.

The data is only fetched once a month, or once there's more data on the sncf site than what has been fetched last time. This data is stored using a cache file **.tardis_cache**

To run the bonus, simply just execute this command:

```bash
streamlit run bonus/tardis_dashboard.py
```

> [!IMPORTANT]
> The requirements must also be installed, like with the dashboard at root
