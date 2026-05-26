import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy
import fetcher


fetcher.auto_refetch()

model: dict[str, Pipeline] = joblib.load("bonus/model.pkl")

temp_df = pd.read_csv("cleaned_dataset.csv")

arrival_stations = numpy.array(temp_df["gare_arrivee"].dropna().unique())
departure_stations = numpy.array(temp_df["gare_depart"].dropna().unique())


def predict_data(departure: str, arrival: str, month: int) -> list[float] | None:

    if not isinstance(month, int) or (month <= 0 or month > 12):
        return None

    if (
        not isinstance(departure, str)
        or not isinstance(arrival, str)
        or arrival == departure
        or arrival not in arrival_stations
        or departure not in departure_stations
    ):
        return None

    data = {
        "gare_depart": [departure],
        "gare_arrivee": [arrival],
        "month": [month],
    }

    x_test = pd.DataFrame(data)

    found = model["delay"].predict(x_test)
    x_test = pd.DataFrame(data)
    found2 = model["cancel"].predict(x_test)
    returned = [found[0], found2[0]]
    return returned
