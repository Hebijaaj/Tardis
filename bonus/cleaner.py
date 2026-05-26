## import packages
import pandas as pd
import numpy as np
from datetime import datetime


def normalize_date(s):
    if not isinstance(s, str):
        return np.nan

    s = s.strip()
    # normalize separators
    for i in range(len(s)):
        if not (s[i].isdigit()):
            s = s[0:i] + "-" + s[i + 1 :]
    parts = s.split("-")

    if len(parts) != 2:
        return np.nan

    # detect format:
    # YYYY-(M)M OR (M)M-YYYY
    if len(parts[0]) == 4:
        y, m = parts
    elif len(parts[1]) == 4:
        m, y = parts
    else:
        return np.nan

    # numeric conversion
    if not (y.isdigit() and m.isdigit()):
        return np.nan

    y, m = int(y), int(m)

    # validate month strictly
    if not (1 <= m <= 12):
        return np.nan

    # normalize using calendar
    try:
        base = datetime(y, m, 1)
    except ValueError:
        return np.nan

    return base.strftime("%Y-%b")


def normalize_journey_time(s):
    if isinstance(s, float) or isinstance(s, int):
        return float(s)
    if not isinstance(s, str):
        return np.nan

    cpy = s[:]
    s = s.strip()
    s = s.replace(",", ".")
    pos = 0
    for i in s:
        if i == " ":
            break
        pos += 1
    if pos != 0:
        s = s[0:pos]
    else:
        s = "0"
    try:
        float(s)
    except ValueError:
        print(f'"{cpy}" is not a valid float')
        return np.nan
    return s


def normalize_integers(s):
    if isinstance(s, float) or isinstance(s, int):
        return int(s)
    if not isinstance(s, str):
        return np.nan

    s = s.strip()
    s = s.replace(",", ".")
    if len(s) != 0 and s[0] == "-":
        s = s[1:]
    var = 0
    for i in s:
        if i == " ":
            break
        if not i.isdigit():
            var += 1
    if var > 1:
        return np.nan
    pos = 0
    for i in s:
        if i == " " or i == ".":
            break
        pos += 1
    if pos != 0:
        s = s[0:pos]
    else:
        s = ""
    try:
        int(s)
    except ValueError:
        return np.nan
    return s


def normalize_numbers(s):
    if isinstance(s, float) or isinstance(s, int):
        return float(s)
    if not isinstance(s, str):
        return np.nan

    if "," in s:
        s = s.replace(",", ".")
    if "%" not in s:
        try:
            float(s)
        except ValueError:
            return np.nan
        return s
    if s[-1] != "%" or ("%" in s[0:-1]):
        return np.nan
    s = s[:-1]
    if len(s) != 0 and s[0] == "-":
        s = s[1:]
    try:
        float(s)
    except ValueError:
        return np.nan
    return s


def clean_dataset(df: pd.DataFrame):
    # Ignore these
    Comments = [
        "commentaire_annulation",
        "commentaire_retards_depart",
        "commentaires_retard_arrivee",
    ]

    # Integer values, such as "19", " 19.0 ", etc.
    Integer_values = [
        "nb_train_prevu",
        "nb_annulation",
        "nb_train_depart_retard",
        "nb_train_retard_arrivee",
        "nb_train_retard_sup_15",
        "nb_train_retard_sup_30",
        "nb_train_retard_sup_60",
    ]

    # Floating point values, such as 0.0, " 185.0 ", etc.
    Float_values = [
        "retard_moyen_depart",
        "retard_moyen_tous_trains_depart",
        "retard_moyen_arrivee",
        "retard_moyen_tous_trains_arrivee",
        "retard_moyen_trains_retard_sup15",
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]

    percentages = [
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]
    df["date"] = df["date"].apply(normalize_date)
    df["date"] = pd.to_datetime(
        df["date"], format="%Y-%b", errors="coerce"
    ).dt.strftime("%Y-%m")

    # then, if a date is nan, check the upper and lower values, and if they are the same, sets this value to this
    mask = df["date"].isna() & (df["date"].shift(1) == df["date"].shift(-1))
    df.loc[mask, "date"] = df["date"].shift(1)

    df["duree_moyenne"] = df["duree_moyenne"].apply(normalize_journey_time)
    df["duree_moyenne"] = pd.to_numeric(df["duree_moyenne"], errors="coerce")

    # remove comments
    for i in Comments:
        df = df.drop(columns=i)

    # parse integer values as ints
    for i in Integer_values:
        df[i] = df[i].apply(normalize_integers)
        df[i] = pd.to_numeric(df[i], errors="coerce")

    # parse floating point values as floats
    for i in Float_values:
        df[i] = df[i].apply(normalize_numbers)
        df[i] = pd.to_numeric(df[i], errors="coerce")

    # get the sum of all percentages in each rows
    row_sum = df[percentages].sum(axis=1, skipna=True)

    # get the number of missing values in each rows
    missing_count = df[percentages].isna().sum(axis=1)

    # mask of values to fill (there needs to be at least one missing value, and sum must be less than or equal to 100)
    mask = (missing_count > 0) & (row_sum <= 100)

    # amount to fill per missing value
    fill_value = (100 - row_sum) / missing_count

    for col in percentages:
        df.loc[mask, col] = df[col].where(df[col].notna(), round(fill_value, 7))

    ## Fill "very close" values with 0
    # re-get sum
    row_sum = df[percentages].sum(axis=1, skipna=True)

    # find lines with missing values
    missing_count = df[percentages].isna().sum(axis=1)

    # apply kinder mask
    mask = (missing_count > 0) & (row_sum <= 100.1)

    for col in percentages:
        df.loc[mask, col] = df[col].where(df[col].notna(), 0)

    df["gare_depart"] = (
        df["gare_depart"]
        .str.upper()
        .str.replace(
            r"\s+", " ", regex=True
        )  # replace group of multiple spaces into one space
        .str.strip()
        .str.replace(
            r"\bSAINT\b", "ST", regex=True, case=True
        )  # \b is "word boundary" (must be full word) allows for SAINT to become ST, but not ASAINT for example
        .str.replace(
            r"\bST\.\b", "ST", regex=True, case=True
        )  # changes ST. into ST (JIC)
    )
    df["gare_depart"] = df["gare_depart"].mask(
        df["gare_depart"].str.contains(r"\d", na=False)
    )

    df["gare_arrivee"] = (
        df["gare_arrivee"]
        .str.upper()
        .str.replace(
            r"\s+", " ", regex=True
        )  # replace group of multiple spaces into one space
        .str.strip()
        .str.replace(
            r"\bSAINT\b", "ST", regex=True, case=True
        )  # \b is "word boundary" (must be full word) allows for SAINT to become ST, but not ASAINT for example
        .str.replace(
            r"\bST\.\b", "ST", regex=True, case=True
        )  # changes ST. into ST (JIC)
    )
    df["gare_arrivee"] = df["gare_arrivee"].mask(
        df["gare_arrivee"].str.contains(r"\d", na=False)
    )

    cols = ["gare_depart", "gare_arrivee"]

    df["service"] = df.groupby(cols)["service"].transform(lambda s: s.ffill().bfill())

    # get all cities certified "National"
    national_cities = set(
        df.loc[df["service"] == "National", ["gare_depart", "gare_arrivee"]]
        .stack()
        .dropna()
        .unique()
    )

    # get all missing values and Internationnal values (if they've been mistakenly set to internationnal)
    missing = df["service"].isna() | (df["service"] == "International")

    # array of lines where the service is National (both cities in the array)
    both_national = df["gare_depart"].isin(national_cities) & df["gare_arrivee"].isin(
        national_cities
    )

    # sets the values for missing values
    df.loc[missing, "service"] = np.where(
        both_national[missing], "National", "International"
    )

    df = df.dropna(subset=["date", "gare_depart", "gare_arrivee"])

    df = df.groupby(
        ["date", "service", "gare_depart", "gare_arrivee"],
        sort=False,
        as_index=False,
    ).first()

    # all num values
    Integer_values = Integer_values + ["duree_moyenne"]

    # get months
    df["date_"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date_"].dt.month

    # try a subset with months
    group_month = ["gare_depart", "gare_arrivee", "month"]
    group_trajet = ["gare_depart", "gare_arrivee"]

    mean_month = df.groupby(group_month)[Integer_values].transform("mean")
    mean_route = df.groupby(group_trajet)[Integer_values].transform("mean")
    global_mean = df[Integer_values].mean()

    df[Integer_values] = df[Integer_values].fillna(round(mean_month, 0))
    df[Integer_values] = df[Integer_values].fillna(round(mean_route, 0))
    df[Integer_values] = df[Integer_values].fillna(round(global_mean, 0))

    mean_month = df.groupby(group_month)[Float_values].transform("mean")
    mean_route = df.groupby(group_trajet)[Float_values].transform("mean")
    global_mean = df[Float_values].mean()

    df[Float_values] = df[Float_values].fillna(mean_month)
    df[Float_values] = df[Float_values].fillna(mean_route)
    df[Float_values] = df[Float_values].fillna(global_mean)

    df = df.drop(columns="date_")
    df = df.drop(columns="month")

    return df
