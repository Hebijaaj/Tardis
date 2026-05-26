import requests
import pandas as pd
import sys
from datetime import datetime
import cleaner
import model

URL = (
    "https://data.sncf.com/api/explore/v2.1/catalog/datasets/"
    "regularite-mensuelle-tgv-aqst/records"
)


def fetch_sncf_by_year(data: list = None, start_year=2018):
    base_url = (
        "https://data.sncf.com/api/explore/v2.1/catalog/datasets/"
        "regularite-mensuelle-tgv-aqst/records"
    )

    current_year = datetime.now().year
    all_rows = []

    response = requests.get(base_url)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("Couldn't Connect to SNCF server", file=sys.stderr)
        return None

    if data is not None:
        data[0] = response.json()["total_count"]

    for year in range(start_year, current_year + 1):
        # one calendar year window
        where = f"date >= date'{year}-01-01' AND date < date'{year + 1}-01-01'"

        offset = 0
        limit = 100

        print(f"Fetching {year}...")

        while True:
            params = {
                "where": where,
                "limit": limit,
                "offset": offset,
                "lang": "fr",
                "timezone": "Europe/Paris",
            }

            response = requests.get(base_url, params=params)

            try:
                response.raise_for_status()

            except requests.HTTPError:
                print(
                    f"Data fetching stopped early in year {year} at offset={offset}",
                    sys.stderr,
                )
                break

            data = response.json()
            rows = data["results"]

            if not rows:
                break

            all_rows.extend(rows)
            offset += limit

    return pd.DataFrame(all_rows)


def get_file():
    try:
        f = open(".tardis_cache", "rb")
    except FileNotFoundError:
        try:
            f = open("bonus/.tardis_cache", "rb")
        except FileNotFoundError:
            return None
    return f


def should_refetch() -> bool:
    f = get_file()

    if f is None:
        return True

    a = int.from_bytes(f.read(4), byteorder="little", signed=True)
    b = int.from_bytes(f.read(4), byteorder="little", signed=True)
    c = int.from_bytes(f.read(4), byteorder="little", signed=True)

    f.close()

    current_time = datetime.now()

    if c != current_time.year and (b != current_time.month):
        return True
    response = requests.get(URL)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(
            "Error, failed to connect to the sncf server, will assume data is on time\n"
            "(data fetched this month)",
            file=sys.stderr,
        )
        return False
    data = response.json()
    if data["total_count"] != a:
        return True
    return False


def auto_refetch():
    data = [0]
    # 1. Check if we need to fetch
    if not should_refetch():
        return None
    # 2. Extract records into DataFrame
    df = fetch_sncf_by_year(data)
    # 3. Apply your cleaning pipeline
    df = cleaner.clean_dataset(df)
    # 4. Save cleaned CSV
    df.to_csv("bonus/cleaned_dataset.csv", index=False, encoding="utf-8")
    # 5. store data into cache file
    f = open(".tardis_cache", "wb+")
    f.write(data[0].to_bytes(4, byteorder="little", signed=True))
    f.write(datetime.now().month.to_bytes(4, byteorder="little", signed=True))
    f.write(datetime.now().year.to_bytes(4, byteorder="little", signed=True))
    f.close()
    # 6. Redo the model
    model.main()
    return None
