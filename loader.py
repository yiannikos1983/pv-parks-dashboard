import os
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))

PARKS = {
    "Mpountalas": {
        "file": os.path.join(_BASE, "MPOUNTALAS", "Mpountalas.csv"),
        "capacity_kw": 999.92,
        "edreth": 26714,
        "electrification_date": "08/11/2023",
    },
    "Agios Athanasios": {
        "file": os.path.join(_BASE, "AGIOS ATHANASIOS", "AGIOS ATHANASIOS.csv"),
        "capacity_kw": 1099.22,
        "edreth": 26713,
        "electrification_date": "07/03/2025",
    },
    "Aiginio": {
        "file": os.path.join(_BASE, "AIGINIO", "AIGINIO.csv"),
        "capacity_kw": 99.55,
        "edreth": 26199,
        "electrification_date": "20/07/2024",
    },
    "Deksameni Thourio": {
        "file": os.path.join(_BASE, "DEKSAMENI THOURIO", "DEKSAMENI THOURIO 26440.csv"),
        "capacity_kw": 99.96,
        "edreth": 26440,
        "electrification_date": "02/05/2024",
    },
    "Deksameni Arkates": {
        "file": os.path.join(_BASE, "DEKSAMENI ARKATES", "DEKSAMENI ARKATES.csv"),
        "capacity_kw": 99.96,
        "edreth": 26284,
        "electrification_date": "03/10/2024",
    },
    "Rousilakis Aspros": {
        "file": os.path.join(_BASE, "ROUSILAKIS ASPROS", "Rousilakis - Aspros.csv"),
        "capacity_kw": 99.63,
        "edreth": 26180,
        "electrification_date": "20/04/2023",
    },
}


def _parse_kwh(series: pd.Series) -> pd.Series:
    """Parse '5.00 kWh' / '1.00 Wh' strings to float kWh. Returns NaN for missing or -1 sentinel."""
    def _conv(s):
        if not isinstance(s, str) or not s.strip():
            return float("nan")
        parts = s.strip().split()
        try:
            val = float(parts[0])
        except (ValueError, IndexError):
            return float("nan")
        if val == -1.0:
            return float("nan")
        unit = parts[1].upper() if len(parts) > 1 else "KWH"
        return val / 1000.0 if unit == "WH" else val

    return series.map(_conv)


def _parse_eur(series: pd.Series) -> pd.Series:
    """Parse '€ 123.45' / '-€ 1.00' strings to float. Returns NaN for empty cells."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace("€", "", regex=False)
              .str.replace(r"\s+", "", regex=True),  # collapse whitespace so '-€ 1.00' -> '-1.00'
        errors="coerce",
    )


def load_all_parks() -> pd.DataFrame:
    """Load all 6 park CSVs and return a single cleaned DataFrame (~195 k rows)."""
    frames = []
    for park_name, meta in PARKS.items():
        raw = pd.read_csv(meta["file"], encoding="utf-8-sig")
        raw.columns = raw.columns.str.strip()

        d = pd.DataFrame()
        d["timestamp"] = pd.to_datetime(raw["Time"], dayfirst=True, errors="coerce")
        d["scada_kwh"] = _parse_kwh(raw["Μέτρηση SCADA"])
        d["admie_initial_kwh"] = _parse_kwh(raw["Αρχική Πιστοποιημένη Μέτρηση ΑΔΜΗΕ"])
        d["admie_corrective_kwh"] = _parse_kwh(raw["Διορθωτική Πιστοποιημένη Μέτρηση ΑΔΜΗΕ"])
        d["curtailed"] = (
            pd.to_numeric(raw["FORENA Curtailment suggestion"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        d["dam_price"] = _parse_eur(raw["DAM Price (EUR/MWh)"])
        d["payment_eur"] = _parse_eur(raw["Payment (EUR)"])

        # Canonical production: use corrective ADMIE if positive, else initial ADMIE
        corr = d["admie_corrective_kwh"].fillna(0.0)
        init = d["admie_initial_kwh"]
        d["production_kwh"] = corr.where(corr > 0, init)

        d["park_name"] = park_name
        d["capacity_kw"] = meta["capacity_kw"]
        frames.append(d)

    df = pd.concat(frames, ignore_index=True)
    return df.dropna(subset=["timestamp"])
