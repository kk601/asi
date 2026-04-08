ASI projekt

Projekt zaliczeniowy dla przedmiotu MLOps oparty na zbiorze `Airline Passenger Satisfaction`.

Zakres sprintu 1:
- import danych CSV do SQLite
- notebook EDA oparty o SQLite
- baseline model klasyfikacyjny
- zapis metryk do JSON

## Wymagania

```powershell
python -m pip install -r requirements.txt
```

## Sprint 1: uruchomienie

1. Zaladuj dane do SQLite:

```powershell
python scripts/load_airline_data_to_sqlite.py --source "airline_passenger_satisfaction.csv"
```

2. Wylicz baseline i zapisz metryki:

```powershell
python scripts/run_sprint1_baseline.py
```

3. Otworz notebook:

```powershell
jupyter notebook notebooks/01_eda.ipynb
```

## Sprint 2 – Kedro pipeline

W ramach Sprintu 2 projekt został rozszerzony o pipeline w frameworku Kedro do przetwarzania danych i trenowania modelu klasyfikacyjnego przewidującego satysfakcję pasażerów.

Pipeline obejmuje:
- przetwarzanie danych
- podział na train / validation / test
- trenowanie modelu RandomForest
- ewaluację modelu

### Uruchomienie
```bash
pip install -r requirements.txt
kedro run

Domyslne sciezki:
- baza SQLite: `data/01_raw/dataset.db`
- metryki: `reports/metrics/baseline_metrics.json`

Zmienne srodowiskowe sa opisane w `.env.example`.
