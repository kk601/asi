# ASI projekt

Projekt zaliczeniowy dla przedmiotu MLOps oparty na zbiorze **Airline Passenger Satisfaction**.

## Wymagania dla całości projektu

1. Zainstalowany Python > 3.0
2. Wykonanie `python -m pip install -r requirements.txt`
3. Pobrany dataset [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction)

## Sprint 1 - Notebook + SQLite + EDA + baseline model

### Zakres sprintu 1:
- import danych CSV do SQLite
- notebook EDA oparty o SQLite
- baseline model klasyfikacyjny
- zapis metryk do JSON

### Uruchomienie:

1. Zaladuj dane do SQLite:

```
python scripts/load_airline_data_to_sqlite.py --source "<ścieżka do pobranego dataset>.csv"
```

2. Wylicz baseline i zapisz metryki:

```
python scripts/run_sprint1_baseline.py
```

3. Otworz notebook:

```
jupyter notebook notebooks/01_eda.ipynb
```

## Sprint 2 – Kedro pipeline

### Zakres sprintu 2:

W ramach Sprintu 2 projekt został rozszerzony o pipeline w frameworku Kedro do przetwarzania danych i trenowania modelu klasyfikacyjnego przewidującego satysfakcję pasażerów.

Pipeline obejmuje:
- przetwarzanie danych
- podział na train / validation / test
- trenowanie modelu RandomForest
- ewaluację modelu

### Uruchomienie:
```
kedro run
```

Domyslne sciezki:
- baza SQLite: `data/01_raw/dataset.db`
- metryki: `reports/metrics/baseline_metrics.json`

Zmienne srodowiskowe sa opisane w `.env.example`.
