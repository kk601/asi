# ASI projekt

Projekt zaliczeniowy dla przedmiotu MLOps oparty na zbiorze **Airline Passenger Satisfaction**.

## Wymagania wstępne

1. Zainstalowany Python > 3.10
2. Wykonanie `python -m pip install -r requirements.txt`
3. Pobrany dataset [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction)

## Sprint 1 - Notebook + SQLite + EDA + baseline model

### Zakres sprintu 1:
- import danych CSV do SQLite
- notebook EDA oparty o SQLite
- baseline model klasyfikacyjny
- zapis metryk do JSON

### Uruchomienie:

1. Załaduj dane do SQLite:

```
python scripts/load_airline_data_to_sqlite.py --source "<ścieżka do pobranego dataset>.csv"
```

2. Wylicz baseline i zapisz metryki:

```
python scripts/run_sprint1_baseline.py
```

3. Otwórz notebook:

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

Dane wejściowe są ładowane przez Kedro Data Catalog (`airline_raw` →
`pandas.SQLTableDataset`) z bazy SQLite wskazanej w credentials.
Zmienne środowiskowe są opisane w `.env.example`.

## Sprint 3 – Weights & Biases

### Zakres sprintu 3:

Pipeline Kedro został zintegrowany z W&B i loguje parametry, metryki oraz artefakty przy każdym kedro run.

Dostarczono:
- W&B zintegrowane z pipeline'em
- Logowane: parametry (config), metryki (RMSE/MAE/R² lub accuracy/F1), artefakt modelu
- Plik .env z kluczem API, entity i nazwą projektu
- 5 runów z różnymi parametrami w parameters.yml
- Dashboard W&B z tabelą runów posortowaną po kluczowej metryce

### Uruchomienie:

1. Wymagane zmienne środowiskowe:
- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
2. Ustawienie konfiguracji pipeline w `conf/base/parameters.yml`
3. `kedro run`

## Sprint 4 – AutoML z AutoGluon

### Zakres sprintu 4:

Projekt został rozszerzony o osobny pipeline `automl`, który wykorzystuje AutoGluon do automatycznego trenowania i porównywania wielu modeli klasyfikacyjnych dla problemu przewidywania satysfakcji pasażerów.

Pipeline `automl` korzysta z danych przygotowanych przez istniejący pipeline `data_processing`, czyli z tych samych zbiorów `X_train`, `X_val`, `y_train`, `y_val`, które były używane dla modelu baseline.

Dostarczono:
- nowy pipeline Kedro `automl`
- trening modelu AutoGluon `TabularPredictor`
- konfigurację AutoGluon w `conf/base/parameters.yml`
- zapis predyktora AutoGluon do katalogu `data/06_models/autogluon/`
- logowanie metryk do Weights & Biases
- logowanie leaderboardu AutoGluon jako tabela W&B
- porównanie modelu baseline RandomForest z modelem AutoML
- możliwość uruchomienia całości przez pipeline `full`

### Konfiguracja AutoGluon:

Parametry AutoGluon znajdują się w pliku: `pip install -r requirements.txt`

### Uruchomienie:
1. Podstawowy pipeline : `kedro run`
2. Pipeline AutoML: `kedro run --pipeline=automl`
3. Pełny pipeline (opcjonalnie): `kedro run --pipeline=full`

## Sprint 5 – REST API z FastAPI

### Zakres sprintu 5:
W ramach Sprintu 5 projekt został rozszerzony o interfejs REST API, który pozwala na serwowanie wytrenowanego modelu klasyfikacyjnego i dokonywanie predykcji w czasie rzeczywistym przez sieć, bez konieczności uruchamiania całego środowiska Kedro.

Dostarczono:

- aplikację opartą o framework FastAPI (w katalogu api/)
- endpoint GET /health zwracający status serwera oraz informację o załadowanym modelu
- endpoint POST /predict przyjmujący dane o pasażerze w formacie JSON i zwracający wynik predykcji
- rygorystyczną walidację danych wejściowych za pomocą biblioteki Pydantic (wymuszanie odpowiednich typów, aliasy,ograniczenia dziedziny np. wiek 1-120, skale ankiet 0-5)
- optymalizację cyklu życia aplikacji (model deserializowany do pamięci tylko raz, podczas startu serwera)
- odpowiednią obsługę błędów (np. zwracanie 422 Unprocessable Entity dla złych danych oraz 503 Service Unavailable w przypadku braku załadowanego modelu)
- automatycznie generowaną, interaktywną dokumentację Swagger UI

## Uruchomienie:
1. Upewnij się, że posiadasz wygenerowany plik z wytrenowanym modelem (zapisany uprzednio przez Kedro w odpowiednim folderze `data/`).
2. Uruchom serwer pomocą uvicorn, wskazując plik i instancję aplikacji: `uvicorn api.main:app --reload`
3. Aplikacja uruchomi się na porcie 8000. Endpointy można testować poprzez graficzny interfejs Swagger UI pod adresem:
http://127.0.0.1:8000/docs