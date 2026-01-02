# Flight Delay Prediction - Spark Application

## Overview
This application loads a pre-trained Spark Pipeline (`final_flight_app_model`) to predict flight arrival delays based on unseen test data. It handles the full ETL process, including cleaning, feature engineering, and joining auxiliary datasets (Planes and Airports).

## Prerequisites
- Apache Spark (installed and configured with `spark-submit`)
- Python 3.x
- Data files: `flights_test.csv`, `plane-data.csv`, `airports.csv`

## Project Structure
The submission is organized as follows:

/Flight-Delay-Predictor
   ├── README.txt                     # This instruction file
   ├── requirements.txt               # List of dependencies
   │
   └── /code
         ├── notebook.ipynb           # Training, EDA, and Model Selection (Jupyter)
         ├── app.py                   # Spark Application for unseen data
         │
         └── /models
                └── final_flight_app_model  # The pre-trained Spark Pipeline (Do not rename)

## Usage
Run the application using `spark-submit`. You must provide the path to the test data. The paths to the auxiliary files are optional (defaults provided in code).

### Usage & Syntax
The application requires the path to the test data as the primary argument.

**Auxiliary Data (Optional):**
The paths for `plane-data.csv` and `airports.csv` are optional. If not provided via the command line, the application defaults to the following relative paths (based on the project's training environment):
- Planes:   `../../training_data/documentation/plane-data.csv`
- Airports: `../../training_data/documentation/airports.csv`

If your data resides elsewhere, simply provide the paths as the 2nd and 3rd arguments.

**Command:**
```bash
spark-submit app.py <path_to_test_data> [path_to_plane_data] [path_to_airport_data]