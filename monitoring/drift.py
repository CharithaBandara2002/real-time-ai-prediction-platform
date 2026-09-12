import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

print("Loading datasets...")

reference_data = pd.read_csv("data/raw/housing.csv")
current_data = pd.read_csv("data/raw/current_data.csv")

# Remove text column
reference_data = reference_data.drop("Address", axis=1)
current_data = current_data.drop("Address", axis=1)

print("Generating drift report...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)
print(dir(report))

print("Drift report generated successfully!")