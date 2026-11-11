import cdsapi
import os

c = cdsapi.Client(timeout=600)

years = ["2023", "2024", "2025"]
months = [f"{m:02d}" for m in range(1, 13)]

area = [13.5, 79.8, 12.7, 80.8]

os.makedirs("era5_data", exist_ok=True)

for year in years:
    for month in months:
        filename = f"era5_data/chennai_{year}_{month}.grib"

        if os.path.exists(filename):
            print(f"Skipping {year}-{month}, already downloaded.")
            continue

        print(f"Downloading {year}-{month}...")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "2m_temperature",
                    "2m_dewpoint_temperature",
                    "surface_pressure",
                    "total_precipitation",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": year,
                "month": month,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": area,
                "format": "grib",
            },
            filename,
        )

print("All downloads completed.")
