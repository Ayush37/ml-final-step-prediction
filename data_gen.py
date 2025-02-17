import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of records to generate
num_records = 100_000  # Adjust this value as needed

# Define function to generate random timestamps within range
def random_time(base_date, start_time, end_time, num_samples):
    start = datetime.combine(base_date, datetime.strptime(start_time, "%H:%M").time())
    end = datetime.combine(base_date, datetime.strptime(end_time, "%H:%M").time())
    delta = end - start
    return [start + timedelta(seconds=np.random.randint(0, delta.total_seconds())) for _ in range(num_samples)]

# Generate random dates within a year's time (365 days)
start_date = datetime(2024, 1, 1)
date_range = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_records)]

# Generate feed arrival times (random within 9 AM - 3 PM)
feed_times = {f"FEED{i}": [random_time(date_range[j], "09:00", "15:00", 1)[0] for j in range(num_records)] for i in range(1, 26)}

# Determine last feed completion time for each record
last_feed_completion = [max([feed_times[f"FEED{i}"][j] for i in range(1, 26)]) for j in range(num_records)]

# Generate FINAL_STEP start time (shortly after last feed completion, adding 1-10 minutes delay)
final_step_start = [time + timedelta(minutes=np.random.randint(1, 11)) for time in last_feed_completion]

# Create DataFrame
data = {
    "Date": date_range,
    **{f"FEED{i}_Completion": feed_times[f"FEED{i}"] for i in range(1, 26)},
    "Last_Feed_Completion": last_feed_completion,
    "FINAL_STEP_Start": final_step_start
}

df = pd.DataFrame(data)

# Sort by date for time-series continuity
df.sort_values(by=["Date"], inplace=True)

# Save dataset to CSV
df.to_csv("batch_processing_data.csv", index=False)

print("Dataset generation complete. Saved as batch_processing_data.csv")

