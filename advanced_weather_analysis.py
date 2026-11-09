import pandas as pd

# Read the dataset
df = pd.read_csv("mecca_weather_data (1).csv")

# Check date type before conversion
print("Before Conversion:")
print(df["time"].dtype)

# Convert date column
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Check date type after conversion
print("\nAfter Conversion:")
print(df["time"].dtype)

# Display first 5 dates
print("\nFirst 5 Dates:")
print(df["time"].head())

print("\nInvalid Dates:")
print(df["time"].isna().sum())
# التكرار و القيم المفقودة
print("\nInvalid Dates:")
print(df["time"].isna().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDuplicate Dates:")
print(df["time"].duplicated().sum())

print("\nMissing Values:")
print(df.isna().sum())

#فحص القيم المفقودة
print("\nMissing Values Percentage:")
print((df.isna().mean() * 100).round(2))


#
# Create a copy of the original data
df_original = df.copy()

# Extract year and month
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month

print("\nYear and Month Columns:")
print(df[["time", "year", "month"]].head())

# حساب متوسط الحرارة لكل شهر
monthly_avg = df.groupby(["year", "month"]).agg(
    avg_temp=("temp", "mean")
)

print("\nMonthly Average Temperature:")
print(monthly_avg)

#
# Highest and Lowest Monthly Average
highest_month = monthly_avg["avg_temp"].idxmax()
lowest_month = monthly_avg["avg_temp"].idxmin()

print("\nHighest Average Month:")
print(highest_month, monthly_avg.loc[highest_month, "avg_temp"])

print("\nLowest Average Month:")
print(lowest_month, monthly_avg.loc[lowest_month, "avg_temp"])

# Yearly Average Temperature
yearly_avg = df.groupby("year").agg(
    avg_temp=("temp", "mean"),
    records=("temp", "count"),
    first_date=("time", "min"),
    last_date=("time", "max")
)

print("\nYearly Average Temperature:")
print(yearly_avg)

#   المتوسط المتحرك
df["moving_avg_7"] = df["temp"].rolling(window=7).mean()
print("\n7-Day Moving Average:")
print(df[["time", "temp", "moving_avg_7"]].head(12))

#
# Calculate monthly mean and standard deviation
df["month_mean"] = df.groupby("month")["temp"].transform("mean")
df["month_std"] = df.groupby("month")["temp"].transform("std")

print("\nMonthly Mean and Standard Deviation:")
print(df[["time", "temp", "month_mean", "month_std"]].head(10))

# حساب z-score
# Calculate Z-score
df["z_score"] = (
    (df["temp"] - df["month_mean"]) / df["month_std"]
)

print("\nZ-score:")
print(df[["time", "temp", "z_score"]].head(10))

# Detect temperature anomalies
anomalies = df[df["z_score"].abs() > 2].copy()

print("\nDetected Temperature Anomalies:")
print(anomalies[["time", "temp", "z_score"]])

print("\nTotal Number of Anomalies:")
print(len(anomalies))

# Visualizations
import matplotlib.pyplot as plt

# Chart 1: Monthly Average Temperature
monthly_plot = monthly_avg.reset_index()

monthly_plot["date"] = pd.to_datetime(
    monthly_plot["year"].astype(str) + "-" +
    monthly_plot["month"].astype(str) + "-01"
)

plt.figure(figsize=(12, 5))
plt.plot(monthly_plot["date"], monthly_plot["avg_temp"])
plt.title("Monthly Average Temperature")
plt.xlabel("Date")
plt.ylabel("Average Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_average_temperature.png", dpi=300)
plt.show()
plt.close()


# Chart 2: Daily Temperature and 7-Day Moving Average
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["temp"], label="Daily Temperature")
plt.plot(df["time"], df["moving_avg_7"], label="7-Day Moving Average")
plt.title("Daily Temperature and 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_temperature_moving_average.png", dpi=300)
plt.show()
plt.close()


# Chart 3: Temperature Anomalies
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["temp"], label="Daily Temperature")
plt.scatter(
    anomalies["time"],
    anomalies["temp"],
    label="Temperature Anomalies"
)
plt.title("Temperature Anomalies Detection")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("temperature_anomalies.png", dpi=300)
plt.show()
plt.close()


# Export Monthly Analysis
monthly_avg.to_csv("monthly_temperature_analysis.csv")

print("\nMonthly Analysis Exported Successfully!")

# Export Temperature Anomalies
anomalies[["time", "temp", "z_score"]].to_csv(
    "temperature_anomalies.csv",
    index=False
)

print("Temperature Anomalies Exported Successfully!")