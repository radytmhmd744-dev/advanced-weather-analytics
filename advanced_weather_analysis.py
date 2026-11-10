import pandas as pd
import matplotlib.pyplot as plt
import calendar

def load_data():
    df = pd.read_csv("mecca_weather_data (1).csv")
    return df

# Function 2: Check Data
def check_data(df):

    print("Before Conversion:")
    print(df["time"].dtype)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    print("\nAfter Conversion:")
    print(df["time"].dtype)

    print("\nFirst 5 Dates:")
    print(df["time"].head())

    print("\nInvalid Dates:")
    print(df["time"].isna().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nDuplicate Dates:")
    print(df["time"].duplicated().sum())

    print("\nMissing Values:")
    print(df.isna().sum())

    print("\nMissing Values Percentage:")
    print((df.isna().mean() * 100).round(2))

    # Sort data by date
    df = df.sort_values("time").reset_index(drop=True)

    return df


# Function 3: Monthly Analysis
def analyze_monthly(df):

    # Extract year and month
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    print("\nYear and Month Columns:")
    print(df[["time", "year", "month"]].head())

    # Calculate monthly average temperature
    monthly_avg = df.groupby(["year", "month"]).agg(
        avg_temp=("temp", "mean")
    )

    print("\nMonthly Average Temperature:")
    print(monthly_avg)

    # Highest and Lowest Monthly Average
    highest_month = monthly_avg["avg_temp"].idxmax()
    lowest_month = monthly_avg["avg_temp"].idxmin()

    print("\nHighest Average Month:")
    print(highest_month, monthly_avg.loc[highest_month, "avg_temp"])

    print("\nLowest Average Month:")
    print(lowest_month, monthly_avg.loc[lowest_month, "avg_temp"])

    return df, monthly_avg

# Yearly Average Temperature
def analyze_yearly(df):

    yearly_avg = df.groupby("year").agg(
        avg_temp=("temp", "mean"),
        records=("temp", "count"),
        first_date=("time", "min"),
        last_date=("time", "max")
    )

    print("\nYearly Average Temperature:")
    print(yearly_avg)

    # Check expected days in each year
    yearly_avg["expected_records"] = [
        366 if calendar.isleap(int(year)) else 365
        for year in yearly_avg.index
    ]

    # Select complete years only
    complete_years = yearly_avg[
        (yearly_avg["records"] == yearly_avg["expected_records"]) &
        (yearly_avg["first_date"].dt.month == 1) &
        (yearly_avg["first_date"].dt.day == 1) &
        (yearly_avg["last_date"].dt.month == 12) &
        (yearly_avg["last_date"].dt.day == 31)
    ]

    print("\nComplete Years Only:")
    print(complete_years)

    # Highest and Lowest Complete Year
    highest_year = complete_years["avg_temp"].idxmax()
    lowest_year = complete_years["avg_temp"].idxmin()

    print("\nHighest Average Year:")
    print(highest_year, complete_years.loc[highest_year, "avg_temp"])

    print("\nLowest Average Year:")
    print(lowest_year, complete_years.loc[lowest_year, "avg_temp"])

    return yearly_avg, complete_years
#   المتوسط المتحرك
def calculate_moving_average(df):

    df["moving_avg_7"] = df["temp"].rolling(window=7).mean()

    print("\n7-Day Moving Average:")
    print(df[["time", "temp", "moving_avg_7"]].head(12))

    return df
# Calculate monthly mean and standard deviation
def detect_anomalies(df):

    # Calculate monthly mean and standard deviation
    df["month_mean"] = df.groupby("month")["temp"].transform("mean")
    df["month_std"] = df.groupby("month")["temp"].transform("std")

    print("\nMonthly Mean and Standard Deviation:")
    print(df[["time", "temp", "month_mean", "month_std"]].head(10))

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

    return df, anomalies

# Function 7: Create Plots
def create_plots(df, monthly_avg, anomalies):

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

#  Save Results
def save_results(monthly_avg, anomalies):

    monthly_avg.to_csv("monthly_temperature_analysis.csv")

    anomalies[["time", "temp", "z_score"]].to_csv(
        "temperature_anomalies.csv",
        index=False
    )

    print("\nAnalysis results saved successfully.")

# Main Function
def main():

    #  Load data
    df = load_data()

    #  Check and sort data
    df = check_data(df)

    #  Monthly analysis
    df, monthly_avg = analyze_monthly(df)

    #  Yearly analysis
    yearly_avg, complete_years = analyze_yearly(df)

    #  Calculate moving average
    df = calculate_moving_average(df)

    #  Detect anomalies
    df, anomalies = detect_anomalies(df)

    #  Create plots
    create_plots(df, monthly_avg, anomalies)

    #  Save results
    save_results(monthly_avg, anomalies)


if __name__ == "__main__":
    main()
