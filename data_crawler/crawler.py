import requests
import json
import time
import os
import datetime
from datetime import timedelta


def get_historical_weather_data_of_month(year, month):
    # Get month end date depending on the month
    if month == 2:
        month_end_date = 28
    elif month in [4, 6, 9, 11]:
        month_end_date = 30
    else:
        month_end_date = 31

    month_str = '0' + str(month) if month < 10 else str(month)

    url = f"https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate={year}{month_str}01&endDate={year}{month_str}{month_end_date}"

    response = requests.get(url)
    data = response.json()
    return data


if __name__ == "__main__":
    interested_fields = ['valid_time_gmt', 'temp', 'wspd', 'pressure', 'precip_hrly']

    year = input("Enter year: ")

    with open(f'{year}_weather_data.csv', 'w') as f:
        f.write(','.join(interested_fields) + '\n')

        for month in range(1, 13):
            data = get_historical_weather_data_of_month(year, month)
            print(f'Data points for {year}-{month}: {len(data["observations"])}')

            for observation in data['observations']:
                f.write(','.join([str(observation[field]) for field in interested_fields]) + '\n')

    print(f'Done')