import yfinance as yf
import datetime
import calendar

def process_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d")
    if not vix_data.empty:
        vix_value = vix_data['Close'].iloc[0]
        if vix_value <= 20:
            print(f"VIX value {vix_value} is within the acceptable range.")
            # Add your logic here to process the VIX value
        else:
            print(f"VIX value {vix_value} is above the acceptable range. Ignoring.")
    else:
        print("Could not fetch VIX data.")


if __name__ == "__main__":
    process_vix()
