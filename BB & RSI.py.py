import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# Sector tickers
sector_tickers = {
    'Nifty Auto Index': '^CNXAUTO',
    'Nifty Consumer Durables Index': '^CNXCONSUMERDUR',
    'Nifty FMCG Index': '^CNXFMCG',
    'Nifty IT Index': '^CNXIT',
    'Nifty Media Index': '^CNXMEDIA',
    'Nifty Metal Index': '^CNXMETAL',
    'Nifty Oil and Gas Index': '^CNXOILGAS',
    'Nifty Pharma Index': '^CNXPHARMA',
    'Nifty PSU Bank Index': '^CNXPSUBANK',
    'Nifty Realty Index': '^CNXREALTY',
    'Nifty Private Bank Index': '^NIFTYPVTBANK',
    'Nifty Bank Index': '^NSEBANK',
    'Nifty Financial Services Index': 'NIFTY_FIN_SERVICE.NS',
    'Nifty Healthcare Index': 'NIFTY_HEALTHCARE.NS'
}

# Function to analyze Bollinger Bands, RSI, and display charts
def analyze_ticker_with_chart(tickerSymbol, period=None, start_date=None, end_date=None):
    try:
        # Validate the ticker symbol
        tickerData = yf.Ticker(tickerSymbol)
        if 'shortName' not in tickerData.info:
            print(f"Error: Ticker symbol '{tickerSymbol}' is not valid or data is unavailable on Yahoo Finance.")
            return

        # Get data for the ticker based on user choice
        if period:
            tickerDf = tickerData.history(period=period)
        elif start_date and end_date:
            tickerDf = tickerData.history(start=start_date, end=end_date)
        else:
            print("Error: No valid timeframe or date range selected.")
            return

        if tickerDf.empty:
            print(f"Error: No data available for ticker {tickerSymbol}.")
            return

        # Check if enough data is available for Bollinger Bands (at least 20 rows)
        if len(tickerDf) < 20:
            print(f"Error: Not enough data for Bollinger Bands calculation (requires at least 20 data points).")
            return

        # Calculate Bollinger Bands
        tickerDf['SMA'] = tickerDf['Close'].rolling(window=20).mean()
        tickerDf['SD'] = tickerDf['Close'].rolling(window=20).std()
        tickerDf['UB'] = tickerDf['SMA'] + 2 * tickerDf['SD']
        tickerDf['LB'] = tickerDf['SMA'] - 2 * tickerDf['SD']

        # Calculate RSI
        delta = tickerDf['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        tickerDf['RSI'] = 100 - (100 / (1 + rs))

        # Create Bollinger Bands Chart
        fig = go.Figure()

        # Add candlestick chart for prices
        fig.add_trace(go.Candlestick(
            x=tickerDf.index,
            open=tickerDf['Open'],
            high=tickerDf['High'],
            low=tickerDf['Low'],
            close=tickerDf['Close'],
            name='Candlestick'
        ))

        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=tickerDf.index, y=tickerDf['UB'],
            line=dict(color='blue', width=1),
            name='Upper Band'
        ))
        fig.add_trace(go.Scatter(
            x=tickerDf.index, y=tickerDf['LB'],
            line=dict(color='blue', width=1),
            name='Lower Band',
            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)'  # Light blue fill
        ))
        fig.add_trace(go.Scatter(
            x=tickerDf.index, y=tickerDf['SMA'],
            line=dict(color='red', width=1),
            name='SMA (20)'
        ))

        # Customize layout for Bollinger Bands
        fig.update_layout(
            title=f'{tickerSymbol} Bollinger Bands',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600
        )

        # Create RSI Chart
        rsi_fig = go.Figure()

        rsi_fig.add_trace(go.Scatter(
            x=tickerDf.index, y=tickerDf['RSI'],
            line=dict(color='purple', width=1.5),
            name='RSI'
        ))

        # Add Overbought and Oversold lines
        rsi_fig.add_hline(y=70, line=dict(color='red', dash='dot'), annotation_text='Overbought (70)')
        rsi_fig.add_hline(y=30, line=dict(color='green', dash='dot'), annotation_text='Oversold (30)')

        # Customize layout for RSI
        rsi_fig.update_layout(
            title=f'{tickerSymbol} RSI (14)',
            xaxis_title='Date',
            yaxis_title='RSI',
            xaxis_rangeslider_visible=False,
            height=400
        )

        # Display both charts
        fig.show()
        rsi_fig.show()

    except Exception as e:
        print(f"Error analyzing {tickerSymbol}: {e}")

# Time frame selection for analysis
def select_timeframe():
    print("Choose the timeframe for analysis:")
    print("1. 1 Day (1d)")
    print("2. 5 Days (5d)")
    print("3. 1 Month (1mo)")
    print("4. 3 Months (3mo)")
    print("5. 6 Months (6mo)")
    print("6. Year-to-Date (ytd)")
    print("7. 1 Year (1y)")
    print("8. 2 Years (2y)")
    print("9. 5 Years (5y)")
    print("10. Custom Date Range")

    timeframe_choice = input("Enter your choice: ")

    # Mapping user choice to Yahoo Finance period strings
    timeframe_map = {
        '1': '1d', '2': '5d', '3': '1mo', '4': '3mo',
        '5': '6mo', '6': 'ytd', '7': '1y', '8': '2y', '9': '5y'
    }

    if timeframe_choice in timeframe_map:
        return timeframe_map[timeframe_choice]
    elif timeframe_choice == '10':
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
            return (start_date, end_date)
        except ValueError:
            print("Error: Invalid date format. Please use YYYY-MM-DD.")
            return None
    else:
        print("Error: Invalid choice.")
        return None

# Main execution
def main():
    print("Enter 'sector' for sector analysis or 'stock' for individual stock analysis:")
    choice = input().lower()

    if choice == 'sector':
        print("Available sectors for analysis:")
        for sector in sector_tickers.keys():
            print(sector)

        sector_name = input("Enter the sector name exactly as listed above: ")

        if sector_name in sector_tickers:
            timeframe = select_timeframe()
            if timeframe:
                analyze_ticker_with_chart(sector_tickers[sector_name], period=timeframe)
        else:
            print("Error: Invalid sector name.")

    elif choice == 'stock':
        ticker_symbol = input("Enter the stock ticker symbol (e.g., 'AAPL'): ")
        timeframe = select_timeframe()
        if timeframe:
            analyze_ticker_with_chart(ticker_symbol, period=timeframe)

    else:
        print("Error: Invalid choice. Please enter 'sector' or 'stock'.")

# Run the main function
if __name__ == "__main__":
    main()