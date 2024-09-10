import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to get the list of S&P 500 companies
def get_sp500_symbols():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_df = pd.read_html(url)[0]
    return sp500_df['Symbol'].tolist()

# Function to fetch stock data, train model, and predict stock prices
def fetch_and_predict_stock(stock_symbol):
    try:
        print(f"Processing {stock_symbol}...")
        stock_data = yf.download(stock_symbol, start='2015-01-01', end='2023-01-01', progress=False)
        
        if stock_data.empty:
            print(f"No data for {stock_symbol}. Skipping...")
            return None
        
        stock_data['Date'] = stock_data.index
        stock_data.reset_index(drop=True, inplace=True)
        
        # Convert dates to ordinal values
        stock_data['Date_ordinal'] = pd.to_datetime(stock_data['Date']).map(pd.Timestamp.toordinal)
        
        # Prepare features and target
        X = stock_data[['Date_ordinal']]
        y = stock_data['Close']
        
        if X.empty or y.isnull().any():
            print(f"Insufficient data for {stock_symbol}. Skipping...")
            return None
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict the stock prices
        y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Date'], stock_data['Close'], label="Actual Stock Price", color='blue')
        plt.plot(stock_data['Date'].iloc[len(X_train):], y_pred, label="Predicted Stock Price", color='red')
        plt.title(f'{stock_symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price (USD)')
        plt.legend()
        plt.show()
        
        return {
            "symbol": stock_symbol,
            "mse": mse,
            "r2": r2
        }
    
    except Exception as e:
        print(f"Error processing {stock_symbol}: {e}")
        return None

# Main function to handle user input
def main():
    # Get the list of S&P 500 stock symbols
    symbols = get_sp500_symbols()
    
    # Display available stock symbols
    print("Available S&P 500 Stock Symbols:")
    for i, symbol in enumerate(symbols[:20], start=1):  # Display first 20 symbols for brevity
        print(f"{i}. {symbol}")
    
    # Get user input
    while True:
        try:
            choice = int(input("\nEnter the number of the stock symbol you want to predict (1-20): "))
            if 1 <= choice <= 20:
                stock_symbol = symbols[choice - 1]
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 20.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Fetch and predict stock data for the chosen symbol
    result = fetch_and_predict_stock(stock_symbol)
    
    if result:
        print(f"\nResults for {result['symbol']}:")
        print(f"Mean Squared Error: {result['mse']}")
        print(f"R^2 Score: {result['r2']}")
    else:
        print("Failed to fetch or process the stock data.")

# Run the main function
if __name__ == "__main__":
    main()
