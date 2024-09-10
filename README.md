# S&P 500 Stock Price Prediction Model

This project implements a simple linear regression model to predict stock prices for companies in the S&P 500 index. It allows users to select a stock symbol and visualize the actual vs. predicted stock prices.

## Features

- Fetches real-time list of S&P 500 companies
- Downloads historical stock data using yfinance
- Implements a linear regression model for stock price prediction
- Visualizes actual vs. predicted stock prices
- Calculates and displays performance metrics (MSE and R^2 score)

## Requirements

- Python 3.7+
- pandas
- yfinance
- scikit-learn
- matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sp500-stock-prediction.git
   cd sp500-stock-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```
python stock_prediction.py
```

Follow the prompts to select a stock symbol from the S&P 500 list. The script will then fetch the data, train the model, and display the results.

## How it Works

1. The script fetches the current list of S&P 500 companies from Wikipedia.
2. It downloads historical stock data for the selected company using yfinance.
3. The data is preprocessed and split into training and testing sets.
4. A linear regression model is trained on the data.
5. The model predicts stock prices for the test set.
6. Actual and predicted prices are visualized using matplotlib.
7. Performance metrics (MSE and R^2 score) are calculated and displayed.

## Limitations

- This model uses a simple linear regression, which may not capture complex market dynamics.
- Past performance does not guarantee future results in stock markets.
- The model does not account for external factors that may influence stock prices.

## Future Improvements

- Implement more advanced machine learning models (e.g., LSTM, ARIMA)
- Include additional features like volume, market indicators, and sentiment analysis
- Optimize hyperparameters for better performance
- Implement backtesting and cross-validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for providing the historical stock data through the yfinance library.
- [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) for maintaining the list of S&P 500 companies.

## Disclaimer

This project is for educational purposes only. It should not be used for making real investment decisions. Always consult with a qualified financial advisor before making investment decisions.
