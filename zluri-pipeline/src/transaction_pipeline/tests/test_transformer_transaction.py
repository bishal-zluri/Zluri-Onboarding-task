import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from unittest.mock import patch, MagicMock

# CHANGED: Removed load_cache, save_cache from imports
from transaction_transformer import (
    fetch_exchange_rates_batch,
    transform_and_load_transactions
)

# --- FIXTURES ---
@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestTransactionsTransformer") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_transactions_df(spark):
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("transaction_type", StringType(), True),
        StructField("transaction_date", StringType(), True),
        StructField("merchant_name", StringType(), True),
        StructField("original_amount", DoubleType(), True),
        StructField("currency_code", StringType(), True),
        StructField("card_id", StringType(), True),
        StructField("budget_id", StringType(), True)
    ])
    data = [("txn1", "AUTH", "2024-01-01", "Merchant", 100.0, "USD", "c1", "b1")]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_cards_df(spark):
    schema = StructType([
        StructField("card_id", StringType(), True),
        StructField("card_name", StringType(), True),
        StructField("card_last_four", StringType(), True),
        StructField("card_type", StringType(), True),
        StructField("card_status", StringType(), True)
    ])
    return spark.createDataFrame([("c1", "Card", "1234", "P", "A")], schema)

@pytest.fixture
def sample_budgets_df(spark):
    schema = StructType([
        StructField("budget_id", StringType(), True),
        StructField("budget_name", StringType(), True),
        StructField("budget_description", StringType(), True)
    ])
    return spark.createDataFrame([("b1", "Budget", "Desc")], schema)

# --- TESTS ---

# CHANGED: Removed TestCacheFunctions class entirely as functions don't exist

class TestFetchExchangeRates:
    # CHANGED: Removed mocks for load_cache/save_cache
    @patch("requests.get")
    def test_fetch_api_success(self, mock_get):
        """Test successful API call"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Mocking logic: 1 USD = 0.79 GBP -> Rate = 1/0.79
        mock_response.json.return_value = {"usd": {"gbp": 0.79}}
        mock_get.return_value = mock_response
        
        rates = fetch_exchange_rates_batch([("GBP", "2024-01-01")])
        
        # Verify Key existence and calculation
        assert "GBP_2024-01-01" in rates
        expected_rate = 1.0 / 0.79
        assert abs(rates["GBP_2024-01-01"] - expected_rate) < 0.0001

    @patch("requests.get")
    def test_fetch_api_failure_response(self, mock_get):
        """Test handling of invalid API response"""
        mock_response = MagicMock()
        mock_response.status_code = 404 # Simulate failure
        mock_get.return_value = mock_response
        
        rates = fetch_exchange_rates_batch([("EUR", "2024-01-01")])
        
        # Should fallback to 1.0 for USD if logic dictates, or fail gracefully
        # Based on your code: rate_map[f"USD_{date_str}"] = 1.0 is always set
        # But EUR won't be in the map if the API fails, OR you might default it.
        # Checking your source code: if API fails, it continues. 
        # So EUR_2024-01-01 will NOT be in the map, but USD_... will be.
        
        assert "USD_2024-01-01" in rates
        assert rates["USD_2024-01-01"] == 1.0
        # Ensure we didn't crash

class TestTransformAndLoadTransactions:
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_initial_load(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                    mock_cards, mock_trans, spark, sample_transactions_df, 
                                    sample_cards_df, sample_budgets_df):
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = None
        mock_fetch.return_value = {"USD_2024-01-01": 1.0}
        
        transform_and_load_transactions(spark)
        
        # Verify load called with 6 args (Spark + 5 DataFrames)
        assert mock_load.called
        args = mock_load.call_args[0]
        assert len(args) == 6