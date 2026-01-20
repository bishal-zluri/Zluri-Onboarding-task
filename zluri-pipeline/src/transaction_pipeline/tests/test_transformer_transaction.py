import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DecimalType
from unittest.mock import patch, MagicMock, mock_open, ANY
from decimal import Decimal
import os
import pickle

import transaction_transformer
from transaction_transformer import (
    load_cache,
    save_cache,
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

class TestCacheFunctions:
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps({"GBP_2023-01-01": 1.5}))
    def test_load_cache_success(self, mock_file, mock_exists):
        mock_exists.return_value = True
        cache = load_cache()
        assert cache == {"GBP_2023-01-01": 1.5}

    @patch("os.path.exists")
    def test_load_cache_no_file(self, mock_exists):
        mock_exists.return_value = False
        assert load_cache() == {}

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_save_cache_success(self, mock_dump, mock_file):
        save_cache({"USD": 1.0})
        mock_dump.assert_called()

class TestFetchExchangeRates:
    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    def test_fetch_usd_skipped(self, mock_save, mock_load):
        mock_load.return_value = {}
        rates = fetch_exchange_rates_batch([("USD", "2024-01-01")])
        assert rates["USD_2024-01-01"] == 1.0

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_success(self, mock_get, mock_save, mock_load):
        mock_load.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "quotes": {"USDGBP": 0.79}}
        mock_get.return_value = mock_response
        
        rates = fetch_exchange_rates_batch([("GBP", "2024-01-01")])
        assert rates["GBP_2024-01-01"] == 1.0 / 0.79

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_failure_response(self, mock_get, mock_save, mock_load, capsys):
        """Test handling of invalid API response (missing rate logic)."""
        mock_load.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        # "quotes" is missing or key not found
        mock_response.json.return_value = {"success": False, "error": "info"} 
        mock_get.return_value = mock_response
        
        rates = fetch_exchange_rates_batch([("EUR", "2024-01-01")])
        
        assert rates["EUR_2024-01-01"] == 1.0
        captured = capsys.readouterr()
        # FIX: Updated assertion string to match current code
        assert "Missing rate for USDEUR" in captured.out

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