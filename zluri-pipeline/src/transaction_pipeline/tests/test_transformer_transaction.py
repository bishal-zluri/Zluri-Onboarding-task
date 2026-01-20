import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DecimalType
from unittest.mock import patch, MagicMock, mock_open, ANY
from decimal import Decimal
import os
import json
import pickle

# Import the module under test
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
    data = [
        ("txn1", "AUTH", "2024-01-01", "Merchant 1", 100.0, "USD", "card1", "budget1"),
        ("txn2", "AUTH", "2024-01-01", "Merchant 2", 50.0, "GBP", "card2", "budget2"),
        ("txn3", "AUTH", "2024-01-02", "Merchant 3", 75.0, "EUR", "card3", "budget3")
    ]
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
    return spark.createDataFrame([
        ("card1", "Card 1", "1234", "PHYSICAL", "ACTIVE"),
        ("card2", "Card 2", "5678", "VIRTUAL", "ACTIVE"),
        ("card3", "Card 3", "9012", "PHYSICAL", "ACTIVE")
    ], schema)

@pytest.fixture
def sample_budgets_df(spark):
    schema = StructType([
        StructField("budget_id", StringType(), True),
        StructField("budget_name", StringType(), True),
        StructField("budget_description", StringType(), True)
    ])
    return spark.createDataFrame([
        ("budget1", "Budget 1", "Desc 1"),
        ("budget2", "Budget 2", "Desc 2"),
        ("budget3", "Budget 3", "Desc 3")
    ], schema)

# --- TESTS FOR CACHING LOGIC (Boosts Coverage) ---

class TestCacheFunctions:
    
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data=pickle.dumps({"GBP_2023-01-01": 1.5}))
    def test_load_cache_success(self, mock_file, mock_exists):
        """Test loading a valid cache file."""
        mock_exists.return_value = True
        cache = load_cache()
        assert cache == {"GBP_2023-01-01": 1.5}
        mock_file.assert_called_with("fx_rate_cache.pkl", 'rb')

    @patch("os.path.exists")
    def test_load_cache_no_file(self, mock_exists):
        """Test missing cache file returns empty dict."""
        mock_exists.return_value = False
        cache = load_cache()
        assert cache == {}

    @patch("os.path.exists")
    @patch("builtins.open", side_effect=IOError("Corrupt file"))
    def test_load_cache_error(self, mock_file, mock_exists, capsys):
        """Test error handling when reading cache."""
        mock_exists.return_value = True
        cache = load_cache()
        assert cache == {}
        captured = capsys.readouterr()
        assert "Error loading cache" in captured.out

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_save_cache_success(self, mock_dump, mock_file):
        """Test saving cache."""
        data = {"USD": 1.0}
        save_cache(data)
        mock_file.assert_called_with("fx_rate_cache.pkl", 'wb')
        mock_dump.assert_called_with(data, ANY)

    @patch("builtins.open", side_effect=IOError("Permission denied"))
    def test_save_cache_error(self, mock_file, capsys):
        """Test error handling when saving cache."""
        save_cache({"USD": 1.0})
        captured = capsys.readouterr()
        assert "Error saving cache" in captured.out

# --- TESTS FOR API LOGIC (Boosts Coverage) ---

class TestFetchExchangeRates:
    
    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    def test_fetch_usd_skipped(self, mock_save, mock_load):
        """Test that USD is skipped and hardcoded to 1.0."""
        mock_load.return_value = {}
        pairs = [("USD", "2024-01-01")]
        
        rates = fetch_exchange_rates_batch(pairs)
        
        assert rates["USD_2024-01-01"] == 1.0
        mock_save.assert_called()

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_cache_hit(self, mock_get, mock_save, mock_load):
        """Test that cache hits avoid API calls."""
        mock_load.return_value = {"GBP_2024-01-01": 1.25}
        pairs = [("GBP", "2024-01-01")]
        
        rates = fetch_exchange_rates_batch(pairs)
        
        assert rates["GBP_2024-01-01"] == 1.25
        mock_get.assert_not_called()

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_success(self, mock_get, mock_save, mock_load):
        """Test successful API fetch."""
        mock_load.return_value = {}
        # Mock API Response: 1 USD = 0.79 GBP (so 1 GBP should be 1/0.79 = ~1.26)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "quotes": {"USDGBP": 0.79}
        }
        mock_get.return_value = mock_response
        
        pairs = [("GBP", "2024-01-01")]
        rates = fetch_exchange_rates_batch(pairs)
        
        expected_rate = 1.0 / 0.79
        assert rates["GBP_2024-01-01"] == expected_rate

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_429_rate_limit(self, mock_get, mock_save, mock_load, capsys):
        """Test fallback on Rate Limit (429)."""
        mock_load.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 429 # Rate Limit
        mock_get.return_value = mock_response
        
        pairs = [("EUR", "2024-01-01")]
        rates = fetch_exchange_rates_batch(pairs)
        
        assert rates["EUR_2024-01-01"] == 1.0 # Fallback
        captured = capsys.readouterr()
        assert "Rate limit reached" in captured.out

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_failure_response(self, mock_get, mock_save, mock_load, capsys):
        """Test handling of success: False from API."""
        mock_load.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": False, "error": {"info": "Invalid Key"}}
        mock_get.return_value = mock_response
        
        pairs = [("EUR", "2024-01-01")]
        rates = fetch_exchange_rates_batch(pairs)
        
        assert rates["EUR_2024-01-01"] == 1.0
        captured = capsys.readouterr()
        assert "API failed for EUR" in captured.out

    @patch("transaction_transformer.load_cache")
    @patch("transaction_transformer.save_cache")
    @patch("requests.get")
    def test_fetch_api_missing_data(self, mock_get, mock_save, mock_load):
        """Test handling of missing quotes in response."""
        mock_load.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "quotes": {}} # Empty quotes
        mock_get.return_value = mock_response
        
        pairs = [("EUR", "2024-01-01")]
        rates = fetch_exchange_rates_batch(pairs)
        
        assert rates["EUR_2024-01-01"] == 1.0

# --- TESTS FOR MAIN PIPELINE (Logic & Delta Checks) ---

class TestTransformAndLoadTransactions:
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_no_transactions(self, mock_load, mock_existing, mock_budgets, 
                                      mock_cards, mock_trans, spark, capsys):
        mock_trans.return_value = None
        transform_and_load_transactions(spark)
        captured = capsys.readouterr()
        assert "No transactions to process" in captured.out
        mock_load.assert_not_called()
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_initial_load(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                    mock_cards, mock_trans, spark, sample_transactions_df, 
                                    sample_cards_df, sample_budgets_df, capsys):
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = None
        mock_fetch.return_value = {"USD_2024-01-01": 1.0, "GBP_2024-01-01": 1.35, "EUR_2024-01-02": 1.10}
        
        transform_and_load_transactions(spark)
        
        captured = capsys.readouterr()
        assert "Records to Process (New + Updates): 3" in captured.out
        mock_load.assert_called()
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_delta_check_new_records(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                               mock_cards, mock_trans, spark, sample_transactions_df, 
                                               sample_cards_df, sample_budgets_df, capsys):
        existing_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("original_amount", DecimalType(18, 2), True)
        ])
        existing_data = [("txn1", Decimal("100.0"))] 
        existing_df = spark.createDataFrame(existing_data, existing_schema)
        
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = existing_df
        mock_fetch.return_value = {"GBP_2024-01-01": 1.35, "EUR_2024-01-02": 1.10}
        
        transform_and_load_transactions(spark)
        
        captured = capsys.readouterr()
        assert "Records to Process (New + Updates): 2" in captured.out
        mock_load.assert_called()
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_no_new_or_updated(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                         mock_cards, mock_trans, spark, sample_transactions_df, 
                                         sample_cards_df, sample_budgets_df, capsys):
        existing_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("original_amount", DecimalType(18, 2), True)
        ])
        existing_data = [
            ("txn1", Decimal("100.0")), 
            ("txn2", Decimal("50.0")), 
            ("txn3", Decimal("75.0"))
        ]
        existing_df = spark.createDataFrame(existing_data, existing_schema)
        
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = existing_df
        
        transform_and_load_transactions(spark)
        
        captured = capsys.readouterr()
        assert "No new or updated transactions found" in captured.out
        mock_load.assert_not_called()