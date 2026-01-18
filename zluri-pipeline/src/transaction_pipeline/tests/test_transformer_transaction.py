import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DecimalType
from unittest.mock import patch, MagicMock
from decimal import Decimal  # --- FIX: IMPORT DECIMAL ---
import os

from transaction_transformer import (
    load_cache,
    save_cache,
    fetch_exchange_rates_batch,
    transform_and_load_transactions
)

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
    data = [
        ("card1", "Card 1", "1234", "PHYSICAL", "ACTIVE"),
        ("card2", "Card 2", "5678", "VIRTUAL", "ACTIVE"),
        ("card3", "Card 3", "9012", "PHYSICAL", "ACTIVE")
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_budgets_df(spark):
    schema = StructType([
        StructField("budget_id", StringType(), True),
        StructField("budget_name", StringType(), True),
        StructField("budget_description", StringType(), True)
    ])
    data = [
        ("budget1", "Budget 1", "Description 1"),
        ("budget2", "Budget 2", "Description 2"),
        ("budget3", "Budget 3", "Description 3")
    ]
    return spark.createDataFrame(data, schema)

# ... [Keep TestLoadCache, TestSaveCache, TestFetchExchangeRatesBatch AS IS] ...
# (They were passing or had unrelated errors, hiding them for brevity, 
#  copy them from your original code if needed, they are fine)

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
        assert mock_load.called
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_delta_check_new_records(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                               mock_cards, mock_trans, spark, sample_transactions_df, 
                                               sample_cards_df, sample_budgets_df, capsys):
        """Test delta check identifies new records"""
        existing_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("original_amount", DecimalType(18, 2), True)
        ])
        # --- FIX: Use Decimal("100.0") instead of float 100.0 ---
        existing_data = [("txn1", Decimal("100.0"))] 
        existing_df = spark.createDataFrame(existing_data, existing_schema)
        
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = existing_df
        mock_fetch.return_value = {"GBP_2024-01-01": 1.35, "EUR_2024-01-02": 1.10}
        
        transform_and_load_transactions(spark)
        
        captured = capsys.readouterr()
        # Should process txn2 and txn3 as new
        assert "Records to Process (New + Updates): 2" in captured.out
    
    @patch('transaction_transformer.process_transactions_schema')
    @patch('transaction_transformer.process_cards_data')
    @patch('transaction_transformer.process_budgets_data')
    @patch('transaction_transformer.get_existing_transaction_ids')
    @patch('transaction_transformer.fetch_exchange_rates_batch')
    @patch('transaction_transformer.load_transaction_pipeline')
    def test_transform_delta_check_updated_records(self, mock_load, mock_fetch, mock_existing, mock_budgets,
                                                   mock_cards, mock_trans, spark, sample_transactions_df, 
                                                   sample_cards_df, sample_budgets_df):
        existing_schema = StructType([
            StructField("transaction_id", StringType(), True),
            StructField("original_amount", DecimalType(18, 2), True)
        ])
        # --- FIX: Use Decimal ---
        existing_data = [
            ("txn1", Decimal("200.0")), # Different amount
            ("txn2", Decimal("50.0")), 
            ("txn3", Decimal("75.0"))
        ]
        existing_df = spark.createDataFrame(existing_data, existing_schema)
        
        mock_trans.return_value = sample_transactions_df
        mock_cards.return_value = sample_cards_df
        mock_budgets.return_value = sample_budgets_df
        mock_existing.return_value = existing_df
        mock_fetch.return_value = {"USD_2024-01-01": 1.0, "GBP_2024-01-01": 1.35, "EUR_2024-01-02": 1.10}
        
        transform_and_load_transactions(spark)
        
        assert mock_load.called
    
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
        # --- FIX: Use Decimal ---
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