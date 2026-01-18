import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DecimalType, DateType
from unittest.mock import patch, MagicMock, Mock
from decimal import Decimal # --- FIX: Import Decimal ---
from datetime import date
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

from transaction_postgres_loader import (
    execute_raw_sql,
    _add_column_if_missing,
    init_db,
    get_existing_transaction_ids,
    load_transaction_pipeline,
    TABLE_TRANS,
    TABLE_TRANS_CARDS,
    TABLE_TRANS_BUDGETS
)

@pytest.fixture(scope="module")
def spark():
    """Real Spark session (used for creating DataFrames only)"""
    spark = SparkSession.builder \
        .appName("TestTransactionsLoader") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_trans_df(spark):
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("transaction_type", StringType(), True),
        StructField("transaction_date", StringType(), True),
        StructField("original_amount", DecimalType(18, 2), True),
        StructField("currency_code", StringType(), True),
        StructField("amount_usd", DecimalType(18, 2), True),
        StructField("merchant_name", StringType(), True),
        StructField("card_id", StringType(), True),
        StructField("budget_id", StringType(), True),
        StructField("exchange_rate", DecimalType(18, 2), True)
    ])
    # --- FIX: Use Decimal for DecimalType fields ---
    data = [
        ("txn1", "AUTH", "2024-01-01", Decimal("100.0"), "USD", Decimal("100.0"), "Merchant 1", "card1", "budget1", Decimal("1.0")),
        ("txn2", "PURCHASE", "2024-01-02", Decimal("50.0"), "GBP", Decimal("67.5"), "Merchant 2", "card2", "budget2", Decimal("1.35"))
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_cards_df(spark):
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("card_id", StringType(), True),
        StructField("card_name", StringType(), True),
        StructField("card_last_four", StringType(), True),
        StructField("card_type", StringType(), True),
        StructField("card_status", StringType(), True)
    ])
    data = [
        ("txn1", "card1", "Test Card 1", "1234", "PHYSICAL", "ACTIVE"),
        ("txn2", "card2", "Test Card 2", "5678", "VIRTUAL", "ACTIVE")
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_budgets_df(spark):
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("budget_id", StringType(), True),
        StructField("budget_name", StringType(), True),
        StructField("budget_description", StringType(), True)
    ])
    data = [
        ("txn1", "budget1", "Test Budget 1", "Description 1"),
        ("txn2", "budget2", "Test Budget 2", "Description 2")
    ]
    return spark.createDataFrame(data, schema)


class TestExecuteRawSql:
    """Tests for execute_raw_sql using Mock Spark"""
    
    def test_execute_raw_sql_success(self):
        """Test successful SQL execution"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        execute_raw_sql(mock_spark, "SELECT 1")
        
        mock_conn.setAutoCommit.assert_called_once_with(True)
        mock_stmt.execute.assert_called_once_with("SELECT 1")
        mock_conn.close.assert_called_once()
    
    def test_execute_raw_sql_failure(self, capsys):
        """Test SQL execution failure"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_stmt.execute.side_effect = Exception("SQL Error")
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        with pytest.raises(Exception) as exc_info:
            execute_raw_sql(mock_spark, "INVALID SQL")
        
        assert "SQL Error" in str(exc_info.value)
        mock_conn.close.assert_called_once()

class TestGetExistingTransactionIds:
    
    @patch('transaction_postgres_loader.init_db')
    def test_get_existing_success(self, mock_init, sample_trans_df):
        """Test successful retrieval using Mock Spark"""
        mock_spark = MagicMock()
        mock_spark.read.jdbc.return_value = sample_trans_df
        
        result = get_existing_transaction_ids(mock_spark)
        
        assert result is not None
        assert "transaction_id" in result.columns
        assert "original_amount" in result.columns
        mock_init.assert_called_once()

    @patch('transaction_postgres_loader.init_db')
    def test_get_existing_failure(self, mock_init):
        mock_spark = MagicMock()
        mock_spark.read.jdbc.side_effect = Exception("Connection failed")
        
        result = get_existing_transaction_ids(mock_spark)
        assert result is None

class TestLoadTransactionPipeline:
    """Tests using Mock Dataframes"""
    
    @patch('transaction_postgres_loader.init_db')
    @patch('transaction_postgres_loader.execute_raw_sql')
    def test_load_all_tables(self, mock_execute, mock_init, spark):
        """Test loading all three tables using Mocks"""
        # Create Mock DataFrames
        mock_trans = MagicMock()
        mock_cards = MagicMock()
        mock_budgets = MagicMock()

        # Set them to NOT be empty
        mock_trans.isEmpty.return_value = False
        mock_cards.isEmpty.return_value = False
        mock_budgets.isEmpty.return_value = False

        # Setup write.jdbc mocks
        mock_trans.write.jdbc = MagicMock()
        mock_cards.write.jdbc = MagicMock()
        mock_budgets.write.jdbc = MagicMock()

        load_transaction_pipeline(spark, mock_trans, mock_cards, mock_budgets)
        
        # Verify calls
        mock_trans.write.jdbc.assert_called()
        mock_cards.write.jdbc.assert_called()
        mock_budgets.write.jdbc.assert_called()
        assert mock_execute.called
    
    # ... (Logic for other tests like test_load_transactions_upsert is similar, 
    # check SQL strings passed to mock_execute using the same logic as User/Group tests)

    @patch('transaction_postgres_loader.init_db')
    @patch('transaction_postgres_loader.execute_raw_sql')
    def test_load_transactions_upsert(self, mock_execute, mock_init, spark):
        mock_trans = MagicMock()
        mock_trans.isEmpty.return_value = False
        mock_cards = MagicMock()
        mock_cards.isEmpty.return_value = True
        mock_budgets = MagicMock()
        mock_budgets.isEmpty.return_value = True
        
        load_transaction_pipeline(spark, mock_trans, mock_cards, mock_budgets)
        
        upsert_call = None
        for call in mock_execute.call_args_list:
            sql = call[0][1]
            if "INSERT INTO" in sql and TABLE_TRANS in sql:
                upsert_call = sql
                break
        
        assert upsert_call is not None
        assert "ON CONFLICT (transaction_id)" in upsert_call