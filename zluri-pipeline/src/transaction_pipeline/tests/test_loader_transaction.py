import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DecimalType
from unittest.mock import patch, MagicMock
from decimal import Decimal
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

from transaction_postgres_loader import (
    execute_raw_sql,
    init_db,
    get_existing_transaction_ids,
    load_transaction_pipeline,
    TABLE_TRANS
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
        StructField("original_amount", DecimalType(18, 2), True),
    ])
    data = [("txn1", Decimal("100.0"))]
    return spark.createDataFrame(data, schema)

class TestExecuteRawSql:
    def test_execute_raw_sql_success(self):
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        execute_raw_sql(mock_spark, "SELECT 1")
        mock_stmt.execute.assert_called_once_with("SELECT 1")

class TestGetExistingTransactionIds:
    @patch('transaction_postgres_loader.init_db')
    def test_get_existing_success(self, mock_init, sample_trans_df):
        mock_spark = MagicMock()
        mock_spark.read.jdbc.return_value = sample_trans_df
        result = get_existing_transaction_ids(mock_spark)
        assert result is not None
        mock_init.assert_called_once()

class TestLoadTransactionPipeline:
    """Tests using Mock Dataframes"""
    
    @patch('transaction_postgres_loader.init_db')
    @patch('transaction_postgres_loader.execute_raw_sql')
    def test_load_all_tables(self, mock_execute, mock_init, spark):
        """Test loading all tables including new Dimensions"""
        # Create Mock DataFrames
        mock_trans = MagicMock()
        mock_cards = MagicMock()
        mock_budgets = MagicMock()
        mock_cards_dim = MagicMock()   # New
        mock_budgets_dim = MagicMock() # New

        # Set them to NOT be empty so logic runs
        for mock_df in [mock_trans, mock_cards, mock_budgets, mock_cards_dim, mock_budgets_dim]:
            mock_df.isEmpty.return_value = False
            mock_df.write.jdbc = MagicMock()

        # Call with NEW signature (6 arguments)
        load_transaction_pipeline(
            spark, 
            mock_trans, 
            mock_cards, 
            mock_budgets, 
            mock_cards_dim, 
            mock_budgets_dim
        )
        
        # Verify calls
        mock_trans.write.jdbc.assert_called()
        mock_cards.write.jdbc.assert_called()
        mock_budgets.write.jdbc.assert_called()
        mock_cards_dim.write.jdbc.assert_called()   # New Check
        mock_budgets_dim.write.jdbc.assert_called() # New Check
        assert mock_execute.called

    @patch('transaction_postgres_loader.init_db')
    @patch('transaction_postgres_loader.execute_raw_sql')
    def test_load_transactions_upsert(self, mock_execute, mock_init, spark):
        # Setup specific scenario: Only Transactions present
        mock_trans = MagicMock()
        mock_trans.isEmpty.return_value = False
        
        # Others empty
        empty_mock = MagicMock()
        empty_mock.isEmpty.return_value = True
        
        load_transaction_pipeline(
            spark, 
            mock_trans, 
            empty_mock, # cards join
            empty_mock, # budgets join
            empty_mock, # cards dim
            empty_mock  # budgets dim
        )
        
        # Verify Upsert SQL was generated
        upsert_call = None
        for call in mock_execute.call_args_list:
            sql = call[0][1]
            if "INSERT INTO" in sql and TABLE_TRANS in sql:
                upsert_call = sql
                break
        
        assert upsert_call is not None
        assert "ON CONFLICT (transaction_id)" in upsert_call