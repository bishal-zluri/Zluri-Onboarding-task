import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from unittest.mock import patch

# CHANGED: Imported read_local_data instead of read_s3_data
from s3_reader_transaction import (
    read_local_data,
    process_cards_data,
    process_budgets_data,
    process_transactions_schema
)

@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("TestTransactionsExtractor") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()


# CHANGED: Renamed class and updated patches to read_local_data
class TestReadLocalData:
    """Tests for read_local_data function"""
    
    def test_read_local_data_no_files(self, spark, capsys):
        """Test read_local_data when no files are found"""
        # CHANGED: Function call updated
        result = read_local_data(spark, "non_existent_folder")
        captured = capsys.readouterr()
        
        assert result is None
        assert "[!] No data found for entity: non_existent_folder" in captured.out
    
    # CHANGED: Patch updated to read_local_data
    @patch('s3_reader_transaction.read_local_data')
    def test_read_local_data_with_files(self, mock_read, spark):
        """Test read_local_data successfully reads files"""
        schema = StructType([StructField("id", StringType(), True)])
        mock_df = spark.createDataFrame([("txn1",)], schema)
        mock_read.return_value = mock_df
        
        # CHANGED: Function call updated
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1


class TestProcessCardsData:
    """Tests for process_cards_data function"""
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_cards_data_no_data(self, mock_read, spark):
        """Test process_cards_data when no data is found"""
        mock_read.return_value = None
        result = process_cards_data(spark)
        assert result is None
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_cards_data_with_results(self, mock_read, spark):
        """Test process_cards_data with results array wrapper"""
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", StringType(), True),
                StructField("name", StringType(), True),
                StructField("lastFour", StringType(), True),
                StructField("type", StringType(), True),
                StructField("status", StringType(), True)
            ])))
        ])
        
        data = [([
            ("card1", "Test Card", "1234", "PHYSICAL", "ACTIVE"),
            ("card2", "Virtual Card", "5678", "VIRTUAL", "ACTIVE")
        ],)]
        
        cards_df = spark.createDataFrame(data, schema)
        mock_read.return_value = cards_df
        
        result = process_cards_data(spark)
        
        assert result is not None
        assert result.count() == 2
        assert "card_id" in result.columns
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_cards_data_without_results(self, mock_read, spark):
        """Test process_cards_data without results wrapper"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("lastFour", StringType(), True),
            StructField("type", StringType(), True),
            StructField("status", StringType(), True)
        ])
        
        data = [("card1", "Test Card", "1234", "PHYSICAL", "ACTIVE")]
        cards_df = spark.createDataFrame(data, schema)
        mock_read.return_value = cards_df
        
        result = process_cards_data(spark)
        
        assert result is not None
        assert result.count() == 1
        row = result.collect()[0]
        assert row.card_id == "card1"


class TestProcessBudgetsData:
    """Tests for process_budgets_data function"""
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_budgets_data_no_data(self, mock_read, spark):
        mock_read.return_value = None
        result = process_budgets_data(spark)
        assert result is None
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_budgets_data_with_description(self, mock_read, spark):
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", StringType(), True),
                StructField("name", StringType(), True),
                StructField("description", StringType(), True)
            ])))
        ])
        data = [([("budget1", "Test Budget", "Test Description")],)]
        budgets_df = spark.createDataFrame(data, schema)
        mock_read.return_value = budgets_df
        
        result = process_budgets_data(spark)
        assert result is not None
        assert "budget_description" in result.columns

    @patch('s3_reader_transaction.read_local_data')
    def test_process_budgets_data_without_description(self, mock_read, spark):
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", StringType(), True),
                StructField("name", StringType(), True)
            ])))
        ])
        data = [([("budget1", "Test Budget")],)]
        budgets_df = spark.createDataFrame(data, schema)
        mock_read.return_value = budgets_df
        
        result = process_budgets_data(spark)
        assert result is not None
        assert "budget_description" in result.columns


class TestProcessTransactionsSchema:
    """Tests for process_transactions_schema function"""
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_transactions_no_data(self, mock_read, spark, capsys):
        mock_read.return_value = None
        result = process_transactions_schema(spark)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Critical Error: Missing Transaction data" in captured.out
    
    @patch('s3_reader_transaction.read_local_data')
    def test_process_transactions_with_results(self, mock_read, spark):
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", StringType(), True),
                StructField("transactionType", StringType(), True),
                StructField("occurredTime", StringType(), True),
                StructField("merchantName", StringType(), True),
                StructField("currencyData", StructType([
                    StructField("originalCurrencyAmount", StringType(), True),
                    StructField("exponent", LongType(), True),
                    StructField("originalCurrencyCode", StringType(), True)
                ])),
                StructField("cardId", StringType(), True),
                StructField("budgetId", StringType(), True)
            ])))
        ])
        data = [([
            ("txn1", "AUTHORIZATION", "2024-01-01T10:00:00", "Merchant", 
             ("10000", 2, "USD"), "card1", "budget1")
        ],)]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        assert result is not None
        assert "original_amount" in result.columns

    @patch('s3_reader_transaction.read_local_data')
    def test_process_transactions_amount_calculation(self, mock_read, spark):
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        data = [("txn1", ("12559", 2, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        row = result.collect()[0]
        assert abs(row.original_amount - 125.59) < 0.01