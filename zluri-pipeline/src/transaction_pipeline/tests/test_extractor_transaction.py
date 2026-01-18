import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, DoubleType
from unittest.mock import patch

from s3_reader_transaction import (
    read_s3_data,
    process_cards_data,
    process_budgets_data,
    process_transactions_schema,
    ENTITY_TRANSACTIONS,
    ENTITY_CARDS,
    ENTITY_BUDGETS
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


class TestReadS3Data:
    """Tests for read_s3_data function"""
    
    def test_read_s3_data_no_files(self, spark, capsys):
        """Test read_s3_data when no files are found"""
        result = read_s3_data(spark, "non_existent_folder")
        captured = capsys.readouterr()
        
        assert result is None
        assert "[!] No data found for entity: non_existent_folder" in captured.out
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_read_s3_data_with_files(self, mock_read, spark):
        """Test read_s3_data successfully reads files"""
        schema = StructType([StructField("id", StringType(), True)])
        mock_df = spark.createDataFrame([("txn1",)], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1


class TestProcessCardsData:
    """Tests for process_cards_data function"""
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_cards_data_no_data(self, mock_read, spark):
        """Test process_cards_data when no data is found"""
        mock_read.return_value = None
        result = process_cards_data(spark)
        assert result is None
    
    @patch('s3_reader_transaction.read_s3_data')
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
        assert "card_name" in result.columns
        assert "card_last_four" in result.columns
        assert "card_type" in result.columns
        assert "card_status" in result.columns
    
    @patch('s3_reader_transaction.read_s3_data')
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
        assert "card_id" in result.columns
        
        row = result.collect()[0]
        assert row.card_id == "card1"
        assert row.card_name == "Test Card"
        assert row.card_last_four == "1234"


class TestProcessBudgetsData:
    """Tests for process_budgets_data function"""
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_budgets_data_no_data(self, mock_read, spark):
        """Test process_budgets_data when no data is found"""
        mock_read.return_value = None
        result = process_budgets_data(spark)
        assert result is None
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_budgets_data_with_description(self, mock_read, spark):
        """Test process_budgets_data with description field"""
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", StringType(), True),
                StructField("name", StringType(), True),
                StructField("description", StringType(), True)
            ])))
        ])
        
        data = [([
            ("budget1", "Test Budget", "Test Description"),
            ("budget2", "Another Budget", "Another Description")
        ],)]
        
        budgets_df = spark.createDataFrame(data, schema)
        mock_read.return_value = budgets_df
        
        result = process_budgets_data(spark)
        
        assert result is not None
        assert result.count() == 2
        assert "budget_id" in result.columns
        assert "budget_name" in result.columns
        assert "budget_description" in result.columns
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_budgets_data_without_description(self, mock_read, spark):
        """Test process_budgets_data without description field"""
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
        # Should have budget_description column even if null
        assert "budget_description" in result.columns
        
        row = result.collect()[0]
        assert row.budget_id == "budget1"
        assert row.budget_name == "Test Budget"
        assert row.budget_description is None


class TestProcessTransactionsSchema:
    """Tests for process_transactions_schema function"""
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_no_data(self, mock_read, spark, capsys):
        """Test process_transactions_schema when no data is found"""
        mock_read.return_value = None
        result = process_transactions_schema(spark)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Critical Error: Missing Transaction data" in captured.out
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_with_results(self, mock_read, spark):
        """Test process_transactions_schema with results array wrapper"""
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
        assert result.count() == 1
        assert "transaction_id" in result.columns
        assert "transaction_type" in result.columns
        assert "original_amount" in result.columns
        assert "currency_code" in result.columns
        assert "merchant_name" in result.columns
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_amount_calculation(self, mock_read, spark):
        """Test that amount calculation with exponent is correct"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        # Amount 12559 with exponent 2 should become 125.59
        data = [("txn1", ("12559", 2, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        row = result.collect()[0]
        assert abs(row.original_amount - 125.59) < 0.01
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_amount_calculation_exponent_3(self, mock_read, spark):
        """Test amount calculation with exponent 3"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        # Amount 1000 with exponent 3 should become 1.000
        data = [("txn1", ("1000", 3, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        row = result.collect()[0]
        assert abs(row.original_amount - 1.0) < 0.001
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_date_coalesce(self, mock_read, spark):
        """Test that transaction_date uses occurredTime when transactionDate is null"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("transactionDate", StringType(), True),
            StructField("occurredTime", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        data = [("txn1", None, "2024-01-01T10:00:00", ("1000", 2, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        # Should use occurredTime when transactionDate is null
        assert "transaction_date" in result.columns
        assert "occurred_time" not in result.columns
        
        row = result.collect()[0]
        assert row.transaction_date is not None
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_date_rename(self, mock_read, spark):
        """Test that occurredTime is renamed when transactionDate doesn't exist"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("occurredTime", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        data = [("txn1", "2024-01-01T10:00:00", ("1000", 2, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        # Should rename occurredTime to transaction_date
        assert "transaction_date" in result.columns
        assert "occurred_time" not in result.columns
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_deduplication(self, mock_read, spark):
        """Test that duplicate transaction IDs are deduplicated"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        # Same transaction ID twice
        data = [
            ("txn1", ("1000", 2, "USD")),
            ("txn1", ("2000", 2, "USD"))
        ]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        # Should be deduplicated to 1 row
        assert result.count() == 1
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_missing_fields(self, mock_read, spark):
        """Test that missing optional fields are handled with None"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ]))
        ])
        
        data = [("txn1", ("1000", 2, "USD"))]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        # Should have all columns with None values for missing fields
        assert "transaction_type" in result.columns
        assert "merchant_name" in result.columns
        assert "card_id" in result.columns
        assert "budget_id" in result.columns
        
        row = result.collect()[0]
        assert row.transaction_type is None
        assert row.merchant_name is None
    
    @patch('s3_reader_transaction.read_s3_data')
    def test_process_transactions_multiple_transactions(self, mock_read, spark):
        """Test processing multiple transactions"""
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("transactionType", StringType(), True),
            StructField("merchantName", StringType(), True),
            StructField("currencyData", StructType([
                StructField("originalCurrencyAmount", StringType(), True),
                StructField("exponent", LongType(), True),
                StructField("originalCurrencyCode", StringType(), True)
            ])),
            StructField("cardId", StringType(), True),
            StructField("budgetId", StringType(), True)
        ])
        
        data = [
            ("txn1", "AUTH", "Merchant 1", ("10000", 2, "USD"), "card1", "budget1"),
            ("txn2", "PURCHASE", "Merchant 2", ("5000", 2, "GBP"), "card2", "budget2"),
            ("txn3", "REFUND", "Merchant 3", ("2500", 2, "EUR"), "card3", "budget3")
        ]
        trans_df = spark.createDataFrame(data, schema)
        mock_read.return_value = trans_df
        
        result = process_transactions_schema(spark)
        
        assert result.count() == 3
        
        rows = {r.transaction_id: r for r in result.collect()}
        
        assert rows["txn1"].original_amount == 100.0
        assert rows["txn1"].currency_code == "USD"
        assert rows["txn2"].original_amount == 50.0
        assert rows["txn2"].currency_code == "GBP"
        assert rows["txn3"].original_amount == 25.0
        assert rows["txn3"].currency_code == "EUR"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=s3_reader_transaction", "--cov-report=term-missing"])