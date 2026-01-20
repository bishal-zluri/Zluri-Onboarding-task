import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from unittest.mock import patch, MagicMock
from user_transformer import transform_and_reconcile_users

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestUserTransformer") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_exploded_df(spark):
    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("role_id", StringType(), True),
        StructField("role_name", StringType(), True),
        StructField("role_desc", StringType(), True)
    ])
    data = [
        ("101", "Alice", "alice@example.com", "2023-01-01", "2023-02-01", "r1", "Admin", "Admin Role"),
        ("102", "Bob", "bob@example.com", "2023-01-01", "2023-02-01", "r2", "Editor", "Editor Role")
    ]
    return spark.createDataFrame(data, schema)

class TestTransformAndReconcileUsers:
    
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_no_data(self, mock_load, mock_process, spark, capsys):
        """Test graceful exit when no data is found"""
        mock_process.return_value = None
        transform_and_reconcile_users(spark)
        captured = capsys.readouterr()
        assert "No new data found" in captured.out
        mock_load.assert_not_called()

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_initial_load(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df, capsys):
        """Test initial load (Database is empty)"""
        mock_process.return_value = sample_exploded_df
        mock_get_db.return_value = None  # Simulates empty DB
        
        transform_and_reconcile_users(spark)
        
        captured = capsys.readouterr()
        assert "Initial Load" in captured.out
        mock_load.assert_called_once()

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_with_reconciliation(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df, capsys):
        """Test reconciliation logic (New vs Existing vs Deleted users)"""
        mock_process.return_value = sample_exploded_df  # Contains users 101 (Alice) and 102 (Bob)
        
        # --- FIX: Added 'updated_at' to schema and data ---
        existing_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("status", StringType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True) # <--- CRITICAL FIX
        ])
        
        existing_data = [
            (101, "Alice Old", "alice@example.com", "active", "2023-01-01", "2023-01-01"),  # User exists
            (999, "Charlie", "charlie@example.com", "active", "2022-01-01", "2022-01-01")   # User missing in new data (should become inactive)
        ]
        existing_df = spark.createDataFrame(existing_data, existing_schema)
        mock_get_db.return_value = existing_df
        
        transform_and_reconcile_users(spark)
        
        # Capture arguments passed to loader
        call_args = mock_load.call_args
        df_final = call_args[0][1] # The dataframe passed to loader
        
        # Assertions
        rows = {row['user_id']: row for row in df_final.collect()}
        
        # 1. Existing User (Alice): Should take new data (Active)
        assert rows[101]['status'] == 'active'
        assert rows[101]['user_name'] == 'Alice' # New name
        
        # 2. New User (Bob): Should be added (Active)
        assert rows[102]['status'] == 'active'
        
        # 3. Missing User (Charlie): Should be marked inactive, updated_at preserved from DB
        assert 999 in rows
        assert rows[999]['status'] == 'inactive'
        # Verification that we didn't lose the old timestamp (it would be None if logic was broken)
        assert rows[999]['updated_at'] is not None 

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_handles_empty_db(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df):
        """Test when DB connection returns empty DataFrame instead of None"""
        mock_process.return_value = sample_exploded_df
        
        # --- FIX: Added 'updated_at' to schema ---
        existing_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("status", StringType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        mock_get_db.return_value = spark.createDataFrame([], existing_schema)
        
        transform_and_reconcile_users(spark)
        mock_load.assert_called()