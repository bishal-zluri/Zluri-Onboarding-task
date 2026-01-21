import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from unittest.mock import patch
from datetime import datetime
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
    # Schema matches what Reader outputs
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("role_id", StringType(), True),
        StructField("role_name", StringType(), True),
        StructField("role_desc", StringType(), True)
    ])
    # Note: user_id is Long here to match schema
    data = [
        (101, "Alice", "alice@example.com", "2023-01-01", "2023-02-01", "r1", "Admin", "Admin Role"),
        (102, "Bob", "bob@example.com", "2023-01-01", "2023-02-01", "r2", "Editor", "Editor Role")
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
    def test_transform_handles_invalid_dates(self, mock_load, mock_get_db, mock_process, spark):
        """
        [NEW] Test that 'try_cast' handles malformed dates without crashing
        """
        schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("created_at", StringType(), True), # Input is String
            StructField("updated_at", StringType(), True),
            StructField("role_id", StringType(), True),
            StructField("role_name", StringType(), True),
            StructField("role_desc", StringType(), True)
        ])
        
        # '2022-05-' is malformed and should result in NULL timestamp
        data = [(101, "Bad Date User", "test@test.com", "2022-05-", "2023-01-01", "r1", "Admin", "")]
        df_invalid_date = spark.createDataFrame(data, schema)
        
        mock_process.return_value = df_invalid_date
        mock_get_db.return_value = None
        
        transform_and_reconcile_users(spark)
        
        # Verify the dataframe passed to loader has NULL for created_at
        df_final = mock_load.call_args[0][1]
        row = df_final.collect()[0]
        assert row.created_at is None
        assert row.updated_at is not None # Valid date remains

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_filters_numeric_roles(self, mock_load, mock_get_db, mock_process, spark):
        """
        [NEW] Test that roles named "12345" are filtered out
        """
        schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True),
            StructField("role_id", StringType(), True),
            StructField("role_name", StringType(), True),
            StructField("role_desc", StringType(), True)
        ])
        
        data = [
            (101, "User1", "u1@test.com", "2023-01-01", "2023-01-01", "r1", "Admin", "Valid"),
            (102, "User2", "u2@test.com", "2023-01-01", "2023-01-01", "r2", "12345", "Invalid") # Should be dropped
        ]
        mock_process.return_value = spark.createDataFrame(data, schema)
        mock_get_db.return_value = None
        
        transform_and_reconcile_users(spark)
        
        # Verify roles dataframe
        df_roles = mock_load.call_args[0][2] # Argument index 2 is roles
        
        roles = [r.role_name for r in df_roles.collect()]
        assert "Admin" in roles
        assert "12345" not in roles

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_reconciliation(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df):
        """Test reconciliation logic"""
        mock_process.return_value = sample_exploded_df
        
        # Existing DB Data (Use TimestampType to mimic JDBC read)
        existing_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("status", StringType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True)
        ])
        
        # User 101 exists, User 999 is old/deleted
        existing_data = [
            (101, "Alice Old", "alice@example.com", "active", datetime(2023,1,1), datetime(2023,1,1)),
            (999, "Charlie", "charlie@example.com", "active", datetime(2022,1,1), datetime(2022,1,1))
        ]
        mock_get_db.return_value = spark.createDataFrame(existing_data, existing_schema)
        
        transform_and_reconcile_users(spark)
        
        df_final = mock_load.call_args[0][1]
        rows = {row['user_id']: row for row in df_final.collect()}
        
        assert rows[101]['status'] == 'active'
        assert rows[101]['user_name'] == 'Alice' # Updated name
        assert rows[102]['status'] == 'active'   # New user
        assert rows[999]['status'] == 'inactive' # Deleted user