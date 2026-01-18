import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    ArrayType, TimestampType
)
from unittest.mock import patch, MagicMock, call
from datetime import datetime

# Import the module to test
from user_transformer import transform_and_reconcile_users

@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("TestUsersTransformer") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_exploded_df(spark):
    """Create sample exploded user data"""
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("role_id", StringType(), True),
        StructField("role_name", StringType(), True),
        StructField("role_desc", StringType(), True)
    ])
    
    data = [
        (1, "User One", "user1@test.com", datetime(2024, 1, 1), datetime(2024, 1, 2), "role1", "Admin", "Admin role"),
        (1, "User One", "user1@test.com", datetime(2024, 1, 1), datetime(2024, 1, 2), "role2", "Manager", "Manager role"),
        (2, "User Two", "user2@test.com", datetime(2024, 1, 1), datetime(2024, 1, 2), "role3", "User", "User role"),
        (3, "User Three", "user3@test.com", datetime(2024, 1, 1), datetime(2024, 1, 2), None, None, None)
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_db_data(spark):
    """Create sample existing database data"""
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True)
    ])
    
    data = [
        (1, "User One Old", "user1@old.com", "active", datetime(2024, 1, 1)),
        (4, "User Four", "user4@test.com", "active", datetime(2024, 1, 1)) 
    ]
    return spark.createDataFrame(data, schema)

class TestTransformAndReconcileUsers:
    
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_no_new_data(self, mock_load, mock_get_db, mock_process, spark, capsys):
        mock_process.return_value = None
        transform_and_reconcile_users(spark)
        captured = capsys.readouterr()
        assert "⚠️ No new data found. Skipping." in captured.out
        mock_load.assert_not_called()
    
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_with_empty_dataframe(self, mock_load, mock_get_db, mock_process, spark, capsys):
        schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True)
        ])
        empty_df = spark.createDataFrame([], schema)
        mock_process.return_value = empty_df
        transform_and_reconcile_users(spark)
        captured = capsys.readouterr()
        assert "⚠️ No new data found. Skipping." in captured.out
        mock_load.assert_not_called()
    
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_initial_load(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df, capsys):
        mock_process.return_value = sample_exploded_df
        mock_get_db.return_value = None
        
        transform_and_reconcile_users(spark)
        
        captured = capsys.readouterr()
        assert "Initial Load. Setting all users to 'active'." in captured.out
        assert mock_load.called
        call_args = mock_load.call_args
        df_users = call_args[0][1]
        active_count = df_users.filter(df_users.status == "active").count()
        assert active_count == 3 

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_with_reconciliation(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df, sample_db_data, capsys):
        mock_process.return_value = sample_exploded_df
        mock_get_db.return_value = sample_db_data
        
        transform_and_reconcile_users(spark)
        
        captured = capsys.readouterr()
        assert "Status Summary" in captured.out
        
        call_args = mock_load.call_args
        df_users = call_args[0][1]
        users = df_users.collect()
        user_dict = {row.user_id: row for row in users}
        
        assert user_dict[1].status == "active"
        assert user_dict[4].status == "inactive"
        assert user_dict[2].status == "active"
        
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_roles_table_generation(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df):
        mock_process.return_value = sample_exploded_df
        mock_get_db.return_value = None
        transform_and_reconcile_users(spark)
        call_args = mock_load.call_args
        df_roles = call_args[0][2]
        assert df_roles.count() == 3
        
    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_filters_null_user_ids(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df):
        """Test that null user_ids are filtered out"""
        schema = sample_exploded_df.schema
        # Row with None user_id
        null_data = [(None, "Null User", "null@test.com", datetime(2024, 1, 1), datetime(2024, 1, 2), None, None, None)]
        null_df = spark.createDataFrame(null_data, schema)
        combined_df = sample_exploded_df.union(null_df)
        
        mock_process.return_value = combined_df
        mock_get_db.return_value = None
        
        transform_and_reconcile_users(spark)
        
        call_args = mock_load.call_args
        df_users = call_args[0][1]
        
        # Verify null user_ids are filtered
        null_count = df_users.filter(df_users.user_id.isNull()).count()
        assert null_count == 0

    @patch('user_transformer.process_agents_data')
    @patch('user_transformer.get_existing_db_data')
    @patch('user_transformer.load_user_pipeline')
    def test_transform_handles_empty_db(self, mock_load, mock_get_db, mock_process, spark, sample_exploded_df):
        """Test transformation when database returns empty dataframe"""
        mock_process.return_value = sample_exploded_df
        
        # FIX: Added required columns (user_email, created_at) to avoid AnalysisException
        schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("user_name", StringType(), True),
            StructField("user_email", StringType(), True),
            StructField("status", StringType(), True),
            StructField("created_at", TimestampType(), True)
        ])
        empty_db_df = spark.createDataFrame([], schema)
        mock_get_db.return_value = empty_db_df
        
        transform_and_reconcile_users(spark)
        
        call_args = mock_load.call_args
        df_users = call_args[0][1]
        active_count = df_users.filter(df_users.status == "active").count()
        assert active_count == 3