import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from unittest.mock import patch, MagicMock, Mock, call
from datetime import datetime
import sys
import os

# Ensure current directory is in path so we can import modules
sys.path.append(os.getcwd())

# UPDATED IMPORTS: Removed _ensure_pk_constraint
from user_postgres_loader import (
    execute_raw_sql,
    init_db,
    get_existing_db_data,
    load_user_pipeline,
    TABLE_USERS,
    TABLE_ROLES,
    TABLE_USER_ROLES
)

@pytest.fixture(scope="module")
def spark():
    """Create a real Spark session for data operations"""
    spark = SparkSession.builder \
        .appName("TestUsersLoader") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_users_df(spark):
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True)
    ])
    data = [
        (1, "User One", "user1@test.com", "active", datetime(2024, 1, 1), datetime(2024, 1, 2)),
        (2, "User Two", "user2@test.com", "active", datetime(2024, 1, 1), datetime(2024, 1, 2)),
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_roles_df(spark):
    schema = StructType([
        StructField("role_id", StringType(), True),
        StructField("role_name", StringType(), True),
        StructField("role_desc", StringType(), True)
    ])
    data = [("role1", "Admin", "Administrator"), ("role2", "User", "Regular User")]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_user_roles_df(spark):
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("role_id", StringType(), True),
        StructField("role_name", StringType(), True)
    ])
    data = [(1, "role1", "Admin"), (2, "role2", "User")]
    return spark.createDataFrame(data, schema)

class TestExecuteRawSql:
    """Tests for execute_raw_sql function"""
    
    def test_execute_raw_sql_success(self):
        """Test successful SQL execution with a Pure Python Mock"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        execute_raw_sql(mock_spark, "SELECT 1")
        
        mock_stmt.execute.assert_called_once_with("SELECT 1")
        mock_conn.close.assert_called_once()
    
    def test_execute_raw_sql_failure(self):
        """Test SQL execution failure with Pure Python Mock"""
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

class TestInitDb:
    """Updated Tests for init_db function with inline Primary Keys"""
    
    @patch('user_postgres_loader.execute_raw_sql')
    def test_init_db_creates_all_tables_with_pks(self, mock_execute, spark, capsys):
        """Test that init_db creates tables with PRIMARY KEY definitions"""
        init_db(spark)
        
        captured = capsys.readouterr()
        assert "Initializing Schemas" in captured.out
        
        # We expect 3 calls (one for each table)
        assert mock_execute.call_count == 3
        
        # Verify SQL content for Users Table
        users_sql_found = False
        roles_sql_found = False
        ur_sql_found = False

        for call_args in mock_execute.call_args_list:
            sql = call_args[0][1] # The SQL string passed to execute_raw_sql
            
            # Check Users
            if TABLE_USERS in sql and "user_id BIGINT PRIMARY KEY" in sql:
                users_sql_found = True
            
            # Check Roles
            if TABLE_ROLES in sql and "role_id TEXT PRIMARY KEY" in sql:
                roles_sql_found = True
                
            # Check User Roles (Composite PK)
            if TABLE_USER_ROLES in sql and "PRIMARY KEY (user_id, role_id)" in sql:
                ur_sql_found = True

        assert users_sql_found, "Users table SQL missing 'user_id BIGINT PRIMARY KEY'"
        assert roles_sql_found, "Roles table SQL missing 'role_id TEXT PRIMARY KEY'"
        assert ur_sql_found, "User Roles table SQL missing composite 'PRIMARY KEY (user_id, role_id)'"

class TestGetExistingDbData:
    @patch('user_postgres_loader.init_db')
    def test_get_existing_db_data_success(self, mock_init, spark, sample_users_df):
        # Create a Mock Spark Session
        mock_spark = MagicMock()
        
        # Configure the mock chain: mock_spark.read.jdbc(...) returns sample_users_df
        mock_spark.read.jdbc.return_value = sample_users_df
        
        result = get_existing_db_data(mock_spark)
        
        assert result is not None
        assert result.count() == 2
        mock_init.assert_called_once()
        mock_spark.read.jdbc.assert_called_once()
            
    @patch('user_postgres_loader.init_db')
    def test_get_existing_db_data_failure(self, mock_init, spark, capsys):
        mock_spark = MagicMock()
        mock_spark.read.jdbc.side_effect = Exception("Connection failed")
        
        result = get_existing_db_data(mock_spark)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Could not read existing users" in captured.out

class TestLoadUserPipeline:
    
    @patch('user_postgres_loader.init_db')
    @patch('user_postgres_loader.execute_raw_sql')
    def test_load_user_pipeline_loads_all_tables(self, mock_execute, mock_init, spark):
        """Pass Mock DataFrames instead of real ones"""
        # Create Mock DataFrames
        mock_users_df = MagicMock()
        mock_roles_df = MagicMock()
        mock_ur_df = MagicMock()

        # Ensure chaining works
        mock_users_df.withColumn.return_value.select.return_value.write.jdbc = MagicMock()
        mock_roles_df.write.jdbc = MagicMock()
        mock_ur_df.withColumn.return_value.write.jdbc = MagicMock()

        # Execute
        load_user_pipeline(spark, mock_users_df, mock_roles_df, mock_ur_df)
        
        # Verify Interactions
        assert mock_init.called
        assert mock_execute.called
        
        # Verify Write calls
        mock_roles_df.write.jdbc.assert_called()
        mock_users_df.withColumn.return_value.select.return_value.write.jdbc.assert_called()
        mock_ur_df.withColumn.return_value.write.jdbc.assert_called()

    @patch('user_postgres_loader.init_db')
    @patch('user_postgres_loader.execute_raw_sql')
    def test_load_user_pipeline_handles_write_failure(self, mock_execute, mock_init, spark, capsys):
        mock_users_df = MagicMock()
        mock_roles_df = MagicMock()
        mock_ur_df = MagicMock()
        
        mock_roles_df.write.jdbc.side_effect = Exception("Write failed")
        
        load_user_pipeline(spark, mock_users_df, mock_roles_df, mock_ur_df)
        
        captured = capsys.readouterr()
        assert "Failed to load Roles" in captured.out

    @patch('user_postgres_loader.init_db')
    @patch('user_postgres_loader.execute_raw_sql')
    def test_load_user_pipeline_cleanup_staging_tables(self, mock_execute, mock_init, spark):
        mock_users_df = MagicMock()
        mock_roles_df = MagicMock()
        mock_ur_df = MagicMock()

        load_user_pipeline(spark, mock_users_df, mock_roles_df, mock_ur_df)
        
        # Check for DROP TABLE calls
        drop_calls = 0
        for call_args in mock_execute.call_args_list:
            sql = call_args[0][1]
            if "DROP TABLE" in sql:
                drop_calls += 1
                
        assert drop_calls >= 3