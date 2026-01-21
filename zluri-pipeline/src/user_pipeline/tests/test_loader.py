import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from unittest.mock import patch, MagicMock
from datetime import datetime
import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

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
        mock_conn.close.assert_called_once()
    
    def test_execute_raw_sql_failure(self):
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_stmt.execute.side_effect = Exception("SQL Error")
        mock_conn.createStatement.return_value = mock_stmt
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        with pytest.raises(Exception):
            execute_raw_sql(mock_spark, "INVALID SQL")
        mock_conn.close.assert_called_once()

class TestInitDb:
    @patch('user_postgres_loader.execute_raw_sql')
    def test_init_db_creates_tables_correctly(self, mock_execute, spark, capsys):
        init_db(spark)
        assert mock_execute.call_count == 3
        
        users_table_checked = False
        roles_table_checked = False
        ur_table_checked = False

        for call_args in mock_execute.call_args_list:
            sql = call_args[0][1]
            # Use specific CREATE TABLE checks to avoid substring confusion
            if f"CREATE TABLE IF NOT EXISTS {TABLE_USERS}" in sql:
                assert "user_id BIGINT PRIMARY KEY" in sql
                users_table_checked = True
            elif f"CREATE TABLE IF NOT EXISTS {TABLE_ROLES}" in sql:
                assert "role_id TEXT PRIMARY KEY" in sql
                roles_table_checked = True
            elif f"CREATE TABLE IF NOT EXISTS {TABLE_USER_ROLES}" in sql:
                assert "PRIMARY KEY (user_id, role_id)" in sql
                ur_table_checked = True

        assert users_table_checked
        assert roles_table_checked
        assert ur_table_checked

class TestGetExistingDbData:
    @patch('user_postgres_loader.init_db')
    def test_get_existing_db_data_success(self, mock_init, spark, sample_users_df):
        mock_spark = MagicMock()
        mock_spark.read.jdbc.return_value = sample_users_df
        
        result = get_existing_db_data(mock_spark)
        assert result is not None
        mock_init.assert_called_once()
            
    @patch('user_postgres_loader.init_db')
    def test_get_existing_db_data_failure(self, mock_init, spark, capsys):
        mock_spark = MagicMock()
        mock_spark.read.jdbc.side_effect = Exception("Connection failed")
        result = get_existing_db_data(mock_spark)
        assert result is None

class TestLoadUserPipeline:
    @patch('user_postgres_loader.init_db')
    @patch('user_postgres_loader.execute_raw_sql')
    def test_load_user_pipeline_execution_flow(self, mock_execute, mock_init, spark):
        mock_users_df = MagicMock()
        mock_roles_df = MagicMock()
        mock_ur_df = MagicMock()
        
        # Setup chaining mocks
        mock_roles_df.write.jdbc = MagicMock()
        mock_users_df.withColumn.return_value.select.return_value.write.jdbc = MagicMock()
        mock_ur_df.withColumn.return_value.write.jdbc = MagicMock()

        load_user_pipeline(spark, mock_users_df, mock_roles_df, mock_ur_df)
        
        mock_init.assert_called_once()
        mock_roles_df.write.jdbc.assert_called()
        mock_users_df.withColumn.return_value.select.return_value.write.jdbc.assert_called()
        
        # Verify SQL logic
        sql_calls = [args[0][1] for args in mock_execute.call_args_list]
        assert any("INSERT INTO roles" in sql for sql in sql_calls)
        assert any("INSERT INTO users" in sql for sql in sql_calls)
        assert any("DELETE FROM user_roles" in sql for sql in sql_calls)

    @patch('user_postgres_loader.init_db')
    @patch('user_postgres_loader.execute_raw_sql')
    def test_load_user_pipeline_handles_write_exceptions(self, mock_execute, mock_init, spark, capsys):
        mock_users_df = MagicMock()
        mock_roles_df = MagicMock()
        mock_ur_df = MagicMock()
        
        mock_roles_df.write.jdbc.side_effect = Exception("JDBC Timeout")
        
        load_user_pipeline(spark, mock_users_df, mock_roles_df, mock_ur_df)
        captured = capsys.readouterr()
        assert "Failed to load Roles" in captured.out