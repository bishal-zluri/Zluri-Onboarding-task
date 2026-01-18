import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType
from unittest.mock import patch, MagicMock, Mock, call
from datetime import datetime
import sys
import os

# Ensure current directory is in path so we can import modules
sys.path.append(os.getcwd())

# UPDATED IMPORTS: Removed _ensure_pk
from groups_postgres_loader import (
    execute_raw_sql,
    init_db,
    get_db_table,
    write_to_db,
    TABLE_GROUPS,
    TABLE_GROUP_MEMBERS
)

@pytest.fixture(scope="module")
def spark():
    """Create a real Spark session for data operations"""
    spark = SparkSession.builder \
        .appName("TestGroupsLoader") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture
def sample_groups_df(spark):
    schema = StructType([
        StructField("group_id", LongType(), True),
        StructField("group_name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("parent_group_id", LongType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True)
    ])
    data = [
        (1, "Group 1", "Test group", None, "active", datetime(2024, 1, 1), datetime(2024, 1, 2)),
        (2, "Group 2", "Another group", 1, "active", datetime(2024, 1, 1), datetime(2024, 1, 2))
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_members_df(spark):
    schema = StructType([
        StructField("group_id", LongType(), True),
        StructField("user_id", LongType(), True),
        StructField("user_status", StringType(), True)
    ])
    data = [(1, 1, "active"), (1, 2, "active"), (2, 3, "inactive")]
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
    
    def test_execute_raw_sql_handles_already_exists(self, capsys):
        """Test that 'already exists' errors don't raise exceptions"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_stmt.execute.side_effect = Exception("already exists")
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        # Should not raise exception
        execute_raw_sql(mock_spark, "CREATE TABLE test")
        mock_conn.close.assert_called_once()

    def test_execute_raw_sql_other_errors(self, capsys):
        """Test that other errors are logged and raised"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        mock_stmt.execute.side_effect = Exception("Connection timeout")
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_driver_manager = MagicMock()
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager = mock_driver_manager
        
        with pytest.raises(Exception) as exc_info:
            execute_raw_sql(mock_spark, "SELECT 1")
        
        assert "Connection timeout" in str(exc_info.value)

class TestInitDb:
    """Tests for init_db function with inline Primary Keys"""
    
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_init_db_creates_tables_with_pks(self, mock_execute, spark, capsys):
        """Test that init_db creates tables with PRIMARY KEY definitions"""
        init_db(spark)
        
        captured = capsys.readouterr()
        assert "Initializing Group Schemas" in captured.out
        
        # Expect at least 2 CREATE TABLE calls + 1 ALTER TABLE (Schema Evolution)
        assert mock_execute.call_count >= 2
        
        groups_found = False
        members_found = False
        
        for call_args in mock_execute.call_args_list:
            sql = call_args[0][1]
            
            # Check Groups Table
            if TABLE_GROUPS in sql and "CREATE TABLE" in sql:
                if "group_id BIGINT PRIMARY KEY" in sql:
                    groups_found = True
            
            # Check Members Table
            if TABLE_GROUP_MEMBERS in sql and "CREATE TABLE" in sql:
                if "PRIMARY KEY (group_id, user_id)" in sql:
                    members_found = True
        
        assert groups_found, "Groups table SQL missing inline Primary Key"
        assert members_found, "Members table SQL missing Composite Primary Key"

    @patch('groups_postgres_loader.execute_raw_sql')
    def test_init_db_schema_evolution(self, mock_execute, spark):
        """Test that schema evolution adds parent_group_id if missing"""
        init_db(spark)
        
        # Should try to add parent_group_id column
        alter_calls = [call[0][1] for call in mock_execute.call_args_list if "ALTER TABLE" in call[0][1]]
        assert any("parent_group_id" in call for call in alter_calls)


class TestGetDbTable:
    """Tests for get_db_table function"""
    
    @patch('groups_postgres_loader.init_db')
    def test_get_db_table_success(self, mock_init, spark, sample_groups_df):
        """Test successful retrieval using Mock Spark Session"""
        mock_spark = MagicMock()
        mock_spark.read.jdbc.return_value = sample_groups_df
        
        result = get_db_table(mock_spark, TABLE_GROUPS)
        
        assert result is not None
        assert result.count() == 2
        mock_init.assert_called_once()
    
    @patch('groups_postgres_loader.init_db')
    def test_get_db_table_failure(self, mock_init, spark, capsys):
        """Test when database read fails"""
        mock_spark = MagicMock()
        mock_spark.read.jdbc.side_effect = Exception("Connection failed")
        
        result = get_db_table(mock_spark, TABLE_GROUPS)
        
        assert result is None
        captured = capsys.readouterr()
        assert "Could not read table" in captured.out
    
    @patch('groups_postgres_loader.init_db')
    def test_get_db_table_casts_ids(self, mock_init, spark):
        """Test that IDs are cast to long"""
        schema = StructType([
            StructField("group_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("parent_group_id", StringType(), True)
        ])
        string_df = spark.createDataFrame([("1", "2", "3")], schema)
        
        mock_spark = MagicMock()
        mock_spark.read.jdbc.return_value = string_df
        
        result = get_db_table(mock_spark, "test_table")
        
        # All ID columns should be cast to long
        assert str(result.schema["group_id"].dataType) == "LongType()"
        assert str(result.schema["user_id"].dataType) == "LongType()"


class TestWriteToDb:
    """Tests for write_to_db function using Mock Objects"""
    
    @patch('groups_postgres_loader.init_db')
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_write_to_db_groups(self, mock_execute, mock_init, spark):
        """Test writing groups to database"""
        mock_groups_df = MagicMock()
        mock_members_df = MagicMock()
        
        # Mock .drop("user_ids").write.jdbc(...)
        mock_groups_df.drop.return_value.write.jdbc = MagicMock()
        
        write_to_db(mock_groups_df, mock_members_df, spark)
        
        assert mock_execute.called
        mock_groups_df.drop.assert_called_with("user_ids")
        mock_groups_df.drop.return_value.write.jdbc.assert_called()
    
    @patch('groups_postgres_loader.init_db')
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_write_to_db_members_sync(self, mock_execute, mock_init, spark):
        """Test members sync (delete + insert)"""
        mock_groups_df = MagicMock()
        mock_members_df = MagicMock()
        
        # Setup mocks
        mock_groups_df.drop.return_value.write.jdbc = MagicMock()
        mock_members_df.isEmpty.return_value = False
        mock_members_df.write.jdbc = MagicMock()
        
        write_to_db(mock_groups_df, mock_members_df, spark)
        
        # Verify SQL calls
        delete_call = None
        insert_call = None
        for call in mock_execute.call_args_list:
            sql = call[0][1]
            if "DELETE FROM" in sql and TABLE_GROUP_MEMBERS in sql:
                delete_call = sql
            elif "INSERT INTO" in sql and TABLE_GROUP_MEMBERS in sql:
                insert_call = sql
        
        assert delete_call is not None
        assert insert_call is not None
        assert "WHERE group_id IN" in delete_call

    @patch('groups_postgres_loader.init_db')
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_write_to_db_empty_members(self, mock_execute, mock_init, spark):
        """Test writing when members dataframe is empty"""
        mock_groups_df = MagicMock()
        mock_members_df = MagicMock()
        mock_members_df.isEmpty.return_value = True
        
        write_to_db(mock_groups_df, mock_members_df, spark)
        
        # Should NOT try to write members
        mock_members_df.write.jdbc.assert_not_called()
    
    @patch('groups_postgres_loader.init_db')
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_write_to_db_cleanup_staging_tables(self, mock_execute, mock_init, spark):
        """Test that staging tables are dropped"""
        mock_groups_df = MagicMock()
        mock_members_df = MagicMock()
        mock_members_df.isEmpty.return_value = False
        
        write_to_db(mock_groups_df, mock_members_df, spark)
        
        # Check for DROP TABLE calls
        drop_calls = [call[0][1] for call in mock_execute.call_args_list if "DROP TABLE" in call[0][1]]
        
        # Should drop groups_stage and group_members_stage
        assert len(drop_calls) >= 2
    
    @patch('groups_postgres_loader.init_db')
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_write_to_db_handles_write_failure(self, mock_execute, mock_init, spark, capsys):
        """Test graceful handling of write failures"""
        mock_groups_df = MagicMock()
        mock_members_df = MagicMock()
        
        # Simulate exception
        mock_groups_df.drop.return_value.write.jdbc.side_effect = Exception("Write failed")
        
        write_to_db(mock_groups_df, mock_members_df, spark)
        
        captured = capsys.readouterr()
        assert "Groups Write Failed" in captured.out