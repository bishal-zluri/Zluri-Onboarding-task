import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql.types import StructType, StructField, LongType, StringType

# Import the actual module to test
from groups_postgres_loader import (
    execute_raw_sql,
    init_db,
    get_db_table,
    write_to_db,
    TABLE_GROUPS,
    TABLE_GROUP_MEMBERS
)

class TestExecuteRawSql:
    def test_execute_raw_sql_success(self):
        """Test the happy path for SQL execution using mocks"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        
        mock_driver_manager = mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_conn.createStatement.return_value = mock_stmt
        
        execute_raw_sql(mock_spark, "SELECT 1")
        
        mock_stmt.execute.assert_called_with("SELECT 1")
        mock_conn.close.assert_called_once()

    def test_execute_raw_sql_exception(self):
        """Test that non-ignorable exceptions are raised"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        
        mock_driver_manager = mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_stmt.execute.side_effect = Exception("Fatal SQL Error")
        
        with pytest.raises(Exception) as exc:
            execute_raw_sql(mock_spark, "BAD SQL")
        
        assert "Fatal SQL Error" in str(exc.value)
        mock_conn.close.assert_called_once()

    def test_execute_raw_sql_ignore_already_exists(self):
        """Test that 'already exists' errors are safely ignored"""
        mock_spark = MagicMock()
        mock_conn = MagicMock()
        mock_stmt = MagicMock()
        
        mock_driver_manager = mock_spark.sparkContext._gateway.jvm.java.sql.DriverManager
        mock_driver_manager.getConnection.return_value = mock_conn
        mock_conn.createStatement.return_value = mock_stmt
        
        mock_stmt.execute.side_effect = Exception("relation \"groups\" already exists")
        
        execute_raw_sql(mock_spark, "CREATE TABLE groups ...")
        mock_conn.close.assert_called_once()

class TestInitDb:
    @patch('groups_postgres_loader.execute_raw_sql')
    def test_init_db_calls(self, mock_execute):
        """Test init_db calls create table statements"""
        mock_spark = MagicMock()
        init_db(mock_spark)
        
        assert mock_execute.call_count >= 2
        
        sqls = [args[0][1] for args in mock_execute.call_args_list]
        
        groups_sql = next((s for s in sqls if TABLE_GROUPS in s and "CREATE TABLE" in s), None)
        assert groups_sql is not None
        assert "group_id BIGINT PRIMARY KEY" in groups_sql
        
        members_sql = next((s for s in sqls if TABLE_GROUP_MEMBERS in s and "CREATE TABLE" in s), None)
        assert members_sql is not None
        assert "PRIMARY KEY (group_id, user_id)" in members_sql

class TestGetDbTable:
    @patch('groups_postgres_loader.init_db')
    def test_get_db_table_failure(self, mock_init, capsys):
        """Test that exceptions are handled gracefully"""
        mock_spark = MagicMock()
        mock_spark.read.jdbc.side_effect = Exception("Connection Refused")
        
        res = get_db_table(mock_spark, "test_table")
        
        assert res is None
        captured = capsys.readouterr()
        assert "Could not read table" in captured.out

class TestWriteToDb:
    @patch('groups_postgres_loader.execute_raw_sql')
    @patch('groups_postgres_loader.init_db')
    def test_write_to_db_handles_exceptions(self, mock_init, mock_execute, capsys):
        """Test that write failure doesn't crash the program"""
        mock_spark = MagicMock()
        
        mock_groups = MagicMock()
        mock_after_drop = MagicMock()
        mock_after_cast = MagicMock()
        
        mock_groups.drop.return_value = mock_after_drop
        mock_after_drop.withColumn.return_value = mock_after_cast
        
        mock_write = MagicMock()
        mock_after_cast.write = mock_write
        mock_write.jdbc.side_effect = Exception("Write Fail")
        
        mock_members = MagicMock()
        mock_members.isEmpty.return_value = True
        
        write_to_db(mock_groups, mock_members, mock_spark)
        
        captured = capsys.readouterr()
        assert "Groups Write Failed" in captured.out

if __name__ == "__main__":
    pytest.main([__file__, "-v"])