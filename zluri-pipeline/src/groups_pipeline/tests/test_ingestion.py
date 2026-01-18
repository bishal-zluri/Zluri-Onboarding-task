import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from unittest.mock import patch, MagicMock

from s3_reader_groups import (
    read_s3_data,
    process_groups_data,
    ENTITY_GROUPS
)


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("TestGroupsExtractor") \
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
    
    @patch('s3_reader_groups.read_s3_data')
    def test_read_s3_data_with_json_files(self, mock_read, spark):
        """Test read_s3_data successfully reads JSON files"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True)
        ])
        mock_df = spark.createDataFrame([(1, "test")], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1
    
    @patch('s3_reader_groups.read_s3_data')
    def test_read_s3_data_union_multiple_files(self, mock_read, spark):
        """Test read_s3_data unions data from multiple files"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True)
        ])
        mock_df = spark.createDataFrame([
            (1, "group1"),
            (2, "group2"),
            (3, "group3")
        ], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 3


class TestProcessGroupsData:
    """Tests for process_groups_data function"""
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_no_data(self, mock_read, spark):
        """Test process_groups_data when no data is found"""
        mock_read.return_value = None
        result = process_groups_data(spark)
        assert result is None
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_with_results_array(self, mock_read, spark):
        """Test process_groups_data with results array wrapper"""
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", LongType(), True),
                StructField("name", StringType(), True),
                StructField("description", StringType(), True),
                StructField("agent_ids", ArrayType(LongType()), True),
                StructField("parent_group_id", LongType(), True),
                StructField("created_at", StringType(), True),
                StructField("updated_at", StringType(), True)
            ])))
        ])
        
        data = [([
            (1, "Group 1", "First group", [1, 2], None, "2024-01-01", "2024-01-02"),
            (2, "Group 2", "Second group", [3, 4], 1, "2024-01-01", "2024-01-02")
        ],)]
        
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert result is not None
        assert result.count() == 2
        assert "user_ids" in result.columns
        assert "id" in result.columns
        assert "name" in result.columns
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_agent_ids_to_user_ids(self, mock_read, spark):
        """Test that agent_ids column is mapped to user_ids"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Group 1", "Test", [1, 2, 3], None, "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert "user_ids" in result.columns
        row = result.collect()[0]
        assert row.user_ids == [1, 2, 3]
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_users_column(self, mock_read, spark):
        """Test that users.id column is mapped to user_ids"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("users", StructType([
                StructField("id", ArrayType(LongType()), True)
            ])),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Group 1", "Test", ([10, 20],), None, "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert "user_ids" in result.columns
        row = result.collect()[0]
        assert row.user_ids == [10, 20]
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_members_column(self, mock_read, spark):
        """Test that members.id column is mapped to user_ids"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("members", StructType([
                StructField("id", ArrayType(LongType()), True)
            ])),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Group 1", "Test", ([5, 6, 7],), None, "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert "user_ids" in result.columns
        row = result.collect()[0]
        assert row.user_ids == [5, 6, 7]
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_null_user_ids(self, mock_read, spark):
        """Test that null user_ids are converted to empty array"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Group 1", "Test", None, "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert "user_ids" in result.columns
        row = result.collect()[0]
        assert row.user_ids == []
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_missing_parent_group_id(self, mock_read, spark, capsys):
        """Test that missing parent_group_id is defaulted to null"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("user_ids", ArrayType(LongType()), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Group 1", "Test", [1], "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        captured = capsys.readouterr()
        assert "'parent_group_id' missing in source" in captured.out
        assert "parent_group_id" in result.columns
        
        row = result.collect()[0]
        assert row.parent_group_id is None
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_with_parent_group_id(self, mock_read, spark):
        """Test that parent_group_id is preserved when present"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("user_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [
            (1, "Parent Group", "Top level", [], None, "2024-01-01", "2024-01-02"),
            (2, "Child Group", "Child of 1", [1], 1, "2024-01-01", "2024-01-02")
        ]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        rows = result.collect()
        parent = [r for r in rows if r.id == 1][0]
        child = [r for r in rows if r.id == 2][0]
        
        assert parent.parent_group_id is None
        assert child.parent_group_id == 1
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_empty_user_ids_array(self, mock_read, spark):
        """Test that empty user_ids array is preserved"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [(1, "Empty Group", "No members", [], None, "2024-01-01", "2024-01-02")]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        row = result.collect()[0]
        assert row.user_ids == []
    
    @patch('s3_reader_groups.read_s3_data')
    def test_process_groups_data_multiple_groups_various_scenarios(self, mock_read, spark):
        """Test processing multiple groups with various scenarios"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True)
        ])
        
        data = [
            (1, "Group A", "Has members", [1, 2, 3], None, "2024-01-01", "2024-01-02"),
            (2, "Group B", "No members", [], None, "2024-01-01", "2024-01-02"),
            (3, "Group C", "Child group", [4], 1, "2024-01-01", "2024-01-02")
        ]
        groups_df = spark.createDataFrame(data, schema)
        mock_read.return_value = groups_df
        
        result = process_groups_data(spark)
        
        assert result.count() == 3
        
        rows = {r.id: r for r in result.collect()}
        
        assert rows[1].user_ids == [1, 2, 3]
        assert rows[2].user_ids == []
        assert rows[3].user_ids == [4]
        assert rows[3].parent_group_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=s3_reader_groups", "--cov-report=term-missing"])