import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from unittest.mock import patch, MagicMock

# Import the module to test
from s3_reader_groups import process_groups_data, read_local_data, ENTITY_GROUPS

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestGroupsExtractor") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

class TestReadLocalData:
    @patch('os.path.exists')
    def test_read_local_data_path_not_exists(self, mock_exists, spark, capsys):
        """Test missing folder"""
        mock_exists.return_value = False
        result = read_local_data(spark, "fake_folder")
        assert result is None
        captured = capsys.readouterr()
        assert "[!] No data found" in captured.out

    @patch('os.path.exists')
    def test_read_local_data_read_exception(self, mock_exists):
        """Test exception handling during file read"""
        mock_exists.return_value = True
        mock_spark = MagicMock()
        # Simulate exception on read
        mock_spark.read.option.return_value.json.side_effect = Exception("Corrupt File")
        mock_spark.read.option.return_value.csv.side_effect = Exception("Corrupt File")
        mock_spark.read.parquet.side_effect = Exception("Corrupt File")
        
        result = read_local_data(mock_spark, "bad_folder")
        assert result is None

class TestProcessGroupsData:
    
    @patch('s3_reader_groups.read_local_data')
    def test_process_groups_handles_results_array(self, mock_read, spark):
        """Test exploding the 'results' array which is common in API responses"""
        schema = StructType([
            StructField("results", ArrayType(StructType([
                StructField("id", LongType(), True),
                StructField("name", StringType(), True),
                StructField("agent_ids", ArrayType(LongType()), True)
            ])))
        ])
        
        data = [([
            (1, "Group A", [101, 102]),
            (2, "Group B", [103])
        ],)]
        
        mock_df = spark.createDataFrame(data, schema)
        mock_read.return_value = mock_df
        
        result = process_groups_data(spark)
        
        assert result is not None
        assert result.count() == 2
        row1 = result.filter("id=1").collect()[0]
        assert row1.user_ids == [101, 102]

    @patch('s3_reader_groups.read_local_data')
    def test_process_groups_mappings_agent_ids(self, mock_read, spark):
        """Test mapping 'agent_ids' to 'user_ids'"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True)
        ])
        mock_read.return_value = spark.createDataFrame([(1, "G1", [1, 2], None)], schema)
        
        result = process_groups_data(spark)
        assert result.collect()[0].user_ids == [1, 2]

    @patch('s3_reader_groups.read_local_data')
    def test_process_groups_mappings_users_struct(self, mock_read, spark):
        """Test mapping 'users.id' to 'user_ids'"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("users", StructType([StructField("id", ArrayType(LongType()))])),
            StructField("parent_group_id", LongType(), True)
        ])
        data = [(1, "G1", ([10, 20],), None)]
        mock_read.return_value = spark.createDataFrame(data, schema)
        
        result = process_groups_data(spark)
        assert result.collect()[0].user_ids == [10, 20]

    @patch('s3_reader_groups.read_local_data')
    def test_process_groups_missing_parent_id(self, mock_read, spark):
        """Test that missing parent_group_id column is created as null"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True)
        ])
        mock_read.return_value = spark.createDataFrame([(1, "Root", [])], schema)
        
        result = process_groups_data(spark)
        
        assert "parent_group_id" in result.columns
        assert result.collect()[0].parent_group_id is None

    @patch('s3_reader_groups.read_local_data')
    def test_process_groups_null_user_ids(self, mock_read, spark):
        """Test that null user_ids become empty arrays"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("agent_ids", ArrayType(LongType()), True),
            StructField("parent_group_id", LongType(), True)
        ])
        mock_read.return_value = spark.createDataFrame([(1, "G1", None, None)], schema)
        
        result = process_groups_data(spark)
        row = result.collect()[0]
        
        assert row.user_ids == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])