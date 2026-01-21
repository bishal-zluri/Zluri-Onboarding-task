import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from unittest.mock import patch, MagicMock

# Import the module to test
from s3_reader_users import (
    create_spark_session,
    process_agents_data,
    read_local_data,
    ENTITY_AGENTS_INDEX,
    ENTITY_AGENT_DETAILS,
    ENTITY_ROLES
)

@pytest.fixture(scope="module")
def spark():
    """Shared Spark Session for tests"""
    spark = SparkSession.builder \
        .appName("TestUsersExtractor") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

class TestReadLocalData:
    """Tests specifically for the file reading utility"""

    @patch('os.path.exists')
    def test_read_local_data_path_not_exists(self, mock_exists, spark, capsys):
        """Test branch: if not os.path.exists(target_dir): continue"""
        mock_exists.return_value = False
        
        result = read_local_data(spark, "fake_folder")
        
        # Should return None and print specific message
        assert result is None
        captured = capsys.readouterr()
        assert "[!] No data found" in captured.out

    @patch('os.path.exists')
    def test_read_local_data_exception_handling(self, mock_exists):
        """
        [FIXED] Test branch: try... except Exception: continue
        Uses a MagicMock for spark to avoid 'AttributeError: property read has no deleter'
        """
        mock_exists.return_value = True
        
        # Create a Mock Spark Session
        mock_spark = MagicMock()
        
        # Configure the mock to raise an exception when reading JSON/CSV/Parquet
        # The chain is: spark.read.option(...).json(...)
        # We make the final call raise the exception
        mock_spark.read.option.return_value.json.side_effect = Exception("Corrupted JSON")
        mock_spark.read.option.return_value.csv.side_effect = Exception("Corrupted CSV")
        mock_spark.read.parquet.side_effect = Exception("Corrupted Parquet")
        
        # Call the function with the MOCK spark object, not the real one
        result = read_local_data(mock_spark, "error_folder")
            
        # Should catch the exception and return None (since found_dfs will be empty)
        assert result is None

class TestProcessAgentsData:
    """Tests for the main logic (process_agents_data)"""
    
    @patch('s3_reader_users.read_local_data')
    def test_process_agents_data_missing_index(self, mock_read, spark):
        """Test branch: if not df_idx: return None"""
        def mock_read_side_effect(spark, folder_name):
            return None # Index is missing
        
        mock_read.side_effect = mock_read_side_effect
        result = process_agents_data(spark)
        assert result is None

    @patch('s3_reader_users.read_local_data')
    def test_process_agents_data_missing_details_full_fallback(self, mock_read, spark):
        """Test branch: if not has_details (Fallback logic)"""
        # Index Schema with NO optional columns (tests fallback defaults)
        idx_schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            # Missing email, created_at, updated_at
        ])
        idx_df = spark.createDataFrame([(1, "Fallback User")], idx_schema)
        
        # Roles (Empty)
        rol_df = spark.createDataFrame([], StructType([
            StructField("id", StringType(), True), 
            StructField("name", StringType(), True), 
            StructField("description", StringType(), True)
        ]))
        
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX: return idx_df
            if folder_name == ENTITY_AGENT_DETAILS: return None # Details Missing
            if folder_name == ENTITY_ROLES: return rol_df
            return None
        
        mock_read.side_effect = mock_read_side_effect
        result = process_agents_data(spark)
        
        assert result is not None
        row = result.collect()[0]
        
        # Assertions for fallback defaults
        assert row.user_name == "Fallback User"
        assert row.user_email == "no-email@placeholder.com" # Default applied
        assert row.created_at is None
        assert row.role_id is None # Explode on None casted array

    @patch('s3_reader_users.read_local_data')
    def test_process_agents_data_composite_name_fallback(self, mock_read, spark):
        """Test branch: elif 'first_name' in idx_cols (Composite Name)"""
        idx_schema = StructType([
            StructField("id", LongType(), True),
            StructField("first_name", StringType(), True),
            StructField("last_name", StringType(), True),
            StructField("email", StringType(), True)
        ])
        idx_df = spark.createDataFrame([(1, "John", "Doe", "jdoe@test.com")], idx_schema)
        
        # Just return idx_df for index, None for others
        mock_read.side_effect = lambda s, f: idx_df if f == ENTITY_AGENTS_INDEX else None
        
        result = process_agents_data(spark)
        row = result.collect()[0]
        
        # Should concat First + Last
        assert row.user_name == "John Doe"

    @patch('s3_reader_users.read_local_data')
    def test_process_agents_data_unknown_name_fallback(self, mock_read, spark):
        """Test branch: else: fallback_name = lit('Unknown Name')"""
        # Schema with NO name fields
        idx_schema = StructType([
            StructField("id", LongType(), True),
            StructField("email", StringType(), True)
        ])
        idx_df = spark.createDataFrame([(1, "unknown@test.com")], idx_schema)
        
        mock_read.side_effect = lambda s, f: idx_df if f == ENTITY_AGENTS_INDEX else None
        
        result = process_agents_data(spark)
        row = result.collect()[0]
        
        assert row.user_name == "Unknown Name"

    @patch('s3_reader_users.read_local_data')
    def test_process_agents_data_missing_roles_file(self, mock_read, spark):
        """Test branch: else (if df_rol is None)"""
        idx_schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True)
        ])
        idx_df = spark.createDataFrame([(1, "User No Role")], idx_schema)
        
        # Details exists this time
        det_schema = StructType([
            StructField("id", LongType(), True),
            StructField("contact", StructType([
                StructField("name", StringType(), True),
                StructField("email", StringType(), True)
            ])),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True),
            StructField("group_ids", ArrayType(LongType()), True),
            StructField("role_ids", ArrayType(StringType()), True)
        ])
        det_data = [(1, ("User No Role", "u@test.com"), "2024-01-01", "2024-01-01", [], ["r1"])]
        det_df = spark.createDataFrame(det_data, det_schema)

        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX: return idx_df
            if folder_name == ENTITY_AGENT_DETAILS: return det_df
            if folder_name == ENTITY_ROLES: return None # Roles file missing!
            return None
        
        mock_read.side_effect = mock_read_side_effect
        result = process_agents_data(spark)
        
        assert result is not None
        row = result.collect()[0]
        
        # Assertions: Should have null role_name/desc because join was skipped
        assert row.user_name == "User No Role"
        assert row.role_id == "r1"
        assert row.role_name is None
        assert row.role_desc is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])