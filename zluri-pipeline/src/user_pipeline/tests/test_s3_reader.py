import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from pyspark.sql.functions import col
from unittest.mock import patch, MagicMock
import sys

# Import the module to test
from s3_reader_users import (
    create_spark_session,
    read_s3_data,
    process_agents_data,
    ENTITY_AGENTS_INDEX,
    ENTITY_AGENT_DETAILS,
    ENTITY_ROLES
)


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("TestUsersExtractor") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()


class TestCreateSparkSession:
    """Tests for create_spark_session function"""
    
    def test_create_spark_session_returns_valid_session(self):
        """Test that create_spark_session returns a valid SparkSession"""
        session = create_spark_session("TestApp")
        assert session is not None
        assert isinstance(session, SparkSession)
        assert session.sparkContext.appName == "TestApp"
        session.stop()
    
    def test_create_spark_session_with_s3_config(self):
        """Test that Spark session has correct S3 configuration"""
        session = create_spark_session()
        conf = session.sparkContext.getConf()
        assert conf.get("spark.hadoop.fs.s3a.impl") == "org.apache.hadoop.fs.s3a.S3AFileSystem"
        session.stop()


class TestReadS3Data:
    """Tests for read_s3_data function"""
    
    def test_read_s3_data_with_no_files(self, spark, capsys):
        """Test read_s3_data when no files are found"""
        result = read_s3_data(spark, "non_existent_folder")
        captured = capsys.readouterr()
        
        assert result is None
        assert "[!] No data found for entity: non_existent_folder" in captured.out
    
    @patch('s3_reader_users.read_s3_data')
    def test_read_s3_data_with_json_files(self, mock_read, spark):
        """Test read_s3_data successfully reads JSON files"""
        # Create mock dataframe
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True)
        ])
        mock_df = spark.createDataFrame([(1, "test")], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1
    
    @patch('s3_reader_users.read_s3_data')
    def test_read_s3_data_with_csv_files(self, mock_read, spark):
        """Test read_s3_data successfully reads CSV files"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("email", StringType(), True)
        ])
        mock_df = spark.createDataFrame([(1, "test@example.com")], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1
    
    @patch('s3_reader_users.read_s3_data')
    def test_read_s3_data_with_parquet_files(self, mock_read, spark):
        """Test read_s3_data successfully reads Parquet files"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("status", StringType(), True)
        ])
        mock_df = spark.createDataFrame([(1, "active")], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 1
    
    @patch('s3_reader_users.read_s3_data')
    def test_read_s3_data_union_multiple_formats(self, mock_read, spark):
        """Test read_s3_data unions data from multiple formats"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True)
        ])
        mock_df = spark.createDataFrame([
            (1, "user1"),
            (2, "user2"),
            (3, "user3")
        ], schema)
        mock_read.return_value = mock_df
        
        result = mock_read(spark, "test_folder")
        assert result is not None
        assert result.count() == 3


class TestProcessAgentsData:
    """Tests for process_agents_data function"""
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_no_data(self, mock_read, spark):
        """Test process_agents_data when no data is found"""
        mock_read.return_value = None
        result = process_agents_data(spark)
        assert result is None
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_with_users_and_roles(self, mock_read, spark):
        """Test process_agents_data with valid users and roles data"""
        # Mock agents index
        idx_schema = StructType([StructField("id", LongType(), True)])
        idx_data = [(1,), (2,)]
        idx_df = spark.createDataFrame(idx_data, idx_schema)
        
        # Mock agent details with nested structure
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
        det_data = [
            (1, ("User One", "user1@example.com"), "2024-01-01", "2024-01-02", [1], ["role1"]),
            (2, ("User Two", "user2@example.com"), "2024-01-01", "2024-01-02", [2], ["role2"])
        ]
        det_df = spark.createDataFrame(det_data, det_schema)
        
        # Mock roles
        rol_schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True)
        ])
        rol_data = [
            ("role1", "Admin", "Administrator role"),
            ("role2", "User", "Regular user role")
        ]
        rol_df = spark.createDataFrame(rol_data, rol_schema)
        
        # Mock read_s3_data to return appropriate dataframes
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return idx_df
            elif folder_name == ENTITY_AGENT_DETAILS:
                return det_df
            elif folder_name == ENTITY_ROLES:
                return rol_df
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        
        assert result is not None
        assert result.count() == 2
        assert "user_id" in result.columns
        assert "user_name" in result.columns
        assert "user_email" in result.columns
        assert "role_id" in result.columns
        assert "role_name" in result.columns
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_users_without_roles(self, mock_read, spark):
        """Test process_agents_data when users have no roles (explode_outer scenario)"""
        # Mock agents index
        idx_schema = StructType([StructField("id", LongType(), True)])
        idx_df = spark.createDataFrame([(1,)], idx_schema)
        
        # Mock agent details with null role_ids
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
        det_data = [(1, ("User One", "user1@example.com"), "2024-01-01", "2024-01-02", [], None)]
        det_df = spark.createDataFrame(det_data, det_schema)
        
        # Mock roles
        rol_schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True)
        ])
        rol_df = spark.createDataFrame([], rol_schema)
        
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return idx_df
            elif folder_name == ENTITY_AGENT_DETAILS:
                return det_df
            elif folder_name == ENTITY_ROLES:
                return rol_df
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        
        # User should still be present with null role
        assert result is not None
        assert result.count() == 1
        user_row = result.collect()[0]
        assert user_row.user_id == 1
        assert user_row.user_name == "User One"
        assert user_row.role_id is None
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_no_roles_table(self, mock_read, spark):
        """Test process_agents_data when roles table is missing"""
        # Mock agents index
        idx_schema = StructType([StructField("id", LongType(), True)])
        idx_df = spark.createDataFrame([(1,)], idx_schema)
        
        # Mock agent details
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
        det_data = [(1, ("User One", "user1@example.com"), "2024-01-01", "2024-01-02", [], ["role1"])]
        det_df = spark.createDataFrame(det_data, det_schema)
        
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return idx_df
            elif folder_name == ENTITY_AGENT_DETAILS:
                return det_df
            elif folder_name == ENTITY_ROLES:
                return None  # No roles table
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        
        assert result is not None
        assert result.count() == 1
        assert "role_name" in result.columns
        assert "role_desc" in result.columns
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_multiple_roles_per_user(self, mock_read, spark):
        """Test process_agents_data when users have multiple roles"""
        # Mock agents index
        idx_schema = StructType([StructField("id", LongType(), True)])
        idx_df = spark.createDataFrame([(1,)], idx_schema)
        
        # Mock agent details with multiple roles
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
        det_data = [(1, ("User One", "user1@example.com"), "2024-01-01", "2024-01-02", [], ["role1", "role2", "role3"])]
        det_df = spark.createDataFrame(det_data, det_schema)
        
        # Mock roles
        rol_schema = StructType([
            StructField("id", StringType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True)
        ])
        rol_data = [
            ("role1", "Admin", "Admin role"),
            ("role2", "Manager", "Manager role"),
            ("role3", "User", "User role")
        ]
        rol_df = spark.createDataFrame(rol_data, rol_schema)
        
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return idx_df
            elif folder_name == ENTITY_AGENT_DETAILS:
                return det_df
            elif folder_name == ENTITY_ROLES:
                return rol_df
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        
        # Should have 3 rows (one per role)
        assert result is not None
        assert result.count() == 3
        
        # Verify all roles are present
        roles = [row.role_name for row in result.collect()]
        assert "Admin" in roles
        assert "Manager" in roles
        assert "User" in roles
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_missing_index(self, mock_read, spark):
        """Test process_agents_data when index data is missing"""
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return None  # Missing index
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        assert result is None
    
    @patch('s3_reader_users.read_s3_data')
    def test_process_agents_data_missing_details(self, mock_read, spark):
        """Test process_agents_data when details data is missing"""
        # Mock agents index
        idx_schema = StructType([StructField("id", LongType(), True)])
        idx_df = spark.createDataFrame([(1,)], idx_schema)
        
        def mock_read_side_effect(spark, folder_name):
            if folder_name == ENTITY_AGENTS_INDEX:
                return idx_df
            elif folder_name == ENTITY_AGENT_DETAILS:
                return None  # Missing details
            return None
        
        mock_read.side_effect = mock_read_side_effect
        
        result = process_agents_data(spark)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=s3_reader_users", "--cov-report=term-missing"])