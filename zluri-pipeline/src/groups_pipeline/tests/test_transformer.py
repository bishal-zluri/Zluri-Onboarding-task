import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    ArrayType, TimestampType
)
from unittest.mock import patch, MagicMock
from datetime import datetime

from groups_transformer import transform_and_reconcile_groups


@pytest.fixture(scope="module")
def spark():
    """Create a Spark session for testing"""
    spark = SparkSession.builder \
        .appName("TestGroupsTransformer") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def sample_groups_df(spark):
    """Create sample groups dataframe"""
    schema = StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("parent_group_id", LongType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("user_ids", ArrayType(LongType()), True)
    ])
    
    data = [
        (1, "Parent Group", "Top level", None, datetime(2024, 1, 1), datetime(2024, 1, 2), [1, 2]),
        (2, "Child Group", "Child of 1", 1, datetime(2024, 1, 1), datetime(2024, 1, 2), [3]),
        (3, "Empty Group", "No users", None, datetime(2024, 1, 1), datetime(2024, 1, 2), [])
    ]
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_users_df(spark):
    """Create sample users dataframe"""
    schema = StructType([
        StructField("user_id", LongType(), True),
        StructField("status", StringType(), True)
    ])
    
    data = [
        (1, "active"),
        (2, "active"),
        (3, "inactive"),
        (4, "active")
    ]
    
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_db_groups_df(spark):
    """Create sample existing database groups"""
    schema = StructType([
        StructField("group_id", LongType(), True),
        StructField("group_name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("parent_group_id", LongType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("user_ids", StringType(), True)
    ])
    
    data = [
        (1, "Old Parent", "Old desc", None, "active", datetime(2024, 1, 1), datetime(2024, 1, 1), "1,2"),
        (5, "Deleted Group", "Will be inactive", None, "active", datetime(2024, 1, 1), datetime(2024, 1, 1), "5,6")
    ]
    
    return spark.createDataFrame(data, schema)


class TestTransformAndReconcileGroups:
    """Tests for transform_and_reconcile_groups function"""
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_no_data(self, mock_write, mock_get_db, mock_process, spark, capsys):
        """Test transformation when no data is available"""
        mock_process.return_value = None
        
        transform_and_reconcile_groups(spark)
        
        captured = capsys.readouterr()
        assert "No group data found in S3" in captured.out
        mock_write.assert_not_called()
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_initial_load(self, mock_write, mock_get_db, mock_process, 
                                    spark, sample_groups_df, sample_users_df, capsys):
        """Test initial load when no existing database data"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        captured = capsys.readouterr()
        assert "Calculating Direct Group Status" in captured.out
        assert "Valid S3 Groups: 3" in captured.out
        assert mock_write.called
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_status_calculation_active(self, mock_write, mock_get_db, mock_process,
                                                 spark, sample_groups_df, sample_users_df):
        """Test that groups with active users are marked as active"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Group 1 has users 1 and 2 (both active) -> should be active
        group1 = df_groups.filter(df_groups.group_id == 1).collect()[0]
        assert group1.status == "active"
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_status_calculation_inactive(self, mock_write, mock_get_db, mock_process,
                                                   spark, sample_groups_df, sample_users_df):
        """Test that groups with no active users are marked as inactive"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Group 3 has no users -> should be inactive
        group3 = df_groups.filter(df_groups.group_id == 3).collect()[0]
        assert group3.status == "inactive"
        
        # Group 2 has user 3 (inactive) -> should be inactive
        group2 = df_groups.filter(df_groups.group_id == 2).collect()[0]
        assert group2.status == "inactive"
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_parent_propagation(self, mock_write, mock_get_db, mock_process,
                                         spark, sample_groups_df, sample_users_df, capsys):
        """Test that status propagates from child to parent groups"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        captured = capsys.readouterr()
        assert "Propagating Status Upwards" in captured.out
        assert "Convergence reached" in captured.out
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_group_members_generation(self, mock_write, mock_get_db, mock_process,
                                                spark, sample_groups_df, sample_users_df):
        """Test that group_members table is correctly generated"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_members = call_args[0][1]
        
        # Should have 3 members (group 1 has 2 users, group 2 has 1 user)
        assert df_members.count() == 3
        
        # Check columns exist
        assert "group_id" in df_members.columns
        assert "user_id" in df_members.columns
        assert "user_status" in df_members.columns
        
        # Verify user statuses are correctly joined
        members = df_members.collect()
        member_dict = {(m.group_id, m.user_id): m.user_status for m in members}
        
        assert member_dict[(1, 1)] == "active"
        assert member_dict[(1, 2)] == "active"
        assert member_dict[(2, 3)] == "inactive"
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_empty_users_table(self, mock_write, mock_get_db, mock_process,
                                        spark, sample_groups_df, capsys):
        """Test transformation when users table is empty"""
        mock_process.return_value = sample_groups_df
        
        empty_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("status", StringType(), True)
        ])
        empty_users = spark.createDataFrame([], empty_schema)
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return empty_users
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        captured = capsys.readouterr()
        assert "Users table empty" in captured.out
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_reconciliation_with_existing_data(self, mock_write, mock_get_db, mock_process,
                                                         spark, sample_groups_df, sample_users_df, 
                                                         sample_db_groups_df):
        """Test reconciliation with existing database groups"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            elif table_name == "groups":
                return sample_db_groups_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Should have 4 groups total (3 from S3 + 1 old from DB)
        assert df_groups.count() == 4
        
        # Group 5 should be inactive (not in new data)
        group5 = df_groups.filter(df_groups.group_id == 5).collect()[0]
        assert group5.status == "inactive"
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_preserves_created_at(self, mock_write, mock_get_db, mock_process,
                                           spark, sample_groups_df, sample_users_df, 
                                           sample_db_groups_df):
        """Test that created_at is preserved from database"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            elif table_name == "groups":
                return sample_db_groups_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Group 1 exists in DB with created_at 2024-01-01
        group1 = df_groups.filter(df_groups.group_id == 1).collect()[0]
        assert group1.created_at == datetime(2024, 1, 1)
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_updates_parent_group_id(self, mock_write, mock_get_db, mock_process,
                                               spark, sample_groups_df, sample_users_df, 
                                               sample_db_groups_df):
        """Test that parent_group_id is updated from S3 data"""
        mock_process.return_value = sample_groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return sample_users_df
            elif table_name == "groups":
                return sample_db_groups_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Verify parent_group_id is present
        group2 = df_groups.filter(df_groups.group_id == 2).collect()[0]
        assert group2.parent_group_id == 1
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_multi_level_hierarchy(self, mock_write, mock_get_db, mock_process, spark):
        """Test status propagation in multi-level hierarchy"""
        # Create 3-level hierarchy
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True),
            StructField("user_ids", ArrayType(LongType()), True)
        ])
        
        data = [
            (1, "Root", "Level 0", None, datetime(2024, 1, 1), datetime(2024, 1, 2), []),
            (2, "Child", "Level 1", 1, datetime(2024, 1, 1), datetime(2024, 1, 2), []),
            (3, "Grandchild", "Level 2", 2, datetime(2024, 1, 1), datetime(2024, 1, 2), [1])
        ]
        
        groups_df = spark.createDataFrame(data, schema)
        
        users_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("status", StringType(), True)
        ])
        users_df = spark.createDataFrame([(1, "active")], users_schema)
        
        mock_process.return_value = groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        groups = {g.group_id: g for g in df_groups.collect()}
        
        # All should be active due to propagation
        assert groups[3].status == "active"  # Has active user
        assert groups[2].status == "active"  # Parent of active child
        assert groups[1].status == "active"  # Grandparent of active grandchild
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_handles_null_groups(self, mock_write, mock_get_db, mock_process, spark):
        """Test that groups with null IDs are filtered out"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", TimestampType(), True),
            StructField("updated_at", TimestampType(), True),
            StructField("user_ids", ArrayType(LongType()), True)
        ])
        
        data = [
            (1, "Valid Group", "Has ID", None, datetime(2024, 1, 1), datetime(2024, 1, 2), [1]),
            (None, "Invalid", "No ID", None, datetime(2024, 1, 1), datetime(2024, 1, 2), [2])
        ]
        
        groups_df = spark.createDataFrame(data, schema)
        
        users_schema = StructType([
            StructField("user_id", LongType(), True),
            StructField("status", StringType(), True)
        ])
        users_df = spark.createDataFrame([(1, "active"), (2, "active")], users_schema)
        
        mock_process.return_value = groups_df
        
        def get_db_side_effect(spark, table_name):
            if table_name == "users":
                return users_df
            return None
        
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        call_args = mock_write.call_args
        df_groups = call_args[0][0]
        
        # Should only have 1 group
        assert df_groups.count() == 1
        assert df_groups.collect()[0].group_id == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=groups_transformer", "--cov-report=term-missing"])