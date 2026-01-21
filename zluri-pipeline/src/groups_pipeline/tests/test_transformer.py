import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
from unittest.mock import patch
from datetime import datetime

# Import module
from groups_transformer import transform_and_reconcile_groups

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
        .appName("TestGroupsTransformer") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "2") \
        .getOrCreate()
    yield spark
    spark.stop()

class TestTransformGroups:
    
    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.write_to_db')
    def test_transform_no_data(self, mock_write, mock_process, spark, capsys):
        mock_process.return_value = None
        transform_and_reconcile_groups(spark)
        captured = capsys.readouterr()
        assert "No group data found" in captured.out
        mock_write.assert_not_called()

    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_filters_invalid_data(self, mock_write, mock_get_db, mock_process, spark):
        """
        Tests filtering of:
        1. Valid Group (Keep)
        2. Null ID (Drop)
        3. Null Name (Drop)
        4. Self-Loop ID==Parent (Drop)
        """
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True),
            StructField("user_ids", ArrayType(LongType()), True)
        ])
        
        # NOTE: Explicitly using None for nulls and matching types
        data = [
            (1, "Valid Group", "desc", None, "2024-01-01", "2024-01-01", []),          # Keep
            (None, "No ID", "desc", None, "2024-01-01", "2024-01-01", []),             # Drop
            (2, None, "desc", None, "2024-01-01", "2024-01-01", []),                   # Drop
            (3, "Self Loop", "desc", 3, "2024-01-01", "2024-01-01", [])                # Drop - FIXED
        ]
        
        mock_process.return_value = spark.createDataFrame(data, schema)
        mock_get_db.return_value = None
        
        transform_and_reconcile_groups(spark)
        
        df_final = mock_write.call_args[0][0]
        ids = [r.group_id for r in df_final.collect()]
        
        # Verify assertions
        assert 1 in ids, "Valid group should be kept"
        assert 3 not in ids, "Self-loop group should be dropped"
        assert len(ids) == 1, f"Expected 1 valid group, found {len(ids)}: {ids}"

    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_hierarchy_propagation(self, mock_write, mock_get_db, mock_process, spark):
        """
        Test Propagation:
        User(100) Active -> Group(3) Active -> Group(2) Active -> Group(1) Active
        """
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True),
            StructField("user_ids", ArrayType(LongType()), True)
        ])
        
        # 1 (Root) <- 2 (Child) <- 3 (Leaf with Active User)
        data = [
            (1, "Root", "d", None, "2024-01-01", "2024-01-01", []),
            (2, "Child", "d", 1, "2024-01-01", "2024-01-01", []),
            (3, "Leaf", "d", 2, "2024-01-01", "2024-01-01", [100])
        ]
        mock_process.return_value = spark.createDataFrame(data, schema)
        
        # User 100 is Active
        users_df = spark.createDataFrame([(100, "active")], ["user_id", "status"])
        
        def get_db_side_effect(spark, table):
            if table == "users": return users_df
            return None
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        df_final = mock_write.call_args[0][0]
        rows = {r.group_id: r.status for r in df_final.collect()}
        
        assert rows[3] == "active"
        assert rows[2] == "active" # Propagated
        assert rows[1] == "active" # Propagated

    @patch('groups_transformer.process_groups_data')
    @patch('groups_transformer.get_db_table')
    @patch('groups_transformer.write_to_db')
    def test_transform_reconciliation(self, mock_write, mock_get_db, mock_process, spark):
        """Test that missing groups are marked inactive"""
        schema = StructType([
            StructField("id", LongType(), True),
            StructField("name", StringType(), True),
            StructField("description", StringType(), True),
            StructField("parent_group_id", LongType(), True),
            StructField("created_at", StringType(), True),
            StructField("updated_at", StringType(), True),
            StructField("user_ids", ArrayType(LongType()), True)
        ])
        
        # Only Group 2 exists in New Data
        mock_process.return_value = spark.createDataFrame([(2, "G2", "d", None, "2024-01-01", "2024-01-01", [])], schema)
        
        # DB has Group 1 (Old) and Group 2
        db_groups_df = spark.createDataFrame([
            (1, "G1", "d", None, "active", datetime(2023,1,1), datetime(2023,1,1)),
            (2, "G2", "d", None, "active", datetime(2023,1,1), datetime(2023,1,1))
        ], "group_id long, group_name string, description string, parent_group_id long, status string, created_at timestamp, updated_at timestamp")
        
        def get_db_side_effect(spark, table):
            if table == "users": return None
            if table == "groups": return db_groups_df
            return None
        mock_get_db.side_effect = get_db_side_effect
        
        transform_and_reconcile_groups(spark)
        
        df_final = mock_write.call_args[0][0]
        rows = {r.group_id: r.status for r in df_final.collect()}
        
        assert rows[1] == "inactive" # Missing from source
        assert rows[2] == "inactive" # Exists but no active users

if __name__ == "__main__":
    pytest.main([__file__, "-v"])