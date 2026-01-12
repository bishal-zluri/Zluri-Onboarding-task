# tests/test_transformer.py
import sys
import os
from unittest.mock import patch, MagicMock
from pyspark.sql.types import StructType, StructField, LongType, StringType, ArrayType, TimestampType
from pyspark.sql.functions import col

# --- BOILERPLATE ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groups_transformer import transform_and_reconcile_groups, TABLE_GROUPS, TABLE_USERS

# --- SCHEMAS ---
def get_ingested_schema():
    """Output of Step 1 (Ingestion)"""
    return StructType([
        StructField("group_id", LongType(), True),
        StructField("group_name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("user_ids", ArrayType(LongType()), True)
    ])

def get_db_users_schema():
    return StructType([
        StructField("user_id", LongType(), True),
        StructField("status", StringType(), True)
    ])

def get_db_groups_schema():
    """Simulates Postgres Table: user_ids is STRING, not Array"""
    return StructType([
        StructField("group_id", LongType(), True),
        StructField("group_name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("user_ids", StringType(), True) # <--- CRITICAL: DB stores as "1,2,3"
    ])

# --- TESTS ---

@patch("groups_transformer.write_to_db")
@patch("groups_transformer.get_db_table")
@patch("groups_transformer.process_groups_data")
def test_status_calculation_logic(mock_process, mock_get_db, mock_write, spark):
    """
    Scenario:
    - Group A has user 101. User 101 is ACTIVE. -> Group should be ACTIVE.
    - Group B has user 102. User 102 is INACTIVE. -> Group should be INACTIVE.
    """
    # 1. Mock Ingested Data (S3)
    data_s3 = [
        (1, "Group A", "Desc", None, None, [101]),
        (2, "Group B", "Desc", None, None, [102])
    ]
    df_s3_raw = spark.createDataFrame(data_s3, get_ingested_schema())
    # The transformer expects 'id' and 'agent_ids' from process_groups_data, 
    # but inside the transformer, it renames them immediately. 
    # To mock correctly, we should mock the output of `process_groups_data` *before* the select.
    # However, to simplify, we can mock the `df_raw_groups` to have the pre-rename columns:
    schema_pre = StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("agent_ids", ArrayType(LongType()), True)
    ])
    data_pre = [
        (1, "Group A", "Desc", "2023-01-01", "2023-01-01", [101]),
        (2, "Group B", "Desc", "2023-01-01", "2023-01-01", [102])
    ]
    mock_process.return_value = spark.createDataFrame(data_pre, schema_pre)

    # 2. Mock DB Users
    # User 101 is Active, User 102 is Inactive
    data_users = [(101, "active"), (102, "inactive")]
    mock_get_db.side_effect = lambda spark, table: \
        spark.createDataFrame(data_users, get_db_users_schema()) if table == TABLE_USERS else None

    # 3. Execute
    transform_and_reconcile_groups(spark)

    # 4. Verify
    args, _ = mock_write.call_args
    df_final = args[0]
    
    rows = df_final.collect()
    group_a = next(r for r in rows if r.group_id == 1)
    group_b = next(r for r in rows if r.group_id == 2)

    assert group_a.status == "active"   # Because member 101 is active
    assert group_b.status == "inactive" # Because member 102 is inactive

@patch("groups_transformer.write_to_db")
@patch("groups_transformer.get_db_table")
@patch("groups_transformer.process_groups_data")
def test_reconciliation_deleted_group(mock_process, mock_get_db, mock_write, spark):
    """
    Scenario: 
    - S3 has Group 2 (New).
    - DB has Group 1 (Old).
    - Result: Group 1 should be marked 'inactive' (Deleted from S3).
    """
    # 1. Mock S3 (Only Group 2)
    schema_pre = StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("agent_ids", ArrayType(LongType()), True)
    ])
    mock_process.return_value = spark.createDataFrame([(2, "New Grp", "D", "2023-01-01", "2023-01-01", [99])], schema_pre)

    # 2. Mock DB Users (User 99 is active)
    df_users = spark.createDataFrame([(99, "active")], get_db_users_schema())

    # 3. Mock DB Groups (Contains Group 1 as string "10,11")
    df_groups_db = spark.createDataFrame([
        (1, "Old Grp", "Desc", "active", None, None, "10,11")
    ], get_db_groups_schema())

    # Side effect to return Users then Groups
    def db_side_effect(spark, table_name):
        if table_name == TABLE_USERS: return df_users
        if table_name == TABLE_GROUPS: return df_groups_db
        return None
    mock_get_db.side_effect = db_side_effect

    # 4. Execute
    transform_and_reconcile_groups(spark)

    # 5. Verify
    args, _ = mock_write.call_args
    df_final = args[0]
    
    # Check Old Group (ID 1)
    row_old = df_final.filter(col("group_id") == 1).first()
    assert row_old.status == "inactive"
    
    # Check New Group (ID 2)
    row_new = df_final.filter(col("group_id") == 2).first()
    assert row_new.status == "active"