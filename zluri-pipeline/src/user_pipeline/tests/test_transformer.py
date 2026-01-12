# tests/test_transformer.py
import sys
import os
from unittest.mock import patch
from pyspark.sql.types import StructType, StructField, LongType, StringType, TimestampType, ArrayType
from datetime import datetime

# --- BOILERPLATE ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_transformer import transform_and_reconcile_users

# --- SCHEMAS ---
def get_ingested_schema():
    """Output of process_agents_data (Step 1)"""
    return StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("user_created_at", StringType(), True),
        StructField("user_updated_at", StringType(), True),
        StructField("role_names", ArrayType(StringType()), True)
    ])

def get_db_schema():
    """Simulates the Postgres 'users' table"""
    return StructType([
        StructField("user_id", LongType(), True),
        StructField("user_name", StringType(), True),
        StructField("user_email", StringType(), True),
        StructField("status", StringType(), True),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("role_names", StringType(), True)
    ])

# --- TESTS ---

@patch("user_transformer.write_to_db_and_show")
@patch("user_transformer.get_existing_db_data")
@patch("user_transformer.process_agents_data")
def test_initial_load(mock_process, mock_get_db, mock_write, spark):
    """
    Scenario: DB is empty.
    Expected: All users from S3 are written as 'active'.
    """
    # 1. Mock S3 Data (Matches your JSON sample data types)
    df_incoming = spark.createDataFrame([
        (
            69010941952, 
            "H1 SaaS", 
            "saas@h1.co", 
            "2021-08-13T11:50:18Z", 
            "2025-03-26T18:19:00Z", 
            ["Support Agent"]
        )
    ], get_ingested_schema())
    mock_process.return_value = df_incoming

    # 2. Mock DB (None for first run)
    mock_get_db.return_value = None

    # 3. Execute
    transform_and_reconcile_users(spark)

    # 4. Verify
    args, _ = mock_write.call_args
    df_written = args[0]
    
    row = df_written.first()
    assert row['user_id'] == 69010941952
    assert row['status'] == 'active'
    # Check Array->String conversion
    assert row['role_names'] == "Support Agent"
    # Check Timestamp casting
    assert isinstance(row['created_at'], datetime)
    assert row['created_at'].year == 2021

@patch("user_transformer.write_to_db_and_show")
@patch("user_transformer.get_existing_db_data")
@patch("user_transformer.process_agents_data")
def test_reconciliation_deleted_user(mock_process, mock_get_db, mock_write, spark):
    """
    Scenario: User 69010941952 exists in DB but is missing from S3.
    Expected: Status becomes 'inactive'.
    """
    # 1. Mock S3 (Empty or different user)
    df_incoming = spark.createDataFrame([
        (999, "New Guy", "new@h1.co", "2024-01-01", "2024-01-01", ["Admin"])
    ], get_ingested_schema())
    mock_process.return_value = df_incoming

    # 2. Mock DB (Original User is Active)
    df_db = spark.createDataFrame([
        (
            69010941952, 
            "H1 SaaS", 
            "saas@h1.co", 
            "active", 
            datetime(2021, 8, 13), 
            datetime(2025, 3, 26), 
            "Support Agent"
        )
    ], get_db_schema())
    mock_get_db.return_value = df_db

    # 3. Execute
    transform_and_reconcile_users(spark)

    # 4. Verify
    args, _ = mock_write.call_args
    df_final = args[0]
    
    # Original user -> Inactive
    old_user = df_final.filter(df_final.user_id == 69010941952).first()
    assert old_user['status'] == 'inactive'
    
    # New user -> Active
    new_user = df_final.filter(df_final.user_id == 999).first()
    assert new_user['status'] == 'active'