import sys
import os
import pytest
from unittest.mock import patch
from pyspark.sql.types import StructType, StructField, LongType, StringType, ArrayType, BooleanType

# --- BOILERPLATE TO FIND SOURCE FILES ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_reader_groups import process_groups_data, read_s3_data

@patch("s3_reader_groups.read_s3_data")
def test_process_groups_alternative_schema(mock_read_s3, spark):
    """Test the 'users' object list scenario (e.g. from a different API version)"""
    # Schema matching a structure like: [{"id": 1}, {"id": 2}]
    schema_users = StructType([
        StructField("id", LongType(), True),
        StructField("users", ArrayType(StructType([StructField("id", LongType(), True)])), True)
    ])
    
    # Data: Group 1 has users with IDs 101 and 102
    data = [(1, [{"id": 101}, {"id": 102}])]
    mock_read_s3.return_value = spark.createDataFrame(data, schema_users)

    df_result = process_groups_data(spark)
    
    # Assert it correctly extracted [101, 102]
    row = df_result.first()
    assert row["user_ids"] == [101, 102]

@patch("s3_reader_groups.read_s3_data")
def test_process_groups_no_ids(mock_read_s3, spark):
    """Test the 'else' block where no IDs are found"""
    schema_empty = StructType([StructField("id", LongType(), True)])
    mock_read_s3.return_value = spark.createDataFrame([(1,)], schema_empty)

    df_result = process_groups_data(spark)
    
    # Assert it created an empty array, not null
    row = df_result.first()
    assert row["user_ids"] == []


# Patching Spark's internal read API, NOT the function itself
@patch("pyspark.sql.DataFrameReader.json")
def test_read_s3_data_internals(mock_json_read, spark):
    """
    Executes the actual 'read_s3_data' function logic 
    but mocks the final 'spark.read.json' call.
    """
    # 1. Setup mock return
    mock_df = spark.createDataFrame([(1,)], ["id"])
    mock_json_read.return_value = mock_df

    # 2. CALL THE REAL FUNCTION (No patch on read_s3_data)
    result = read_s3_data(spark, "test_folder")

    # 3. Assertions
    assert result is not None
    assert result.count() == 1
    # Verify the code actually tried to read JSON
    assert mock_json_read.called


def get_raw_json_schema():
    """Matches the exact structure of the provided raw JSON."""
    return StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("description", StringType(), True),
        StructField("agent_ids", ArrayType(LongType()), True), # Raw is Array of Longs
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("type", StringType(), True),
        # Add other fields if necessary for strict schema, but these are the core ones used
    ])

@patch("s3_reader_groups.read_s3_data")
def test_process_groups_ingestion(mock_read_s3, spark):
    """
    Test that process_groups_data correctly handles the specific JSON structure provided.
    Key check: 'agent_ids' must be copied to 'user_ids'.
    """
    # 1. Prepare Mock Data (Subset of your raw JSON)
    data = [
        (
            69000630118, 
            "DevOps", 
            "Kicks off devops assignment workflow", 
            [69013042475, 69030996455], 
            "2022-02-09T17:14:12Z", 
            "2022-02-21T15:31:54Z",
            "support_agent_group"
        ),
        (
            69000521031, 
            "H1 IT Support", 
            "Internal H1 IT support team.", 
            [], # Empty list to test edge case
            "2021-08-13T17:17:00Z", 
            "2021-12-13T14:09:44Z",
            "support_agent_group"
        )
    ]
    df_raw = spark.createDataFrame(data, get_raw_json_schema())
    
    # Mock return
    mock_read_s3.return_value = df_raw

    # 2. Execute
    df_result = process_groups_data(spark)

    # 3. Assertions
    assert df_result is not None
    assert df_result.count() == 2
    
    # Check Column Mapping (agent_ids -> user_ids)
    cols = df_result.columns
    assert "user_ids" in cols
    
    # Check Content
    row_devops = df_result.filter(df_result.id == 69000630118).first()
    assert row_devops['user_ids'] == [69013042475, 69030996455]
    
    row_empty = df_result.filter(df_result.id == 69000521031).first()
    assert row_empty['user_ids'] == []