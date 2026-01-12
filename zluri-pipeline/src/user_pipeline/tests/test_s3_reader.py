# tests/test_ingestion.py
import sys
import os
from unittest.mock import patch
from pyspark.sql.types import StructType, StructField, LongType, StringType, ArrayType, BooleanType

# --- BOILERPLATE to import from parent directory ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_reader_users import process_agents_data

def get_agent_details_schema():
    """Matches the exact complex structure of your raw JSON."""
    return StructType([
        StructField("id", LongType(), True),
        StructField("org_agent_id", StringType(), True),
        StructField("role_ids", ArrayType(LongType()), True), # Raw JSON has integers in array
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("contact", StructType([
            StructField("active", BooleanType(), True),
            StructField("email", StringType(), True),
            StructField("name", StringType(), True),
            StructField("mobile", StringType(), True),
            StructField("created_at", StringType(), True)
        ]), True)
    ])

@patch("s3_reader_users.read_s3_data")
def test_process_agents_ingestion(mock_read_s3, spark):
    """
    Test that process_agents_data correctly joins Agents + Details + Roles
    based on the specific JSON structure provided.
    """
    # 1. Mock Agents Index (Simple ID list)
    schema_idx = StructType([StructField("id", LongType(), True)])
    df_agents = spark.createDataFrame([(69010941952,), (69010941953,)], schema_idx)

    # 2. Mock Agent Details (Complex JSON)
    df_details = spark.createDataFrame([
        (
            69010941952, 
            "346268513062377277", 
            [69000378065], 
            "2021-08-13T11:50:18Z", 
            "2025-03-26T18:19:00Z",
            (True, "saas@h1.co", "H1 SaaS", "3473201776", "2021-08-13T11:50:18Z")
        ),
        (
            69010941953, 
            "999", 
            [69000378066], 
            "2022-01-01T10:00:00Z", 
            "2022-01-01T10:00:00Z",
            (True, "bob@h1.co", "Bob Builder", "12345", "2022-01-01T10:00:00Z")
        )
    ], get_agent_details_schema())

    # 3. Mock Roles
    schema_roles = StructType([
        StructField("id", LongType(), True),
        StructField("name", StringType(), True),
        StructField("description", StringType(), True)
    ])
    df_roles = spark.createDataFrame([
        (69000378065, "Support Agent", "Handles tickets"),
        (69000378066, "Admin", "Super User")
    ], schema_roles)

    # Setup the mock to return specific DFs based on folder name
    def side_effect(spark, folder_name):
        if folder_name == "agents": return df_agents
        if folder_name == "agent_details": return df_details
        if folder_name == "roles": return df_roles
        return None
    mock_read_s3.side_effect = side_effect

    # 4. Execute
    df_result = process_agents_data(spark)

    # 5. Assertions
    assert df_result is not None
    assert df_result.count() == 2
    
    # Check User 1 (H1 SaaS)
    user1 = df_result.filter(df_result.user_id == 69010941952).first()
    assert user1['user_name'] == "H1 SaaS"
    assert user1['user_email'] == "saas@h1.co"
    # Verify Role Mapping: ID 69000378065 -> Name "Support Agent"
    assert "Support Agent" in user1['role_names']