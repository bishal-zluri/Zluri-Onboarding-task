import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pyspark.sql.utils import AnalysisException
from pyspark.sql.types import StructType, StructField, StringType, LongType

# --- BOILERPLATE to import from parent directory ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from user_postgres_loader import get_existing_db_data, write_to_db_and_show, TARGET_TABLE

# --- HELPER TO CREATE EXCEPTIONS ---
def create_analysis_exception(message):
    e = AnalysisException(message, [], [])
    return e

# --- TESTS ---

def test_get_existing_db_success(spark):
    """
    Test that data is read, user_id is cast to Long, and caching occurs.
    """
    # 1. Prepare Expected Data using the REAL spark fixture
    # We use the real spark session to create a valid DataFrame that supports .withColumn, .cache, etc.
    schema = StructType([
        StructField("user_id", StringType(), True), 
        StructField("status", StringType(), True)
    ])
    df_expected = spark.createDataFrame([("101", "active")], schema)
    
    # 2. Create a Mock Spark Session
    # We pretend this MagicMock is the spark session passed to the function.
    mock_spark_session = MagicMock()
    
    # When code calls spark.read.jdbc(...), return our real DataFrame
    mock_spark_session.read.jdbc.return_value = df_expected

    # 3. Execute with the Mock Session
    df_result = get_existing_db_data(mock_spark_session)

    # 4. Assertions
    assert df_result is not None
    
    # Check if casting logic (String "101" -> Long 101) worked
    fields = {f.name: f.dataType for f in df_result.schema.fields}
    assert isinstance(fields["user_id"], LongType)
    
    row = df_result.first()
    assert row["user_id"] == 101


def test_get_existing_db_table_missing(spark):
    """
    Scenario: The table does not exist in Postgres (First run).
    """
    # 1. Create a Mock Spark Session
    mock_spark_session = MagicMock()
    
    # 2. Configure it to raise an error when read.jdbc is called
    error_msg = f"Relation '{TARGET_TABLE}' does not exist"
    mock_spark_session.read.jdbc.side_effect = create_analysis_exception(error_msg)

    # 3. Execute
    result = get_existing_db_data(mock_spark_session)

    # 4. Assertions
    assert result is None


def test_write_to_db_mode_overwrite(spark):
    """
    Test that writing to DB uses mode='overwrite'.
    """
    # 1. Create a dummy dataframe to write (using real spark)
    df_to_write = spark.createDataFrame([(1, "active")], ["user_id", "status"])
    
    # 2. Create a Mock Spark Session (for the verification step at end of function)
    mock_spark_session = MagicMock()
    mock_spark_session.read.jdbc.return_value = df_to_write

    # 3. Patch the DataFrameWriter inside PySpark
    # We mock this because df.write is a property of the DataFrame object itself
    with patch("pyspark.sql.DataFrameWriter.jdbc") as mock_jdbc_write:
        
        # 4. Execute
        write_to_db_and_show(df_to_write, mock_spark_session)

        # 5. Assertions
        assert mock_jdbc_write.called
        
        args, kwargs = mock_jdbc_write.call_args
        assert kwargs.get("mode") == "overwrite"
        assert kwargs.get("table") == TARGET_TABLE