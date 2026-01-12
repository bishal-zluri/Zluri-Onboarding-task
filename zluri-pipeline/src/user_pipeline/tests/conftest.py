# tests/conftest.py
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """
    Creates a local Spark session for testing.
    'local[1]' means it runs on one core in the local machine.
    """
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("UnitTests") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    yield spark
    spark.stop()

