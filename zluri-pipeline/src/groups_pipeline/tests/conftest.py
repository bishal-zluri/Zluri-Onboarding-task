import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """
    Creates a local Spark session for testing.
    Scope='session' means it runs once per test suite, not per test function.
    """
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("UnitTests") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    
    yield spark
    spark.stop()