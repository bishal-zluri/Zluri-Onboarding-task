from pyspark.sql import SparkSession
import os

# Function to create a Spark session with PostgreSQL JDBC driver for Users Pipeline
def create_spark_session(app_name="Zluri_User_Pipeline"):
    """
    Creates a Spark session with PostgreSQL JDBC driver config.
    """

    postgres_jar = "/Users/bishalpb/onboarding task/jars/postgresql-42.7.8.jar"
    
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars", postgres_jar) \
        .config("spark.driver.extraClassPath", postgres_jar) \
        .getOrCreate()

