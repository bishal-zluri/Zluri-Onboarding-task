from pyspark.sql.functions import col
from pyspark.sql.utils import AnalysisException
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
POSTGRES_JAR = "/Users/bishalpb/onboarding task/jars/postgresql-42.7.8.jar"

DB_URL = "jdbc:postgresql://localhost:5432/postgres"
DB_PROPERTIES = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "driver": "org.postgresql.Driver"
}
TARGET_TABLE = "users"

def get_existing_db_data(spark):
    """
    Reads current Users table. 
    """
    try:
        df = spark.read.jdbc(url=DB_URL, table=TARGET_TABLE, properties=DB_PROPERTIES)
        
        # Cast to Long to match S3 data types
        df = df.withColumn("user_id", col("user_id").cast("long"))
        
        # Safe cast for group_ids if it exists in DB (to ensure consistency)
        if "group_ids" in df.columns:
            df = df.withColumn("group_ids", col("group_ids").cast("string"))

        # Force Materialization
        df.cache()
        count = df.count() 
        
        print(f"[DB] Connection Success. Loaded {count} existing users into memory.")
        return df
        
    except AnalysisException as e:
        if "does not exist" in str(e).lower():
            print(f"[DB] Table '{TARGET_TABLE}' does not exist. Starting Initial Load.")
            return None
        raise e
    except Exception as e:
        print(f"❌ [CRITICAL] Database Read Failed: {e}")
        raise e

def write_to_db_and_show(df, spark):
    """Writes the reconciled dataframe to DB and displays the table."""
    print("\n--- Writing Reconciled Data to DB ---")
    try:
        # 1. Write to DB
        df.write.jdbc(
            url=DB_URL,
            table=TARGET_TABLE,
            mode="overwrite",
            properties=DB_PROPERTIES
        )
        print("✅ [Success] Database updated successfully.")
        
        # 2. Read Back for Verification
        print("\n--- FINAL DATABASE STATE ---")
        df_stored = spark.read.jdbc(DB_URL, TARGET_TABLE, properties=DB_PROPERTIES)
        
        # Order by ID for consistent viewing
        df_stored = df_stored.orderBy("user_id")
        
        # Display in Tabular Form
        df_stored.show(n=50, truncate=False)
        
        print(f"   Total Records: {df_stored.count()}")
        
    except Exception as e:
        print(f"❌ [Error] DB Write Failed: {e}")