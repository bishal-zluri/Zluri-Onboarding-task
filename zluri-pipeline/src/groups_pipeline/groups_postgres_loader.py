from pyspark.sql.functions import col, concat_ws
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
POSTGRES_JAR = "/Users/bishalpb/onboarding task/jars/postgresql-42.7.8.jar"
DB_URL = "jdbc:postgresql://localhost:5432/postgres"

# DB Credentials
DB_PROPERTIES = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "driver": "org.postgresql.Driver"
}

# Tables
TABLE_USERS = "users"
TABLE_GROUPS = "groups"

def get_db_table(spark, table_name):
    """
    Reads a table from Postgres. 
    Includes caching to prevent 'Truncate before Read' race conditions.
    """
    try:
        df = spark.read.jdbc(url=DB_URL, table=table_name, properties=DB_PROPERTIES)
        df.cache()
        count = df.count()
        print(f"[DB] Successfully loaded '{table_name}': {count} rows.")
        return df
    except Exception as e:
        # It's normal for the table to not exist on the first run
        print(f"⚠️ Could not read table '{table_name}' (It might not exist yet).")
        return None

def write_to_db(df, spark):
    """Writes reconciled groups to DB and shows preview."""
    print("\n--- Writing Groups to DB ---")
    
    # FIX: Convert Arrays to Strings before writing to Postgres
    # Postgres JDBC often fails with Spark Arrays unless using advanced dialects.
    # We convert ['101', '102'] -> "101,102"
    df_write = df
    if "user_ids" in [f.name for f in df.schema.fields]:
        dtype = dict(df.dtypes)["user_ids"]
        if "array" in dtype:
            print("-> Converting 'user_ids' array to comma-separated string for DB storage...")
            df_write = df.withColumn("user_ids", concat_ws(",", col("user_ids")))

    try:
        df_write.write.jdbc(
            url=DB_URL,
            table=TABLE_GROUPS,
            mode="overwrite",
            properties=DB_PROPERTIES
        )
        print("✅ [Success] Groups table updated.")
        
        # Verify
        print("\n--- Final Groups Table Content ---")
        df_stored = spark.read.jdbc(DB_URL, TABLE_GROUPS, properties=DB_PROPERTIES)
        df_stored.orderBy("group_id").show(20, truncate=False)
        print(f"Total Groups: {df_stored.count()}")
        
    except Exception as e:
        print(f"❌ [Error] DB Write Failed: {e}")
