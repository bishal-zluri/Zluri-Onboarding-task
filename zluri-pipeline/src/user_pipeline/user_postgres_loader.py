from pyspark.sql.functions import col
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Adjust path if needed or keep using the variable from your environment
POSTGRES_JAR = "/Users/bishalpb/onboarding task/jars/postgresql-42.7.8.jar"

DB_URL = "jdbc:postgresql://localhost:5432/postgres"
DB_PROPERTIES = {
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "password"),
    "driver": "org.postgresql.Driver"
}

TARGET_TABLE = "users"
TEMP_TABLE = "users_stage"

def execute_raw_sql(spark, sql_query):
    """Executes raw SQL (DDL/DML) using the JVM driver."""
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        con = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        stmt = con.createStatement()
        stmt.execute(sql_query)
        con.close()
    except Exception as e:
        print(f"❌ [SQL Error]: {e}")
        raise e

def init_users_table(spark):
    """
    Creates table if missing, and attempts to FORCE Primary Key if it exists but is missing the constraint.
    """
    # 1. Basic Creation
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        user_id BIGINT PRIMARY KEY,
        user_name TEXT NOT NULL,
        user_email TEXT NOT NULL,
        role_names TEXT,
        group_ids TEXT,
        status TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE,
        updated_at TIMESTAMP WITH TIME ZONE
    );
    """
    execute_raw_sql(spark, ddl)

    # 2. Self-Healing: Force Primary Key if missing
    # (Handles cases where table exists but was created without PK)
    try:
        print(f"--- [DB] Verifying Constraints for '{TARGET_TABLE}' ---")
        alter_sql = f"ALTER TABLE {TARGET_TABLE} ADD PRIMARY KEY (user_id);"
        execute_raw_sql(spark, alter_sql)
        print("   -> Added missing Primary Key constraint.")
    except Exception as e:
        err = str(e).lower()
        if "multiple primary keys" in err or "already exists" in err:
            print("   -> Primary Key already exists. (Good)")
        elif "could not create unique index" in err:
            print("❌ [CRITICAL] Cannot add Primary Key: Duplicate IDs exist in DB.")
            print("   PLEASE RUN: DROP TABLE users;")
            raise e
        else:
            # Other errors might be benign (like connection issues handled elsewhere)
            print(f"   -> Constraint check skipped: {err[:50]}...")

def get_existing_db_data(spark):
    """
    Reads current Users table for the Transformer's reconciliation step.
    """
    try:
        # Ensure table exists first
        init_users_table(spark)
        
        df = spark.read.jdbc(url=DB_URL, table=TARGET_TABLE, properties=DB_PROPERTIES)
        
        # Cast User ID to Long (S3 usually provides Long, DB provides BigInt)
        df = df.withColumn("user_id", col("user_id").cast("long"))
        
        print(f"[DB] Loaded {df.count()} existing users.")
        return df
    except Exception as e:
        print(f"⚠️ Could not read table '{TARGET_TABLE}'. Returning None. Error: {e}")
        return None

def write_to_db_and_show(df, spark):
    """
    Performs UPSERT based on the User Transformer's output.
    """
    print("\n--- Writing Reconciled Users to DB (UPSERT) ---")
    
    # 1. Cast for Safety (Match DB Types)
    df_write = df.withColumn("user_id", col("user_id").cast("long"))
    
    # Ensure string columns are strictly strings (handle potential nulls)
    # The transformer handles most of this, but this is a safety net for JDBC
    if "group_ids" in df.columns:
        df_write = df_write.withColumn("group_ids", col("group_ids").cast("string"))

    try:
        # 2. Write to Staging Table
        df_write.write.jdbc(
            url=DB_URL, 
            table=TEMP_TABLE, 
            mode="overwrite", 
            properties=DB_PROPERTIES
        )
        
        # 3. UPSERT SQL
        # We update ALL mutable fields if the ID exists.
        # We do NOT update 'created_at' on conflict (creation time shouldn't change).
        upsert_sql = f"""
        INSERT INTO {TARGET_TABLE} (
            user_id, user_name, user_email, role_names, 
            group_ids, status, created_at, updated_at
        )
        SELECT 
            user_id, user_name, user_email, role_names, 
            group_ids, status, created_at, updated_at
        FROM {TEMP_TABLE}
        ON CONFLICT (user_id) 
        DO UPDATE SET 
            user_name   = EXCLUDED.user_name,
            user_email  = EXCLUDED.user_email,
            role_names  = EXCLUDED.role_names,
            group_ids   = EXCLUDED.group_ids,
            status      = EXCLUDED.status,
            updated_at  = EXCLUDED.updated_at;
        """
        
        execute_raw_sql(spark, upsert_sql)
        print("✅ [Success] Users UPSERT completed.")
        
        # 4. Clean up
        execute_raw_sql(spark, f"DROP TABLE {TEMP_TABLE}")

        # 5. Verify Output
        print("\n--- FINAL USERS STATE (Top 10) ---")
        df_stored = spark.read.jdbc(DB_URL, TARGET_TABLE, properties=DB_PROPERTIES)
        df_stored.orderBy("user_id").show(10, truncate=False)
        
    except Exception as e:
        print(f"❌ [Error] DB Write Failed: {e}")