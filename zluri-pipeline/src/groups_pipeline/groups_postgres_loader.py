from pyspark.sql.functions import col, concat_ws
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

TABLE_GROUPS = "groups"
TABLE_USERS = "users"  # Needed for cross-referencing in transformer
TEMP_GROUPS = "groups_stage"

def execute_raw_sql(spark, sql_query):
    """Executes raw SQL using the JVM driver."""
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        con = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        stmt = con.createStatement()
        stmt.execute(sql_query)
        con.close()
    except Exception as e:
        print(f"❌ [SQL Error]: {e}")
        raise e

def init_groups_table(spark):
    """
    Creates table if missing, and attempts to FORCE Primary Key if it exists but is missing the constraint.
    """
    # 1. Basic Creation
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_GROUPS} (
        group_id BIGINT PRIMARY KEY,
        group_name TEXT NOT NULL,
        description TEXT,
        user_ids TEXT, 
        status TEXT,
        created_at TIMESTAMP WITH TIME ZONE,
        updated_at TIMESTAMP WITH TIME ZONE
    );
    """
    execute_raw_sql(spark, ddl)

    # 2. Self-Healing: Force Primary Key
    try:
        print(f"--- [DB] Verifying Constraints for '{TABLE_GROUPS}' ---")
        alter_sql = f"ALTER TABLE {TABLE_GROUPS} ADD PRIMARY KEY (group_id);"
        execute_raw_sql(spark, alter_sql)
        print("   -> Added missing Primary Key constraint.")
    except Exception as e:
        err = str(e).lower()
        if "multiple primary keys" in err or "already exists" in err:
            print("   -> Primary Key already exists. (Good)")
        elif "could not create unique index" in err:
            print("❌ [CRITICAL] Cannot add Primary Key: Duplicate IDs exist in DB.")
            print("   PLEASE RUN: DROP TABLE groups;")
            raise e
        else:
            print(f"   -> Constraint check skipped: {err[:50]}...")

def get_db_table(spark, table_name):
    """
    Reads a table from Postgres. 
    Used by Transformer to read both 'users' (for status) and 'groups'.
    """
    try:
        # Helper: Ensure schema exists if we are asking for groups
        if table_name == TABLE_GROUPS:
            init_groups_table(spark)
            
        df = spark.read.jdbc(url=DB_URL, table=table_name, properties=DB_PROPERTIES)
        
        # Standardize IDs to Long for Spark operations
        if "user_id" in df.columns:
            df = df.withColumn("user_id", col("user_id").cast("long"))
        if "group_id" in df.columns:
            df = df.withColumn("group_id", col("group_id").cast("long"))
            
        return df
    except Exception as e:
        print(f"⚠️ Could not read table '{table_name}' (It might not exist yet).")
        return None

def write_to_db(df, spark):
    """
    Writes reconciled groups using UPSERT.
    """
    print("\n--- Writing Groups to DB (UPSERT) ---")
    
    # 1. Convert Arrays to Strings
    # Your transformer leaves 'user_ids' as an Array<Long>.
    # Postgres JDBC requires 'String' for the TEXT column.
    df_write = df
    if "user_ids" in [f.name for f in df.schema.fields]:
        dtype = dict(df.dtypes)["user_ids"]
        # Check if it is an array type (e.g., 'array<bigint>')
        if "array" in dtype:
            print("-> Converting 'user_ids' Array to String for DB storage...")
            df_write = df.withColumn("user_ids", concat_ws(",", col("user_ids")))

    try:
        # 2. Write to Staging
        df_write.write.jdbc(
            url=DB_URL,
            table=TEMP_GROUPS,
            mode="overwrite",
            properties=DB_PROPERTIES
        )

        # 3. UPSERT SQL
        upsert_sql = f"""
        INSERT INTO {TABLE_GROUPS} (
            group_id, group_name, description, 
            user_ids, status, created_at, updated_at
        )
        SELECT 
            group_id, group_name, description, 
            user_ids, status, created_at, updated_at
        FROM {TEMP_GROUPS}
        ON CONFLICT (group_id) 
        DO UPDATE SET 
            group_name  = EXCLUDED.group_name,
            description = EXCLUDED.description,
            user_ids    = EXCLUDED.user_ids,
            status      = EXCLUDED.status,
            updated_at  = EXCLUDED.updated_at;
        """
        
        execute_raw_sql(spark, upsert_sql)
        print("✅ [Success] Groups UPSERT completed.")
        
        # 4. Cleanup
        execute_raw_sql(spark, f"DROP TABLE {TEMP_GROUPS}")
        
        # 5. Verify
        print("\n--- FINAL GROUPS STATE (Top 10) ---")
        df_stored = spark.read.jdbc(DB_URL, TABLE_GROUPS, properties=DB_PROPERTIES)
        df_stored.orderBy("group_id").show(10, truncate=False)
        
    except Exception as e:
        print(f"❌ [Error] DB Write Failed: {e}")