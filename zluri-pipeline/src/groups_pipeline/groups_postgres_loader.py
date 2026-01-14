import os
from pyspark.sql.functions import col, concat_ws
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
TABLE_GROUP_MEMBERS = "group_members" 
TABLE_USERS = "users"
TEMP_GROUPS = "groups_stage"
TEMP_MEMBERS = "group_members_stage"

def execute_raw_sql(spark, sql_query):
    """Executes raw SQL using the JVM driver."""
    conn = None
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        conn = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        stmt = conn.createStatement()
        stmt.execute(sql_query)
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"❌ [SQL Error]: {e}")
            raise e
    finally:
        if conn: conn.close()

def _ensure_pk(spark, table, pk_col):
    try:
        execute_raw_sql(spark, f"ALTER TABLE {table} ADD PRIMARY KEY {pk_col};")
        print(f"   -> Added Primary Key to {table}.")
    except Exception:
        pass 

def init_db(spark):
    """
    Initializes Groups and GroupMembers tables.
    """
    print("\n--- [DB] Initializing Group Schemas ---")

    # 1. GROUPS TABLE
    execute_raw_sql(spark, f"""
    CREATE TABLE IF NOT EXISTS {TABLE_GROUPS} (
        group_id BIGINT,
        group_name TEXT NOT NULL,
        description TEXT,
        user_ids TEXT, 
        parent_group_id BIGINT,
        status TEXT,
        created_at TIMESTAMP WITH TIME ZONE,
        updated_at TIMESTAMP WITH TIME ZONE
    );
    """)
    _ensure_pk(spark, TABLE_GROUPS, "(group_id)")
    
    # Schema Evolution
    try:
        execute_raw_sql(spark, f"ALTER TABLE {TABLE_GROUPS} ADD COLUMN parent_group_id BIGINT")
    except Exception:
        pass

    # 2. GROUP_MEMBERS TABLE
    execute_raw_sql(spark, f"""
    CREATE TABLE IF NOT EXISTS {TABLE_GROUP_MEMBERS} (
        group_id BIGINT,
        user_id BIGINT,
        user_status TEXT
    );
    """)
    _ensure_pk(spark, TABLE_GROUP_MEMBERS, "(group_id, user_id)")


def get_db_table(spark, table_name):
    """
    Reads a table from Postgres. 
    """
    try:
        init_db(spark) 
        df = spark.read.jdbc(url=DB_URL, table=table_name, properties=DB_PROPERTIES)
        
        if "user_id" in df.columns:
            df = df.withColumn("user_id", col("user_id").cast("long"))
        if "group_id" in df.columns:
            df = df.withColumn("group_id", col("group_id").cast("long"))
        if "parent_group_id" in df.columns:
            df = df.withColumn("parent_group_id", col("parent_group_id").cast("long"))
            
        return df
    except Exception as e:
        print(f"⚠️ Could not read table '{table_name}': {e}")
        return None

def write_to_db(df_groups, df_members, spark):
    """
    Writes Groups (Upsert) and GroupMembers (Sync).
    """
    init_db(spark)

    # --- PART 1: GROUPS (UPSERT) ---
    print("\n--- Writing GROUPS (UPSERT) ---")
    
    # Convert Array to String for display column
    df_write_groups = df_groups
    if "user_ids" in [f.name for f in df_groups.schema.fields]:
        if "array" in dict(df_groups.dtypes)["user_ids"]:
            df_write_groups = df_groups.withColumn("user_ids", concat_ws(",", col("user_ids")))

    try:
        df_write_groups.write.jdbc(DB_URL, TEMP_GROUPS, "overwrite", DB_PROPERTIES)

        # Upsert with parent_group_id
        upsert_sql = f"""
        INSERT INTO {TABLE_GROUPS} (
            group_id, group_name, description, 
            user_ids, parent_group_id, status, created_at, updated_at
        )
        SELECT 
            group_id, group_name, description, 
            user_ids, parent_group_id, status, created_at, updated_at
        FROM {TEMP_GROUPS}
        ON CONFLICT (group_id) 
        DO UPDATE SET 
            group_name  = EXCLUDED.group_name,
            description = EXCLUDED.description,
            user_ids    = EXCLUDED.user_ids,
            parent_group_id = EXCLUDED.parent_group_id,
            status      = EXCLUDED.status,
            updated_at  = EXCLUDED.updated_at;
        """
        execute_raw_sql(spark, upsert_sql)
        print("✅ Groups Upserted.")
    except Exception as e:
        print(f"❌ Groups Write Failed: {e}")
    finally:
        execute_raw_sql(spark, f"DROP TABLE IF EXISTS {TEMP_GROUPS}")

    # --- PART 2: GROUP MEMBERS (SYNC) ---
    if df_members and not df_members.isEmpty():
        print(f"\n--- Writing GROUP MEMBERS ({df_members.count()} rows) ---")
        try:
            df_members.write.jdbc(DB_URL, TEMP_MEMBERS, "overwrite", DB_PROPERTIES)

            delete_sql = f"""
            DELETE FROM {TABLE_GROUP_MEMBERS} 
            WHERE group_id IN (SELECT group_id FROM {TEMP_MEMBERS})
            """
            execute_raw_sql(spark, delete_sql)

            insert_sql = f"""
            INSERT INTO {TABLE_GROUP_MEMBERS} (group_id, user_id, user_status)
            SELECT group_id, user_id, user_status FROM {TEMP_MEMBERS}
            """
            execute_raw_sql(spark, insert_sql)
            print("✅ Group Members Synced.")
        except Exception as e:
            print(f"❌ Group Members Write Failed: {e}")
        finally:
            execute_raw_sql(spark, f"DROP TABLE IF EXISTS {TEMP_MEMBERS}")