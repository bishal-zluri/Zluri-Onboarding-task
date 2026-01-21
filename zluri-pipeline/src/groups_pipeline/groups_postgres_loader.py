import os
from pyspark.sql.functions import col
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

def init_db(spark):
    """
    Initializes Groups and GroupMembers tables with inline Primary Keys.
    """
    print("\n--- [DB] Initializing Group Schemas ---")

    # 1. GROUPS TABLE
    execute_raw_sql(spark, f"""
    CREATE TABLE IF NOT EXISTS {TABLE_GROUPS} (
        group_id BIGINT PRIMARY KEY,
        group_name TEXT NOT NULL,
        description TEXT,
        parent_group_id BIGINT,
        status TEXT,
        created_at TIMESTAMP WITH TIME ZONE,
        updated_at TIMESTAMP WITH TIME ZONE
    );
    """)
    
    # Schema Evolution Check
    try:
        execute_raw_sql(spark, f"ALTER TABLE {TABLE_GROUPS} ADD COLUMN parent_group_id BIGINT")
    except Exception:
        pass

    # 2. GROUP_MEMBERS TABLE
    execute_raw_sql(spark, f"""
    CREATE TABLE IF NOT EXISTS {TABLE_GROUP_MEMBERS} (
        group_id BIGINT,
        user_id BIGINT,
        user_status TEXT,
        PRIMARY KEY (group_id, user_id)
    );
    """)

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
    
    # Drop user_ids as we normalize it into group_members
    df_write_groups = df_groups.drop("user_ids")

    try:
        # Cast group_id safely again before write just in case
        df_write_groups = df_write_groups.withColumn("group_id", col("group_id").cast("long"))
        
        df_write_groups.write.jdbc(DB_URL, TEMP_GROUPS, "overwrite", DB_PROPERTIES)

        # Upsert Logic with Safe Date Casting
        upsert_sql = f"""
        INSERT INTO {TABLE_GROUPS} (
            group_id, group_name, description, 
            parent_group_id, status, created_at, updated_at
        )
        SELECT 
            group_id, 
            group_name, 
            description, 
            parent_group_id, 
            status, 
            created_at::timestamptz, 
            updated_at::timestamptz
        FROM {TEMP_GROUPS}
        ON CONFLICT (group_id) 
        DO UPDATE SET 
            group_name  = EXCLUDED.group_name,
            description = EXCLUDED.description,
            parent_group_id = EXCLUDED.parent_group_id,
            status      = EXCLUDED.status,
            updated_at  = EXCLUDED.updated_at::timestamptz;
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
            df_members = df_members.withColumn("group_id", col("group_id").cast("long")) \
                                   .withColumn("user_id", col("user_id").cast("long"))
            
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