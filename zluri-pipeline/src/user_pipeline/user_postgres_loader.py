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

# Table Names
TABLE_USERS = "users"
TABLE_ROLES = "roles"
TABLE_USER_ROLES = "user_roles"

def execute_raw_sql(spark, sql_query):
    conn = None
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        conn = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        stmt = conn.createStatement()
        stmt.execute(sql_query)
    except Exception as e:
        raise e
    finally:
        if conn:
            conn.close()

def init_db(spark):
    print("\n--- [DB] Initializing Schemas ---")
    # A. USERS TABLE
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_USERS} (
            user_id BIGINT PRIMARY KEY, 
            user_name TEXT NOT NULL,
            user_email TEXT NOT NULL,
            status TEXT,
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE
        )
    """)
    
    # B. ROLES TABLE
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_ROLES} (
            role_id TEXT PRIMARY KEY, 
            role_name TEXT,
            role_desc TEXT
        )
    """)

    # C. USER_ROLES
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_USER_ROLES} (
            user_id BIGINT,
            role_id TEXT,
            role_name TEXT,
            PRIMARY KEY (user_id, role_id)
        )
    """)

def get_existing_db_data(spark):
    try:
        init_db(spark)
        df = spark.read.jdbc(url=DB_URL, table=TABLE_USERS, properties=DB_PROPERTIES)
        df = df.withColumn("user_id", col("user_id").cast("long"))
        return df
    except Exception as e:
        print(f"⚠️ Could not read existing users: {e}")
        return None

def load_user_pipeline(spark, df_users, df_roles, df_user_roles):
    init_db(spark) 

    # --- PART A: ROLES (UPSERT) ---
    print(f"\n--- Loading ROLES ({df_roles.count()} rows) ---")
    try:
        df_roles.write.jdbc(DB_URL, "roles_stage", "overwrite", DB_PROPERTIES)
        
        upsert_roles = f"""
        INSERT INTO {TABLE_ROLES} (role_id, role_name, role_desc)
        SELECT role_id, role_name, role_desc FROM roles_stage
        ON CONFLICT (role_id) DO UPDATE SET 
            role_name = EXCLUDED.role_name,
            role_desc = EXCLUDED.role_desc;
        """
        execute_raw_sql(spark, upsert_roles)
        print("✅ Roles Updated.")
    except Exception as e:
        print(f"❌ Failed to load Roles: {e}")
    finally:
        execute_raw_sql(spark, "DROP TABLE IF EXISTS roles_stage")

    # --- PART B: USERS (UPSERT) ---
    print(f"\n--- Loading USERS ({df_users.count()} rows) ---")
    try:
        # Safeguard cast to ensure schema matches
        df_users_safe = df_users.withColumn("user_id", col("user_id").cast("long"))
        
        df_users_safe.select(
            "user_id", "user_name", "user_email", 
            "status", "created_at", "updated_at"
        ).write.jdbc(DB_URL, "users_stage", "overwrite", DB_PROPERTIES)
        
        # Note: Using ::timestamptz to safely cast generic string dates if they passed validation
        upsert_users = f"""
        INSERT INTO {TABLE_USERS} (
            user_id, user_name, user_email,
            status, created_at, updated_at
        )
        SELECT 
            user_id, 
            user_name, 
            user_email, 
            status, 
            created_at::timestamptz, 
            updated_at::timestamptz 
        FROM users_stage
        ON CONFLICT (user_id) DO UPDATE SET 
            user_name  = EXCLUDED.user_name,
            user_email = EXCLUDED.user_email,
            status     = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at::timestamptz;
        """
        execute_raw_sql(spark, upsert_users)
        print("✅ Users Updated.")
    except Exception as e:
        print(f"❌ Failed to load Users: {e}")
        # We assume invalid rows were filtered upstream, so this catches DB connection/schema errors
    finally:
        execute_raw_sql(spark, "DROP TABLE IF EXISTS users_stage")

    # --- PART C: USER_ROLES (SYNC) ---
    print(f"\n--- Loading USER_ROLES ({df_user_roles.count()} mappings) ---")
    try:
        df_ur_safe = df_user_roles.withColumn("user_id", col("user_id").cast("long"))
        df_ur_safe.write.jdbc(DB_URL, "user_roles_stage", "overwrite", DB_PROPERTIES)
        
        delete_sql = f"""
        DELETE FROM {TABLE_USER_ROLES} 
        WHERE user_id IN (SELECT user_id FROM user_roles_stage)
        """
        execute_raw_sql(spark, delete_sql)
        
        insert_sql = f"""
        INSERT INTO {TABLE_USER_ROLES} (user_id, role_id, role_name)
        SELECT user_id, role_id, role_name FROM user_roles_stage
        """
        execute_raw_sql(spark, insert_sql)
        print("✅ User-Role Mappings Synced.")
        
    except Exception as e:
        print(f"❌ Failed to load User-Role mappings: {e}")
    finally:
        execute_raw_sql(spark, "DROP TABLE IF EXISTS user_roles_stage")

    print("\n✅ [PIPELINE COMPLETE] All data processed successfully.")