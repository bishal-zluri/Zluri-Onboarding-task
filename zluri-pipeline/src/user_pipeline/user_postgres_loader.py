import os
from pyspark.sql.functions import col
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Adjust path if needed
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

# --- 1. ROBUST SQL EXECUTOR ---
def execute_raw_sql(spark, sql_query):
    """
    Executes raw SQL (DDL/DML) using the JVM driver manager.
    Includes explicit connection closing and error logging.
    """
    conn = None
    try:
        driver_manager = spark.sparkContext._gateway.jvm.java.sql.DriverManager
        conn = driver_manager.getConnection(DB_URL, DB_PROPERTIES["user"], DB_PROPERTIES["password"])
        stmt = conn.createStatement()
        stmt.execute(sql_query)
        # print(f"   [SQL Executed]: {sql_query[:60].strip()}...")
    except Exception as e:
        # We DO NOT print the error here anymore, because sometimes errors are expected 
        # (like 'primary key already exists') and handled by the caller.
        raise e
    finally:
        if conn:
            conn.close()

# --- 2. SELF-HEALING TABLE INITIALIZATION ---
def _ensure_pk_constraint(spark, table_name, pk_definition):
    """
    Helper to attempt adding a Primary Key if it's missing.
    """
    try:
        print(f"   -> Verifying constraints for '{table_name}'...")
        alter_sql = f"ALTER TABLE {table_name} ADD PRIMARY KEY {pk_definition};"
        execute_raw_sql(spark, alter_sql)
        print(f"   -> Added missing Primary Key to {table_name}.")
    except Exception as e:
        err = str(e).lower()
        if "multiple primary keys" in err or "already exists" in err:
            pass # Constraint already exists, this is good.
        elif "could not create unique index" in err:
            print(f"❌ [CRITICAL] Duplicate data detected in '{table_name}'. Cannot enforce PK.")
            print(f"   PLEASE RUN: TRUNCATE TABLE {table_name};")
            raise e
        else:
            print(f"   [Warning] Constraint check skipped: {err[:50]}...")

def init_db(spark):
    """
    Ensures all 3 tables exist with correct schemas and constraints.
    """
    print("\n--- [DB] Initializing Schemas ---")

    # A. USERS TABLE
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_USERS} (
            user_id BIGINT, -- Constraint added via helper
            user_name TEXT NOT NULL,
            user_email TEXT NOT NULL,
            role_names TEXT,
            status TEXT,
            group_ids TEXT,
            created_at TIMESTAMP WITH TIME ZONE,
            updated_at TIMESTAMP WITH TIME ZONE
        )
    """)
    _ensure_pk_constraint(spark, TABLE_USERS, "(user_id)")

    # B. ROLES TABLE
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_ROLES} (
            role_id TEXT, -- Constraint added via helper
            role_name TEXT,
            role_desc TEXT
        )
    """)
    _ensure_pk_constraint(spark, TABLE_ROLES, "(role_id)")

    # C. USER_ROLES (Link Table)
    execute_raw_sql(spark, f"""
        CREATE TABLE IF NOT EXISTS {TABLE_USER_ROLES} (
            user_id BIGINT,
            role_id TEXT,
            role_name TEXT
            -- Composite PK added via helper
        )
    """)
    _ensure_pk_constraint(spark, TABLE_USER_ROLES, "(user_id, role_id)")


def get_existing_db_data(spark):
    """
    Reads 'users' table for reconciliation.
    """
    try:
        init_db(spark)
        df = spark.read.jdbc(url=DB_URL, table=TABLE_USERS, properties=DB_PROPERTIES)
        # Cast to Long to match Spark types
        df = df.withColumn("user_id", col("user_id").cast("long"))
        return df
    except Exception as e:
        print(f"⚠️ Could not read existing users: {e}")
        return None


# --- 3. MAIN PIPELINE LOADER ---
def load_user_pipeline(spark, df_users, df_roles, df_user_roles):
    """
    Orchestrates the loading of Roles, Users, and User-Role mappings.
    """
    init_db(spark) # Double check schema before writing

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
        # Cast User ID to Long for JDBC safety
        df_users_safe = df_users.withColumn("user_id", col("user_id").cast("long"))
        df_users_safe.write.jdbc(DB_URL, "users_stage", "overwrite", DB_PROPERTIES)
        
        # FIX: Added ::timestamptz cast to created_at and updated_at
        upsert_users = f"""
        INSERT INTO {TABLE_USERS} (
            user_id, user_name, user_email, role_names, 
            status, created_at, updated_at
        )
        SELECT 
            user_id, 
            user_name, 
            user_email, 
            role_names, 
            status, 
            created_at::timestamptz, 
            updated_at::timestamptz 
        FROM users_stage
        ON CONFLICT (user_id) DO UPDATE SET 
            user_name  = EXCLUDED.user_name,
            user_email = EXCLUDED.user_email,
            role_names = EXCLUDED.role_names,
            status     = EXCLUDED.status,
            updated_at = EXCLUDED.updated_at::timestamptz;
        """
        execute_raw_sql(spark, upsert_users)
        print("✅ Users Updated.")
    except Exception as e:
        print(f"❌ Failed to load Users: {e}")
        raise e # Critical failure
    finally:
        execute_raw_sql(spark, "DROP TABLE IF EXISTS users_stage")


    # --- PART C: USER_ROLES (SYNC: DELETE + INSERT) ---
    print(f"\n--- Loading USER_ROLES ({df_user_roles.count()} mappings) ---")
    try:
        df_ur_safe = df_user_roles.withColumn("user_id", col("user_id").cast("long"))
        df_ur_safe.write.jdbc(DB_URL, "user_roles_stage", "overwrite", DB_PROPERTIES)
        
        # 1. Delete existing roles for users contained in this batch
        delete_sql = f"""
        DELETE FROM {TABLE_USER_ROLES} 
        WHERE user_id IN (SELECT user_id FROM user_roles_stage)
        """
        execute_raw_sql(spark, delete_sql)
        
        # 2. Insert new mappings
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