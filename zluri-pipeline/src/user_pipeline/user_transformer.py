from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, to_timestamp, expr
from pyspark.sql.types import LongType
from s3_reader_users import process_agents_data
from user_postgres_loader import load_user_pipeline, get_existing_db_data, POSTGRES_JAR

def transform_and_reconcile_users(spark):
    print("\n=== STEP 1: Ingesting Data ===")
    df_exploded = process_agents_data(spark)
    
    if df_exploded is None or df_exploded.isEmpty():
        print("⚠️ No new data found. Skipping.")
        return

    # --- VALIDATION & CLEANING STEP ---
    # 1. Cast user_id to Long. Invalid strings (e.g. "abc") become null automatically.
    # 2. Filter out null user_ids.
    # 3. Ensure mandatory fields (name, email) are present.
    # 4. Use try_cast for timestamps to handle malformed dates like '2022-05-' safely (returns NULL instead of crashing).
    
    df_clean = df_exploded.withColumn("valid_user_id", col("user_id").cast(LongType())) \
        .filter(col("valid_user_id").isNotNull()) \
        .filter(col("user_name").isNotNull() & (col("user_name") != "")) \
        .filter(col("user_email").isNotNull() & (col("user_email") != "")) \
        .withColumn("user_id", col("valid_user_id")) \
        .withColumn("created_at", expr("try_cast(created_at as timestamp)")) \
        .withColumn("updated_at", expr("try_cast(updated_at as timestamp)")) \
        .drop("valid_user_id")

    # Logging dropped count
    original_count = df_exploded.count()
    clean_count = df_clean.count()
    dropped_count = original_count - clean_count
    
    if dropped_count > 0:
        print(f"⚠️ Dropped {dropped_count} invalid records (null/alphanumeric IDs or missing mandatory fields).")
        print("   Only valid numeric user_ids with proper names/emails will be loaded.")

    df_clean.cache()

    # --- PREPARE ROLES TABLE (Distinct List) ---
    # Validation: Exclude Role Names that are just numbers (e.g. "12334")
    df_roles = df_clean.select("role_id", "role_name", "role_desc").distinct() \
        .filter(col("role_id").isNotNull()) \
        .filter(~col("role_name").rlike("^\d+$"))

    # --- PREPARE USER_ROLES TABLE (Link Table) ---
    df_user_roles = df_clean.select("user_id", "role_id", "role_name").distinct() \
        .filter(col("user_id").isNotNull() & col("role_id").isNotNull()) \
        .filter(~col("role_name").rlike("^\d+$"))

    # --- PREPARE USERS AGGREGATION (For Main Table) ---
    df_users_agg = df_clean.groupBy("user_id", "user_name", "user_email", "created_at", "updated_at") \
        .count().drop("count") 

    # --- USER RECONCILIATION LOGIC ---
    print("\n=== STEP 2: Reconciling Users (Status Calculation) ===")
    df_db = get_existing_db_data(spark)
    
    df_new = df_users_agg.alias("new")
    
    if df_db is None:
        # Initial Load: Everyone is active
        print("   -> Initial Load. Setting all users to 'active'.")
        df_final_users = df_new.select(
            col("new.*"), 
            lit("active").alias("status")
        )
    else:
        # Merge Logic
        df_old = df_db.select(
            col("user_id").alias("db_id"),
            col("user_name").alias("db_name"),
            col("user_email").alias("db_email"),
            col("status").alias("db_status"),
            col("created_at").alias("db_created"),
            col("updated_at").alias("db_updated")
        ).alias("old")

        # Full Outer Join
        df_joined = df_new.join(df_old, col("new.user_id") == col("old.db_id"), "full_outer")

        # --- KEY STATUS LOGIC HERE ---
        df_final_users = df_joined.select(
            coalesce(col("new.user_id"), col("old.db_id")).alias("user_id"),
            coalesce(col("new.user_name"), col("old.db_name")).alias("user_name"),
            coalesce(col("new.user_email"), col("old.db_email")).alias("user_email"),
            when(col("new.user_id").isNotNull(), "active").otherwise("inactive").alias("status"),
            coalesce(col("new.created_at"), col("old.db_created")).alias("created_at"),
            coalesce(col("new.updated_at"), col("old.db_updated")).alias("updated_at") 
        )
        
    # Final filter for safety
    df_final_users = df_final_users.filter(col("user_id").isNotNull())

    # Validation Print
    active_count = df_final_users.filter(col("status") == "active").count()
    inactive_count = df_final_users.filter(col("status") == "inactive").count()
    print(f"   -> Status Summary: Active={active_count}, Inactive={inactive_count}")

    # --- SEND ALL TO LOADER ---
    load_user_pipeline(spark, df_final_users, df_roles, df_user_roles)


if __name__ == "__main__":
    print("--- Launching User Transformer (Local) ---")
    
    spark = (
        SparkSession.builder
        .appName("UserTransformer")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1") 
        .config("spark.jars", POSTGRES_JAR) 
        .config("spark.driver.extraClassPath", POSTGRES_JAR)
        .getOrCreate()
    )

    transform_and_reconcile_users(spark)