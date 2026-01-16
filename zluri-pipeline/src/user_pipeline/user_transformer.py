from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, concat_ws, collect_set
from s3_reader_users import process_agents_data
from user_postgres_loader import load_user_pipeline, get_existing_db_data, POSTGRES_JAR

def transform_and_reconcile_users(spark):
    print("\n=== STEP 1: Ingesting Data ===")
    df_exploded = process_agents_data(spark)
    
    if df_exploded is None or df_exploded.isEmpty():
        print("⚠️ No new data found. Skipping.")
        return

    df_exploded.cache()

    # --- PREPARE ROLES TABLE (Distinct List) ---
    # Filter out NULL role_ids so we don't try to insert NULL into Primary Key
    df_roles = df_exploded.select("role_id", "role_name", "role_desc").distinct() \
        .filter(col("role_id").isNotNull())

    # --- PREPARE USER_ROLES TABLE (Link Table) ---
    # Filter out mappings where role is missing
    df_user_roles = df_exploded.select("user_id", "role_id", "role_name").distinct() \
        .filter(col("user_id").isNotNull() & col("role_id").isNotNull())

    # --- PREPARE USERS AGGREGATION (For Main Table) ---
    # We aggregate to get unique users, but we won't pass role_names to the final DB table
    df_users_agg = df_exploded.groupBy("user_id", "user_name", "user_email", "created_at", "updated_at") \
        .count().drop("count") # Simple distinct without role aggregation needed for the user table itself

    # --- USER RECONCILIATION LOGIC ---
    print("\n=== STEP 2: Reconciling Users (Status Calculation) ===")
    df_db = get_existing_db_data(spark)
    
    # Define Aliases to be safe
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
        # FIX: Removed 'role_names' from this selection because it is no longer in the DB
        df_old = df_db.select(
            col("user_id").alias("db_id"),
            col("user_name").alias("db_name"),
            col("user_email").alias("db_email"),
            col("status").alias("db_status"),
            col("created_at").alias("db_created")
        ).alias("old")

        # Full Outer Join to find both New and Deleted users
        df_joined = df_new.join(df_old, col("new.user_id") == col("old.db_id"), "full_outer")

        # --- KEY STATUS LOGIC HERE ---
        # FIX: Removed role_names/db_roles from final selection
        df_final_users = df_joined.select(
            # ID: If new ID exists use it, otherwise use DB ID
            coalesce(col("new.user_id"), col("old.db_id")).alias("user_id"),

            # Info: Prefer new info, fallback to old
            coalesce(col("new.user_name"), col("old.db_name")).alias("user_name"),
            coalesce(col("new.user_email"), col("old.db_email")).alias("user_email"),

            # STATUS LOGIC:
            # If 'new.user_id' IS NOT NULL -> User is present in today's sync -> 'active'
            # If 'new.user_id' IS NULL     -> User is missing from sync     -> 'inactive'
            when(col("new.user_id").isNotNull(), "active")
            .otherwise("inactive")
            .alias("status"),

            # Timestamps
            coalesce(col("new.created_at"), col("old.db_created")).alias("created_at"),
            col("new.updated_at") 
        )
        
        # Filter out rows that are completely null (safety check)
        df_final_users = df_final_users.filter(col("user_id").isNotNull())

    # Validation Print
    active_count = df_final_users.filter(col("status") == "active").count()
    inactive_count = df_final_users.filter(col("status") == "inactive").count()
    print(f"   -> Status Summary: Active={active_count}, Inactive={inactive_count}")

    # --- SEND ALL TO LOADER ---
    load_user_pipeline(spark, df_final_users, df_roles, df_user_roles)


if __name__ == "__main__":

    print("--- Launching User Transformer ---")
    
    spark = (
        SparkSession.builder
        .appName("UserTransformer")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000")
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000")
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
        .config("spark.jars", POSTGRES_JAR) 
        .config("spark.driver.extraClassPath", POSTGRES_JAR)
        .getOrCreate()
    )

    transform_and_reconcile_users(spark)