from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, concat_ws
from s3_reader_users import process_agents_data
from user_postgres_loader import write_to_db_and_show, get_existing_db_data, POSTGRES_JAR

def transform_and_reconcile_users(spark):
    # --- STEP 1: Ingest New Data ---
    print("\n=== STEP 1: Ingesting Daily Data ===")
    df_raw = process_agents_data(spark)

    if df_raw is None or df_raw.isEmpty():
        print("⚠️ No new data found in S3. Exiting.")
        return

    # Select & Cast
    # Added: group_ids (flattened to string "1,2,3")
    df_new = df_raw.select(
        col("user_id").cast("long"),
        col("user_name"),
        col("user_email"),
        col("user_created_at").alias("created_at").cast("timestamp"),
        col("user_updated_at").alias("updated_at").cast("timestamp"),
        concat_ws(", ", col("role_names")).alias("role_names"),
        concat_ws(",", col("group_ids")).alias("group_ids") 
    )
    
    # Cache new data
    df_new.cache()
    print(f"   New incoming users: {df_new.count()}")

    # --- STEP 2: Load Existing DB Data ---
    print("\n=== STEP 2: Loading DB History ===")
    df_old = get_existing_db_data(spark)

    # --- INITIAL LOAD SCENARIO ---
    if df_old is None:
        print("   -> First Run Detected. Marking all new users as 'active'.")
        
        df_clean_initial = df_new.filter(
            col("user_id").isNotNull() & 
            col("user_name").isNotNull() & 
            col("user_email").isNotNull()
        ).withColumn("status", lit("active"))
        
        # Log dropped rows
        dropped = df_new.count() - df_clean_initial.count()
        if dropped > 0:
            print(f"⚠️ Warning: Dropped {dropped} rows with NULL ID/Name/Email.")

        write_to_db_and_show(df_clean_initial, spark)
        return

    # --- HANDLE SCHEMA EVOLUTION ---
    # If the DB exists but doesn't have 'group_ids' yet, add it as null
    if "group_ids" not in df_old.columns:
        print("   -> 'group_ids' column missing in DB. Adding placeholder.")
        df_old = df_old.withColumn("group_ids", lit(None).cast("string"))

    # --- STEP 3: RECONCILIATION ---
    print("\n=== STEP 3: Reconciling (Preserving Inactive Users) ===")
    
    # Rename DB columns
    df_old_renamed = df_old.select(
        col("user_id").alias("db_id"),
        col("user_name").alias("db_name"),
        col("user_email").alias("db_email"),
        col("status").alias("db_status"),
        col("created_at").alias("db_created"),
        col("updated_at").alias("db_updated"),
        col("role_names").alias("db_roles"),
        col("group_ids").alias("db_groups") 
    )

    # Full Outer Join
    df_joined = df_new.join(
        df_old_renamed,
        df_new.user_id == df_old_renamed.db_id,
        how="full_outer"
    )

    # Apply Logic
    df_final = df_joined.select(
        # ID: Prefer New, fallback to Old
        coalesce(col("user_id"), col("db_id")).alias("user_id"),

        # INFO UPDATES (Upsert)
        coalesce(col("user_name"), col("db_name")).alias("user_name"),
        coalesce(col("user_email"), col("db_email")).alias("user_email"),
        coalesce(col("role_names"), col("db_roles")).alias("role_names"),
        coalesce(col("group_ids"), col("db_groups")).alias("group_ids"),

        # STATUS LOGIC
        when(col("user_id").isNotNull(), "active")
        .otherwise("inactive")
        .alias("status"),

        # TIMESTAMPS
        coalesce(col("created_at"), col("db_created")).alias("created_at"),
        coalesce(col("updated_at"), col("db_updated")).alias("updated_at")
    )

    # --- STEP 4: VALIDATION ---
    print("\n=== STEP 4: Enforcing NOT NULL Constraints ===")
    df_clean_final = df_final.filter(
        col("user_id").isNotNull() & 
        col("user_name").isNotNull() & 
        col("user_email").isNotNull()
    )

    # Summary
    df_clean_final.cache()
    
    total = df_clean_final.count()
    active = df_clean_final.filter(col("status") == "active").count()
    inactive = df_clean_final.filter(col("status") == "inactive").count()
    
    print(f"   RECONCILIATION SUMMARY:")
    print(f"   Total Valid Users : {total}")
    print(f"   Active            : {active}")
    print(f"   Inactive          : {inactive}")

    write_to_db_and_show(df_clean_final, spark)
    
    # Cleanup
    df_clean_final.unpersist()
    df_new.unpersist()
    df_old.unpersist()

if __name__ == "__main__":
    print(f"--- Launching Spark with JAR: {POSTGRES_JAR} ---")
    
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