from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, explode, max as spark_max, split
from pyspark.sql.types import StructType, StructField, LongType, StringType

# different file imports
from s3_reader_groups import process_groups_data
from groups_postgres_loader import get_db_table, write_to_db, TABLE_GROUPS, TABLE_USERS, POSTGRES_JAR

def transform_and_reconcile_groups(spark):
    # ---------------------------------------------------------
    # STEP 1: Ingest Groups from S3
    # ---------------------------------------------------------
    df_raw_groups = process_groups_data(spark)
    if df_raw_groups is None: 
        print("No group data found in S3.")
        return

    # Clean & Select relevant columns
    df_groups_clean = df_raw_groups.select(
        col("id").cast("long").alias("group_id"),
        col("name").alias("group_name"),
        col("description"),
        col("created_at").cast("timestamp"),
        col("updated_at").cast("timestamp"),
        col("agent_ids").alias("user_ids")
    )

    # --- VALIDATION ADDED HERE ---
    # Filter out rows where group_id or group_name is Null
    initial_count = df_groups_clean.count()
    
    df_groups_clean = df_groups_clean.filter(
        col("group_id").isNotNull() & col("group_name").isNotNull()
    )
    
    valid_count = df_groups_clean.count()
    dropped_count = initial_count - valid_count
    
    print(f"[INGEST] Total S3 Rows: {initial_count}")
    print(f"[INGEST] Dropped Rows (Null ID/Name): {dropped_count}")
    print(f"[INGEST] Valid S3 Groups: {valid_count}")
    # -----------------------------

    # ---------------------------------------------------------
    # STEP 2: Calculate "Derived Status" based on Members
    # ---------------------------------------------------------
    print("\n=== Calculating Group Status based on User Activity ===")
    
    df_db_users = get_db_table(spark, TABLE_USERS)
    
    if df_db_users is None or df_db_users.isEmpty():
        print("⚠️ Users table empty or missing. Assuming NO active users.")
        schema = StructType([
            StructField("u_id", LongType(), True),
            StructField("u_status", StringType(), True)
        ])
        df_user_status = spark.createDataFrame([], schema)
    else:
        df_user_status = df_db_users.select(
            col("user_id").cast("long").alias("u_id"),
            col("status").alias("u_status")
        )

    # Explode 'user_ids' to link Users
    df_exploded = df_groups_clean.select(
        col("group_id"),
        explode(col("user_ids")).alias("member_id")
    )

    df_members_status = df_exploded.join(
        df_user_status,
        df_exploded.member_id == df_user_status.u_id,
        how="left"
    )

    df_calc = df_members_status.groupBy("group_id").agg(
        spark_max(when(col("u_status") == "active", 1).otherwise(0)).alias("has_active_member")
    )

    df_groups_with_status = df_groups_clean.join(df_calc, on="group_id", how="left") \
        .withColumn("derived_status", 
                    when(col("has_active_member") == 1, "active").otherwise("inactive"))

    # ---------------------------------------------------------
    # STEP 3: Reconcile with DB (Handle Deleted Groups)
    # ---------------------------------------------------------
    print("\n=== Reconciling with Existing DB Groups ===")
    df_db_groups = get_db_table(spark, TABLE_GROUPS)

    # Initial Load
    if df_db_groups is None:
        print("-> Initial Load: Writing calculated groups to DB.")
        final_df = df_groups_with_status.select(
            "group_id", "group_name", "description", "user_ids",
            col("derived_status").alias("status"), 
            "created_at", "updated_at"
        )
        write_to_db(final_df, spark)
        return

    # FIX: Prepare DB data safely
    db_columns = df_db_groups.columns
    
    # Base columns that always exist
    sel_cols = [
        col("group_id").alias("db_gid"),
        col("group_name").alias("db_gname"),
        col("description").alias("db_desc"),
        col("status").alias("db_status"),
        col("created_at").alias("db_created"),
        col("updated_at").alias("db_updated")
    ]
    
    # --- FIX FOR DATATYPE MISMATCH ---
    if "user_ids" in db_columns:
        # DB has "101,102" (String). S3 has [101, 102] (Array<Long>).
        # We must split the DB string and cast to Array<Long> to match S3 for coalesce to work.
        sel_cols.append(
            split(col("user_ids"), ",").cast("array<long>").alias("db_user_ids")
        )
    else:
        # DB doesn't have this column yet, so we use null
        sel_cols.append(lit(None).alias("db_user_ids"))

    df_db_renamed = df_db_groups.select(*sel_cols)

    # Full Outer Join
    df_joined = df_groups_with_status.join(
        df_db_renamed,
        df_groups_with_status.group_id == df_db_renamed.db_gid,
        how="full_outer"
    )

    # ---------------------------------------------------------
    # STEP 4: Final Status Logic
    # ---------------------------------------------------------
    df_final = df_joined.select(
        coalesce(col("group_id"), col("db_gid")).alias("group_id"),
        coalesce(col("group_name"), col("db_gname")).alias("group_name"),
        coalesce(col("description"), col("db_desc")).alias("description"),
        # Now both inputs are Array<Long>, so coalesce works
        coalesce(col("user_ids"), col("db_user_ids")).alias("user_ids"),
        
        when(col("group_id").isNull(), "inactive") # Removed from S3
        .otherwise(col("derived_status"))          # Present in S3
        .alias("status"),
        
        coalesce(col("created_at"), col("db_created")).alias("created_at"),
        coalesce(col("updated_at"), col("db_updated")).alias("updated_at")
    )

    if df_db_users: df_db_users.unpersist()
    if df_db_groups: df_db_groups.unpersist()

    write_to_db(df_final, spark)


if __name__ == "__main__":
    print(f"--- Launching Group Transformer ---")
    spark = SparkSession.builder \
        .appName("GroupTransformer") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
        .config("spark.jars", POSTGRES_JAR) \
        .config("spark.driver.extraClassPath", POSTGRES_JAR) \
        .getOrCreate()

    transform_and_reconcile_groups(spark)