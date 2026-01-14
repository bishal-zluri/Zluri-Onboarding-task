from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, explode, max as spark_max, split, array, collect_list
from pyspark.sql.types import StructType, StructField, LongType, StringType

# Import Ingestion & Loader
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
        col("user_ids") # Array of user IDs
    ).filter(col("group_id").isNotNull())

    print(f"[INGEST] Valid S3 Groups: {df_groups_clean.count()}")

    # ---------------------------------------------------------
    # STEP 2: Generate GroupMembers & Calculate Status
    # ---------------------------------------------------------
    print("\n=== generating GroupMembers and Calculating Status ===")
    
    # A. Get User Status from DB
    df_db_users = get_db_table(spark, TABLE_USERS)
    
    if df_db_users is None or df_db_users.isEmpty():
        print("⚠️ Users table empty. Assuming all users inactive.")
        df_user_status = spark.createDataFrame([], schema=StructType([
            StructField("u_id", LongType(), True), 
            StructField("u_status", StringType(), True)
        ]))
    else:
        df_user_status = df_db_users.select(
            col("user_id").cast("long").alias("u_id"),
            col("status").alias("u_status")
        )

    # B. Explode to create GroupMembers Table
    # This creates one row per User per Group
    df_members_exploded = df_groups_clean.select(
        col("group_id"),
        explode(col("user_ids")).alias("member_id")
    )

    # C. Join with User Status
    df_group_members = df_members_exploded.join(
        df_user_status,
        df_members_exploded.member_id == df_user_status.u_id,
        how="left"
    ).select(
        col("group_id"),
        col("member_id").alias("user_id"),
        # If user not found in DB, assume inactive
        coalesce(col("u_status"), lit("inactive")).alias("user_status")
    )

    # D. Calculate Group Status
    # Logic: Group is Active if at least ONE member is 'active'
    df_group_status_calc = df_group_members.groupBy("group_id").agg(
        spark_max(when(col("user_status") == "active", 1).otherwise(0)).alias("has_active_member")
    )

    # E. Join Status back to Groups
    df_groups_with_status = df_groups_clean.join(df_group_status_calc, on="group_id", how="left") \
        .withColumn("derived_status", 
                    when(col("has_active_member") == 1, "active").otherwise("inactive"))

    # ---------------------------------------------------------
    # STEP 3: Reconcile with DB (Handle Deleted Groups)
    # ---------------------------------------------------------
    print("\n=== Reconciling with Existing DB Groups ===")
    df_db_groups = get_db_table(spark, TABLE_GROUPS)

    if df_db_groups is None:
        # Initial Load
        df_final_groups = df_groups_with_status.select(
            "group_id", "group_name", "description", "user_ids",
            col("derived_status").alias("status"), 
            "created_at", "updated_at"
        )
    else:
        # Prepare DB Data for Join
        sel_cols = [
            col("group_id").alias("db_gid"),
            col("group_name").alias("db_gname"),
            col("description").alias("db_desc"),
            col("status").alias("db_status"),
            col("created_at").alias("db_created"),
            col("updated_at").alias("db_updated")
        ]
        
        # Handle Array vs String mismatch for user_ids
        if "user_ids" in df_db_groups.columns:
            sel_cols.append(split(col("user_ids"), ",").cast("array<long>").alias("db_user_ids"))
        else:
            sel_cols.append(lit(None).alias("db_user_ids"))

        df_db_renamed = df_db_groups.select(*sel_cols)

        # Full Join
        df_joined = df_groups_with_status.join(
            df_db_renamed,
            df_groups_with_status.group_id == df_db_renamed.db_gid,
            how="full_outer"
        )

        # Final Selection
        df_final_groups = df_joined.select(
            coalesce(col("group_id"), col("db_gid")).alias("group_id"),
            coalesce(col("group_name"), col("db_gname")).alias("group_name"),
            coalesce(col("description"), col("db_desc")).alias("description"),
            coalesce(col("user_ids"), col("db_user_ids")).alias("user_ids"),
            
            # Status Logic: If missing in S3 -> Inactive, else Calculated Status
            when(col("group_id").isNull(), "inactive")
            .otherwise(col("derived_status"))
            .alias("status"),
            
            coalesce(col("created_at"), col("db_created")).alias("created_at"),
            coalesce(col("updated_at"), col("db_updated")).alias("updated_at")
        )

    # ---------------------------------------------------------
    # STEP 4: Write Both Tables
    # ---------------------------------------------------------
    # 1. df_final_groups: The main groups list (Upserted)
    # 2. df_group_members: The join table (Synced)
    
    write_to_db(df_final_groups, df_group_members, spark)


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