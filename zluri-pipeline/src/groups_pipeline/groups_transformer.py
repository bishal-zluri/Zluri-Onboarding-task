from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, coalesce, explode, max as spark_max, split, array, collect_list
from pyspark.sql.types import StructType, StructField, LongType, StringType

# Import Ingestion & Loader
from s3_reader_groups import process_groups_data
from groups_postgres_loader import get_db_table, write_to_db, TABLE_GROUPS, TABLE_USERS, POSTGRES_JAR

def transform_and_reconcile_groups(spark):
    # ---------------------------------------------------------
    # STEP 1: Ingest Groups from Local
    # ---------------------------------------------------------
    df_raw_groups = process_groups_data(spark)
    if df_raw_groups is None: 
        print("No group data found.")
        return

    df_groups_clean = df_raw_groups.select(
        col("id").cast("long").alias("group_id"),
        col("name").alias("group_name"),
        col("description"),
        col("parent_group_id").cast("long").alias("parent_group_id"), 
        col("created_at").cast("timestamp"),
        col("updated_at").cast("timestamp"),
        col("user_ids")
    ).filter(col("group_id").isNotNull())

    print(f"[INGEST] Valid Groups: {df_groups_clean.count()}")

    # ---------------------------------------------------------
    # STEP 2: Generate GroupMembers & Calculate Direct Status
    # ---------------------------------------------------------
    print("\n=== Calculating Direct Group Status (User Based) ===")
    
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

    df_members_exploded = df_groups_clean.select(
        col("group_id"),
        explode(col("user_ids")).alias("member_id")
    )

    df_group_members = df_members_exploded.join(
        df_user_status,
        df_members_exploded.member_id == df_user_status.u_id,
        how="left"
    ).select(
        col("group_id"),
        col("member_id").alias("user_id"),
        coalesce(col("u_status"), lit("inactive")).alias("user_status")
    )

    df_direct_status = df_group_members.groupBy("group_id").agg(
        spark_max(when(col("user_status") == "active", 1).otherwise(0)).alias("is_active_direct")
    )

    # ---------------------------------------------------------
    # STEP 2.5: Bottom-Up Propagation (N-Level Dynamic)
    # ---------------------------------------------------------
    print("\n=== Propagating Status Upwards (Leaf -> Root) ===")

    current_df = df_groups_clean.select("group_id", "parent_group_id") \
        .join(df_direct_status, on="group_id", how="left") \
        .na.fill(0, ["is_active_direct"]) \
        .withColumn("final_active_flag", col("is_active_direct")) \
        .cache()

    iteration = 0
    
    while True:
        iteration += 1
        
        active_nodes_with_parents = current_df.filter(col("final_active_flag") == 1) \
                                              .filter(col("parent_group_id").isNotNull())
        
        parents_to_activate = current_df.alias("parent") \
            .join(active_nodes_with_parents.alias("child"), 
                  col("parent.group_id") == col("child.parent_group_id"), 
                  "inner") \
            .filter(col("parent.final_active_flag") == 0) \
            .select(col("parent.group_id").alias("target_id")) \
            .distinct() 
            
        count_changes = parents_to_activate.count()
        if count_changes == 0:
            print(f"-> Convergence reached after {iteration-1} iterations.")
            break
            
        print(f"   Iteration {iteration}: Propagating activity to {count_changes} parent groups...")

        current_df = current_df.alias("main").join(
            parents_to_activate.alias("updates"),
            col("main.group_id") == col("updates.target_id"),
            "left"
        ).select(
            col("main.group_id"),
            col("main.parent_group_id"),
            col("main.is_active_direct"),
            when(
                (col("main.final_active_flag") == 1) | (col("updates.target_id").isNotNull()), 
                1
            ).otherwise(0).alias("final_active_flag")
        ).localCheckpoint()

    df_calc_status = current_df.select(
        col("group_id"),
        when(col("final_active_flag") == 1, "active").otherwise("inactive").alias("derived_status")
    )

    df_groups_with_status = df_groups_clean.join(df_calc_status, on="group_id", how="left")

    # ---------------------------------------------------------
    # STEP 3: Reconcile with DB
    # ---------------------------------------------------------
    print("\n=== Reconciling with Existing DB Groups ===")
    df_db_groups = get_db_table(spark, TABLE_GROUPS)

    if df_db_groups is None:
        df_final_groups = df_groups_with_status.select(
            "group_id", "group_name", "description", "user_ids", "parent_group_id",
            col("derived_status").alias("status"), 
            "created_at", "updated_at"
        )
    else:
        sel_cols = [
            col("group_id").alias("db_gid"),
            col("group_name").alias("db_gname"),
            col("description").alias("db_desc"),
            col("parent_group_id").alias("db_parent_id"),
            col("status").alias("db_status"),
            col("created_at").alias("db_created"),
            col("updated_at").alias("db_updated")
        ]
        
        if "user_ids" in df_db_groups.columns:
            sel_cols.append(split(col("user_ids"), ",").cast("array<long>").alias("db_user_ids"))
        else:
            sel_cols.append(lit(None).alias("db_user_ids"))

        df_db_renamed = df_db_groups.select(*sel_cols)

        df_joined = df_groups_with_status.join(
            df_db_renamed,
            df_groups_with_status.group_id == df_db_renamed.db_gid,
            how="full_outer"
        )

        df_final_groups = df_joined.select(
            coalesce(col("group_id"), col("db_gid")).alias("group_id"),
            coalesce(col("group_name"), col("db_gname")).alias("group_name"),
            coalesce(col("description"), col("db_desc")).alias("description"),
            coalesce(col("parent_group_id"), col("db_parent_id")).alias("parent_group_id"),
            coalesce(col("user_ids"), col("db_user_ids")).alias("user_ids"),
            
            when(col("group_id").isNull(), "inactive")
            .otherwise(col("derived_status"))
            .alias("status"),
            
            coalesce(col("created_at"), col("db_created")).alias("created_at"),
            coalesce(col("updated_at"), col("db_updated")).alias("updated_at")
        )

    df_final_groups = df_final_groups.filter(col("group_id").isNotNull())

    # ---------------------------------------------------------
    # STEP 4: Write Both Tables
    # ---------------------------------------------------------
    write_to_db(df_final_groups, df_group_members, spark)


if __name__ == "__main__":
    print(f"--- Launching Group Transformer (Local) ---")
    
    spark = (
        SparkSession.builder
        .appName("GroupTransformer")
        # FIXED: Forces Localhost binding to avoid VPN/Network IP issues
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1") 
        .config("spark.jars", POSTGRES_JAR) 
        .config("spark.driver.extraClassPath", POSTGRES_JAR)
        .getOrCreate()
    )

    transform_and_reconcile_groups(spark)