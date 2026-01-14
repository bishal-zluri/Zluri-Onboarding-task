from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, array, lit, when, size, explode
from functools import reduce

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day2"

# TARGET ENTITY FOLDER
ENTITY_GROUPS = "admin_groups" 

# Base S3A Path
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. READER UTILS ---
def read_s3_data(spark, folder_name):
    """
    Scans a specific entity folder across the sync-day path.
    """
    base_path = f"{S3_BASE_PATH}/{DAY_FOLDER}/{folder_name}/"
    formats = ['json', 'csv', 'parquet']
    found_dfs = []

    print(f"Scanning: {base_path} ...")

    for fmt in formats:
        full_pattern = f"{base_path}*.{fmt}"
        try:
            if fmt == 'json':
                df = spark.read.option("multiline", "true").json(full_pattern)
            elif fmt == 'csv':
                df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_pattern)
            elif fmt == 'parquet':
                df = spark.read.parquet(full_pattern)
            
            df = df.withColumn("source_path", input_file_name())
            
            if not df.isEmpty():
                print(f"  -> Loaded {fmt} files.")
                found_dfs.append(df)
        except Exception:
            continue

    if not found_dfs:
        print(f"  [!] No data found for entity: {folder_name}")
        return None

    final_df = reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), found_dfs)
    return final_df

def process_groups_data(spark):
    """
    Ingests Group data. Handles nested 'results' array if present.
    """
    print(f"--- Loading Groups from folder: {ENTITY_GROUPS} ---")
    df_groups = read_s3_data(spark, ENTITY_GROUPS)
    
    if df_groups is None:
        return None

    # Handle 'results' wrapper if it exists (Common in API responses)
    if "results" in df_groups.columns:
        print("  -> Exploding 'results' array...")
        df_groups = df_groups.select(explode(col("results")).alias("g")).select("g.*")

    cols = df_groups.columns
    
    # Normalize User IDs column
    if "agent_ids" in cols:
        df_groups = df_groups.withColumn("user_ids", col("agent_ids"))
    elif "users" in cols:
        df_groups = df_groups.withColumn("user_ids", col("users.id"))
    elif "members" in cols:
        df_groups = df_groups.withColumn("user_ids", col("members.id"))
    else:
        df_groups = df_groups.withColumn("user_ids", array())

    # Ensure it's not null
    df_groups = df_groups.withColumn("user_ids", 
                                     when(col("user_ids").isNull(), array())
                                     .otherwise(col("user_ids")))
    
    # Normalize parent_group_id
    if "parent_group_id" not in df_groups.columns:
        print("  [!] 'parent_group_id' missing in source, defaulting to null.")
        df_groups = df_groups.withColumn("parent_group_id", lit(None))

    return df_groups