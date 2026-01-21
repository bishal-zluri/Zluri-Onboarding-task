from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, array, lit, when, size, explode
from functools import reduce
import os

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day1"

# TARGET ENTITY FOLDER
ENTITY_GROUPS = "admin_groups" 

# Base S3A Path (kept for reference, though local reader overrides it)
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. READER UTILS ---
def read_local_data(spark, folder_name):
    """
    Scans a specific entity folder in the local filesystem.
    Navigates up from src/groups_pipeline -> src -> zluri-pipeline -> ONBOARDING TASK
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    target_dir = os.path.join(project_root, DAY_FOLDER, folder_name)
    base_path = f"file://{target_dir}"
    
    formats = ['json', 'csv', 'parquet']
    found_dfs = []

    print(f"Scanning Local Path: {base_path} ...")

    for fmt in formats:
        full_pattern = os.path.join(base_path, f"*.{fmt}")
        try:
            if not os.path.exists(target_dir):
                continue

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
        print(f"  [!] No data found for entity: {folder_name} at {target_dir}")
        return None

    final_df = reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), found_dfs)
    return final_df

def process_groups_data(spark):
    """
    Ingests Group data. Handles nested 'results' array if present.
    """
    print(f"--- Loading Groups from folder: {ENTITY_GROUPS} ---")
    
    # Use local reader
    df_groups = read_local_data(spark, ENTITY_GROUPS)
    
    if df_groups is None:
        return None

    if "results" in df_groups.columns:
        print("  -> Exploding 'results' array...")
        df_groups = df_groups.select(explode(col("results")).alias("g")).select("g.*")

    cols = df_groups.columns
    
    if "agent_ids" in cols:
        df_groups = df_groups.withColumn("user_ids", col("agent_ids"))
    elif "users" in cols:
        df_groups = df_groups.withColumn("user_ids", col("users.id"))
    elif "members" in cols:
        df_groups = df_groups.withColumn("user_ids", col("members.id"))
    else:
        df_groups = df_groups.withColumn("user_ids", array())

    df_groups = df_groups.withColumn("user_ids", 
                                     when(col("user_ids").isNull(), array())
                                     .otherwise(col("user_ids")))
    
    if "parent_group_id" not in df_groups.columns:
        print("  [!] 'parent_group_id' missing in source, defaulting to null.")
        df_groups = df_groups.withColumn("parent_group_id", lit(None))

    return df_groups

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- Running Groups Reader Individually ---")
    spark = SparkSession.builder \
        .appName("GroupsReaderLocal") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .master("local[*]") \
        .getOrCreate()
    
    try:
        df = process_groups_data(spark)
        if df:
            df.show(truncate=False)
            print(f"Total Groups Processed: {df.count()}")
    finally:
        spark.stop()