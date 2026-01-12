from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, array, lit, when, size
from functools import reduce

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day1"  # Update this to 'sync-day2' when running the second batch

# TARGET ENTITY FOLDER
# Based on your screenshot, the folder name is 'admin_groups'
ENTITY_GROUPS = "admin_groups" 

# Base S3A Path
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. SPARK SESSION FACTORY ---
def create_spark_session(app_name="GroupDataExtractor"):
    """
    Creates a Spark session configured for Public S3 access.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider") \
        .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60") \
        .config("spark.hadoop.fs.s3a.connection.establish.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "30000") \
        .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400") \
        .getOrCreate()

# --- 3. GENERIC READER FUNCTION (Unchanged) ---
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
                # Multiline is often needed for complex nested JSON groups
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

# --- 4. DATA PROCESSING LOGIC (Modified for Groups) ---

def process_groups_data(spark):
    """
    Ingests Group data from the 'admin_groups' folder.
    Ensures 'user_ids' are extracted as an array.
    """
    
    # Step A: Load the 'Groups' Data
    print(f"--- Loading Groups from folder: {ENTITY_GROUPS} ---")
    df_groups = read_s3_data(spark, ENTITY_GROUPS)
    
    if df_groups is None:
        print("Critical Error: Could not find group files.")
        return None

    # Step B: Extract User IDs
    # We check for common patterns: 'agent_ids' (flat array) or 'users' (array of objects with 'id')
    cols = df_groups.columns
    
    if "agent_ids" in cols:
        # If 'agent_ids' exists, alias it to 'user_ids' if not present
        print("  -> Found 'agent_ids', mapping to 'user_ids'.")
        df_groups = df_groups.withColumn("user_ids", col("agent_ids"))
        
    elif "users" in cols:
        # If 'users' exists (likely array of structs), extract the 'id' field from it
        # This transforms [{id: 1}, {id: 2}] -> [1, 2]
        print("  -> Found 'users' object list, extracting IDs to 'user_ids'.")
        df_groups = df_groups.withColumn("user_ids", col("users.id"))
        
    elif "members" in cols:
        # Fallback for 'members' pattern
        print("  -> Found 'members' object list, extracting IDs to 'user_ids'.")
        df_groups = df_groups.withColumn("user_ids", col("members.id"))
        
    else:
        # If no identifiable user list is found, create an empty array column
        print("  [!] No user/member list found. Creating empty 'user_ids' column.")
        df_groups = df_groups.withColumn("user_ids", array())

    # Ensure user_ids is actually an array (handle potential nulls or types)
    # This coalesce ensures we don't have null columns, but empty arrays instead
    df_groups = df_groups.withColumn("user_ids", 
                                     when(col("user_ids").isNull(), array())
                                     .otherwise(col("user_ids")))

    # Step C: Validation / Preview
    count = df_groups.count()
    print(f"\n--- Success! Total Raw Groups Ingested: {count} ---")
    
    # Print Schema to confirm user_ids is present
    df_groups.printSchema()
    
    return df_groups


# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    spark_session = create_spark_session()
    
    # Run Ingestionx
    raw_groups_df = process_groups_data(spark_session)
    
    # Show sample if successful, explicitly selecting user_ids to verify
    if raw_groups_df:
        print("\n--- Previewing Data (including user_ids) ---")
        # We select a few key columns + user_ids for clarity in the output
        display_cols = [c for c in raw_groups_df.columns if c in ["id", "name", "group_name", "user_ids", "agent_ids"]]
        raw_groups_df.select(*display_cols).show(truncate=False)