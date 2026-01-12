from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, lit, explode, collect_list, concat_ws
from pyspark.sql.utils import AnalysisException
from functools import reduce

# --- 1. CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day2"

ENTITY_AGENTS_INDEX = "agents"          
ENTITY_AGENT_DETAILS = "agent_details"  
ENTITY_ROLES = "roles"                  

# Base S3A Path
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

# --- 2. SPARK SESSION FACTORY ---
def create_spark_session(app_name="AgentDataExtractor"):
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

# --- 3. GENERIC READER FUNCTION ---
def read_s3_data(spark, folder_name):
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
        
    return reduce(lambda df1, df2: df1.unionByName(df2, allowMissingColumns=True), found_dfs)

# --- 4. DATA PROCESSING LOGIC ---

def process_agents_data(spark):
    """
    Orchestrates extraction of Agents, Details, and Roles, then joins them.
    """
    
    # 1. Load DataFrames
    df_agents_index = read_s3_data(spark, ENTITY_AGENTS_INDEX)
    df_agent_details = read_s3_data(spark, ENTITY_AGENT_DETAILS)
    df_roles = read_s3_data(spark, ENTITY_ROLES)

    if not df_agents_index or not df_agent_details:
        print("Critical Error: Missing Agents or Agent Details data.")
        return

    print("\n--- Performing Data Joins ---")

    try:
        # Step A: Join Users + Details
        df_users_raw = df_agents_index.alias("idx").join(
            df_agent_details.alias("det"),
            on="id", 
            how="left"
        )

        # If roles data is missing, return raw user data
        if df_roles is None:
            print("Warning: Roles data missing. Returning data without Role Names.")
            return df_users_raw

        # Step B: Flatten User Roles
        # FIX: Added 'det.group_ids' to the selection list
        df_exploded = df_users_raw.select(
            col("idx.id").alias("user_id"),
            col("det.contact.name").alias("user_name"),
            col("det.contact.email").alias("user_email"),
            col("det.created_at").alias("user_created_at"),
            col("det.updated_at").alias("user_updated_at"),
            col("det.group_ids"),
            explode(col("det.role_ids")).alias("single_role_id") 
        )

        # Step C: Join with Roles Data
        df_joined = df_exploded.join(
            df_roles.alias("r"),
            col("single_role_id") == col("r.id"),
            how="left"
        )

        # Step D: Re-aggregate
        # FIX: Added 'group_ids' to the groupBy list so it persists
        final_df = df_joined.groupBy(
            "user_id", "user_name", "user_email", "user_created_at", "user_updated_at", "group_ids"
        ).agg(
            collect_list("r.name").alias("role_names"),
            collect_list("r.description").alias("role_descriptions")
        )
        
        print(f"\n--- Success! Enriched Users with Roles: {final_df.count()} rows ---")
        # final_df.show(truncate=False)
        
        return final_df

    except Exception as e:
        print(f"Processing Failed. Error: {e}")
        import traceback
        traceback.print_exc()

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    spark_session = create_spark_session()
    process_agents_data(spark_session)