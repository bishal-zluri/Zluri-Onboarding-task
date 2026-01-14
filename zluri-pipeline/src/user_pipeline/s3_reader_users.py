from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, col, explode_outer, lit
from pyspark.sql.utils import AnalysisException
from functools import reduce

# `s3://zluri-data-assignment/assignment-jan-2026/`
# --- CONFIGURATION ---
BUCKET_NAME = "zluri-data-assignment"
BASE_FOLDER = "assignment-jan-2026"
DAY_FOLDER = "sync-day2"

ENTITY_AGENTS_INDEX = "agents"          
ENTITY_AGENT_DETAILS = "agent_details"  
ENTITY_ROLES = "roles"                  
S3_BASE_PATH = f"s3a://{BUCKET_NAME}/{BASE_FOLDER}"

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

# --- ROBUST READER FUNCTION ---
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

def process_agents_data(spark):
    """
    Returns the joined, EXPLODED dataframe (one row per user-role combo).
    Uses explode_outer to preserve users with NO roles.
    """
    df_idx = read_s3_data(spark, ENTITY_AGENTS_INDEX)
    df_det = read_s3_data(spark, ENTITY_AGENT_DETAILS)
    df_rol = read_s3_data(spark, ENTITY_ROLES)

    if not df_idx or not df_det: return None

    # Join Users + Details
    df_users = df_idx.alias("idx").join(df_det.alias("det"), on="id", how="left")

    # --- CRITICAL CHANGE: explode_outer ---
    # If role_ids is null or empty, this keeps the user row with role_id = NULL
    df_exploded = df_users.select(
        col("idx.id").alias("user_id"),
        col("det.contact.name").alias("user_name"),
        col("det.contact.email").alias("user_email"),
        col("det.created_at").alias("created_at"),
        col("det.updated_at").alias("updated_at"),
        col("det.group_ids").alias("raw_group_ids"), 
        explode_outer(col("det.role_ids")).alias("role_id")
    )

    # Join with Roles to get Role Names
    if df_rol:
        # FIX APPLIED HERE:
        # 1. Alias df_exploded as "u" (users)
        # 2. Alias df_rol as "r" (roles)
        # 3. Explicitly select "u.created_at" to avoid ambiguity with "r.created_at"
        df_final = df_exploded.alias("u").join(df_rol.alias("r"), col("u.role_id") == col("r.id"), how="left") \
            .select(
                col("u.user_id"), 
                col("u.user_name"), 
                col("u.user_email"), 
                col("u.created_at"),  # Explicitly pick User's created_at
                col("u.updated_at"),  # Explicitly pick User's updated_at
                col("u.role_id"), 
                col("r.name").alias("role_name"), 
                col("r.description").alias("role_desc")
            )
    else:
        # Fallback if no roles table exists at all
        df_final = df_exploded.select("*", lit(None).alias("role_name"), lit(None).alias("role_desc"))

    return df_final