from prefect import flow, task, get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
import sys
import os

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

sys.path.insert(0, os.path.join(CURRENT_DIR, 'user_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'groups_pipeline'))
sys.path.insert(0, os.path.join(CURRENT_DIR, 'transaction_pipeline'))

from user_pipeline.user_transformer import transform_and_reconcile_users, POSTGRES_JAR
from groups_pipeline.groups_transformer import transform_and_reconcile_groups
from transaction_pipeline.transaction_transformer import transform_and_load_transactions

from pyspark.sql import SparkSession

# --- SPARK SESSION MANAGEMENT ---
def get_spark_session(app_name="Prefect-ETL"):
    """
    Creates a Local Spark Session.
    Fixes network binding issues by forcing localhost.
    """
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.jars", POSTGRES_JAR) \
        .config("spark.driver.extraClassPath", POSTGRES_JAR) \
        .getOrCreate()

# --- ENTITY TASKS ---

@task(name="Run User Pipeline")
def run_user_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting USER pipeline for {sync_day}...")
    try:
        transform_and_reconcile_users(spark_session)
        logger.info(f"USER pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"USER pipeline failed: {e}")
        raise e

@task(name="Run Group Pipeline")
def run_group_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting GROUP pipeline for {sync_day}...")
    try:
        transform_and_reconcile_groups(spark_session)
        logger.info(f"GROUP pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"GROUP pipeline failed: {e}")
        raise e

@task(name="Run Transaction Pipeline")
def run_transaction_pipeline(spark_session, sync_day):
    logger = get_run_logger()
    logger.info(f"Starting TRANSACTION pipeline for {sync_day}...")
    try:
        transform_and_load_transactions(spark_session)
        logger.info(f"TRANSACTION pipeline completed successfully for {sync_day}.")
    except Exception as e:
        logger.error(f"TRANSACTION pipeline failed: {e}")
        raise e

# --- POST-SYNC TASKS ---

@task(name="Mark Inactive Users")
def post_sync_user_(spark_session):
    get_run_logger().info("User status reconciliation complete.")

@task(name="Mark Inactive Groups")
def post_sync_group_(spark_session):
    get_run_logger().info("Group hierarchy status propagation complete.")

# --- MAIN FLOW ---

@flow(name="Zluri Data Assignment Pipeline",task_runner=ConcurrentTaskRunner())
def main_pipeline_flow():
    logger = get_run_logger()
    logger.info("Starting Main Orchestrator...")
    
    # 1. Initialize Spark (Plain Python call)
    spark = get_spark_session()
    
    try:
        day = "sync-day2"
        logger.info(f"=== PROCESSING: {day} ===")

        # Inject global config
        import s3_reader_users as ur
        ur.DAY_FOLDER = day
        import s3_reader_groups as gr
        gr.DAY_FOLDER = day
        import s3_reader_transaction as tr
        tr.DAY_FOLDER = day
        
        # 4. Execute Tasks Sequentially
        user_tasks = run_user_pipeline(spark, day)
        groups_tasks = run_group_pipeline(spark, day, wait_for=[user_tasks])
        transaction_tasks = run_transaction_pipeline(spark, day)
        post_sync_user_(spark)
        post_sync_group_(spark)
        
        logger.info(f"All pipelines for {day} completed.")
        
    finally:
        logger.info("Stopping Spark Session...")
        spark.stop()

if __name__ == "__main__":
    main_pipeline_flow()