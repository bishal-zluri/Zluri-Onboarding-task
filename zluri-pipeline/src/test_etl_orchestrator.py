import pytest
import sys
import os
import warnings
from unittest.mock import MagicMock, patch, call

# --- 1. GLOBAL MOCK SETUP (CRITICAL) ---
mock_user_transformer = MagicMock()
mock_user_transformer.POSTGRES_JAR = "mock_postgres.jar" 

mock_groups_transformer = MagicMock()
mock_trans_transformer = MagicMock()
mock_spark_session = MagicMock()

mock_s3_user = MagicMock()
mock_s3_group = MagicMock()
mock_s3_trans = MagicMock()

module_patches = {
    "user_pipeline": MagicMock(),
    "user_pipeline.user_transformer": mock_user_transformer,
    "groups_pipeline": MagicMock(),
    "groups_pipeline.groups_transformer": mock_groups_transformer,
    "transaction_pipeline": MagicMock(),
    "transaction_pipeline.transaction_transformer": mock_trans_transformer,
    "pyspark.sql": MagicMock(),
    "s3_reader_users": mock_s3_user,
    "s3_reader_groups": mock_s3_group,
    "s3_reader_transaction": mock_s3_trans,
}
sys.modules.update(module_patches)

# --- 2. CONFIG FIXTURES ---
@pytest.fixture(autouse=True, scope="session")
def configure_prefect_env():
    os.environ["PREFECT_EVENTS_CLIENT_ENABLED"] = "False"
    os.environ["PREFECT_LOGGING_TO_API_ENABLED"] = "False"
    os.environ["PREFECT_PROFILE"] = "test"

@pytest.fixture(autouse=True)
def ignore_unwanted_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="coolname.loader")

# --- 3. IMPORT MODULE ---
import etl_orchestrator as pipeline

# --- 4. TEST FIXTURES ---
@pytest.fixture
def mock_logger():
    with patch("etl_orchestrator.get_run_logger") as mock:
        yield mock

@pytest.fixture
def mock_spark():
    return MagicMock(name="SparkSession")

# --- 5. TESTS ---

class TestSparkSession:
    @patch("pyspark.sql.SparkSession.builder")
    def test_get_spark_session(self, mock_builder):
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = "MockSparkSession"

        session = pipeline.get_spark_session.fn()

        assert session == "MockSparkSession"
        mock_builder.appName.assert_called_with("Prefect-ETL")
        args_list = mock_builder.config.call_args_list
        assert any("mock_postgres.jar" in str(call) for call in args_list)


class TestPipelineTasks:
    def test_run_user_pipeline_success(self, mock_logger, mock_spark):
        pipeline.run_user_pipeline.fn(mock_spark, "day1")
        mock_user_transformer.transform_and_reconcile_users.assert_called_once_with(mock_spark)
        mock_logger.return_value.info.assert_any_call("USER pipeline completed successfully for day1.")

    def test_run_user_pipeline_failure(self, mock_logger, mock_spark):
        mock_user_transformer.transform_and_reconcile_users.side_effect = RuntimeError("Boom")
        with pytest.raises(RuntimeError):
            pipeline.run_user_pipeline.fn(mock_spark, "day1")
        mock_logger.return_value.error.assert_called_with("USER pipeline failed: Boom")

    def test_run_group_pipeline_success(self, mock_logger, mock_spark):
        pipeline.run_group_pipeline.fn(mock_spark, "day1")
        mock_groups_transformer.transform_and_reconcile_groups.assert_called_once_with(mock_spark)

    def test_run_transaction_pipeline_success(self, mock_logger, mock_spark):
        pipeline.run_transaction_pipeline.fn(mock_spark, "day1")
        mock_trans_transformer.transform_and_load_transactions.assert_called_once_with(mock_spark)


class TestPostSyncTasks:
    def test_post_sync_logs(self, mock_logger, mock_spark):
        pipeline.post_sync_user_.fn(mock_spark)
        # FIX 1: Updated string to match actual code
        mock_logger.return_value.info.assert_any_call("User status reconciliation complete.")


class TestMainFlow:
    @patch("etl_orchestrator.post_sync_group_")
    @patch("etl_orchestrator.post_sync_user_")
    @patch("etl_orchestrator.run_transaction_pipeline")
    @patch("etl_orchestrator.run_group_pipeline")
    @patch("etl_orchestrator.run_user_pipeline")
    @patch("etl_orchestrator.get_spark_session")
    def test_flow_execution_logic(self, mock_get_spark, mock_user_task, mock_group_task, mock_trans_task, mock_post_user, mock_post_group):
        """
        Verifies initialization and that tasks are SUBMITTED correctly.
        """
        # Execute
        pipeline.main_pipeline_flow()

        # Assert Spark Init
        mock_get_spark.assert_called_once()
        spark_instance = mock_get_spark.return_value

        # Assert Global Config Injection
        expected_day = "sync-day2"
        assert mock_s3_user.DAY_FOLDER == expected_day
        assert mock_s3_group.DAY_FOLDER == expected_day
        assert mock_s3_trans.DAY_FOLDER == expected_day

        # FIX 2: Check .submit() instead of direct calls
        
        # A. Check User Submission
        mock_user_task.submit.assert_called_once_with(spark_instance, expected_day)
        user_future = mock_user_task.submit.return_value

        # B. Check Group Submission (Waiting for User)
        mock_group_task.submit.assert_called_once()
        _, kwargs = mock_group_task.submit.call_args
        assert kwargs["wait_for"] == [user_future]
        
        # C. Check Transaction Submission (No Wait)
        mock_trans_task.submit.assert_called_once_with(spark_instance, expected_day)

        # D. Check Wait calls (Ensuring logic waits for completion)
        # Verify that .wait() was called on the futures returned by submit
        mock_trans_task.submit.return_value.wait.assert_called_once()
        mock_post_user.submit.return_value.wait.assert_called_once()
        mock_post_group.submit.return_value.wait.assert_called_once()