import statistics
from unittest.mock import MagicMock

import pytest
import torch
from lightning import LightningModule, Trainer
from torchmetrics import MeanAbsoluteError, MetricCollection

from icenet_mp.callbacks.metric_summary_callback import MetricSummaryCallback
from icenet_mp.models.metrics.base_metrics import MAEDaily, RMSEDaily
from icenet_mp.models.metrics.sie_error_new import SIEErrorNew
from icenet_mp.types import ModelTestOutput


@pytest.fixture
def callback() -> MetricSummaryCallback:
    """Create a MetricSummaryCallback instance."""
    return MetricSummaryCallback(average_loss=True)


@pytest.fixture
def mock_trainer() -> MagicMock:
    """Create a mock Trainer."""
    trainer = MagicMock(spec=Trainer)
    mock_logger = MagicMock()
    trainer.loggers = [mock_logger]
    return trainer


@pytest.fixture
def mock_module() -> MagicMock:
    """Create a mock LightningModule."""
    return MagicMock(spec=LightningModule)


class TestMetricSummaryCallbackInit:
    """Tests for MetricSummaryCallback initialization."""

    def test_init_with_average_loss_true(self) -> None:
        """Test initialization with average_loss=True."""
        callback = MetricSummaryCallback(average_loss=True)
        assert "average_loss" in callback.metrics
        assert callback.metrics["average_loss"] == []

    def test_init_with_average_loss_false(self) -> None:
        """Test initialization with average_loss=False."""
        callback = MetricSummaryCallback(average_loss=False)
        assert "average_loss" not in callback.metrics


class TestOnTestBatchEnd:
    """Tests for on_test_batch_end method."""

    def test_on_test_batch_end_with_valid_output(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test that loss is accumulated when output is ModelTestOutput."""
        loss_value = 0.5
        prediction = torch.randn(1, 10)
        target = torch.randn(1, 10)
        output = ModelTestOutput(
            loss=torch.tensor(loss_value), prediction=prediction, target=target
        )

        callback.on_test_batch_end(
            mock_trainer, mock_module, output, _batch=None, _batch_idx=0
        )

        assert loss_value in callback.metrics["average_loss"]

    def test_on_test_batch_end_with_invalid_output(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test that invalid output types are skipped."""
        callback.on_test_batch_end(
            mock_trainer, mock_module, outputs=None, _batch=None, _batch_idx=0
        )

        assert callback.metrics["average_loss"] == []

    def test_on_test_batch_end_multiple_batches(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test accumulation of loss across multiple batches."""
        losses = [0.1, 0.2, 0.3]
        prediction = torch.randn(1, 10)
        target = torch.randn(1, 10)
        for i, loss_value in enumerate(losses):
            output = ModelTestOutput(
                loss=torch.tensor(loss_value), prediction=prediction, target=target
            )
            callback.on_test_batch_end(
                mock_trainer, mock_module, output, _batch=None, _batch_idx=i
            )

        assert callback.metrics["average_loss"] == pytest.approx(losses)


class TestOnTestEpochEnd:
    """Tests for on_test_epoch_end method."""

    def test_on_test_epoch_end_logs_average_loss(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test that on_test_epoch_end computes and logs average loss."""
        losses = [0.1, 0.2, 0.3]
        callback.metrics["average_loss"] = losses

        callback.on_test_epoch_end(mock_trainer, mock_module)

        expected_average = statistics.mean(losses)
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once()
        call_args = mock_logger.log_metrics.call_args[0][0]
        assert call_args["average_loss"] == expected_average

    def test_on_test_epoch_end_with_empty_metrics(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_epoch_end with no accumulated metrics."""
        callback.on_test_epoch_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called_once_with({})


class TestOnTestEnd:
    """Tests for on_test_end method."""

    def test_on_test_end_with_metric_collection(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end with a valid MetricCollection."""
        metric_collection = MetricCollection({"mae": MeanAbsoluteError()})
        mock_module.test_metrics = metric_collection

        # Create sample predictions and targets
        preds = torch.randn(10)
        targets = torch.randn(10)

        for pred, target in zip(preds, targets, strict=False):
            metric_collection.update(pred.unsqueeze(0), target.unsqueeze(0))

        callback.on_test_end(mock_trainer, mock_module)

        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_called()

    def test_on_test_end_with_invalid_test_metrics(
        self,
        callback: MetricSummaryCallback,
        mock_trainer: MagicMock,
        mock_module: MagicMock,
    ) -> None:
        """Test on_test_end when test_metrics is not a MetricCollection."""
        mock_module.test_metrics = "invalid"

        callback.on_test_end(mock_trainer, mock_module)

        # Should not raise an error, just log a warning
        mock_logger = mock_trainer.loggers[0]
        mock_logger.log_metrics.assert_not_called()


class TestMetricCalculations:
    """Tests for on_test_end method."""

    def test_calculates_mean_mae_correctly(self) -> None:
        """Test that MAE is calculated correctly."""
        # Create predictable test data
        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([[1.5], [2.5], [2.5], [4.5]])
        computed_mae = MeanAbsoluteError()
        computed_mae.update(preds, targets)

        # Expected MAE: (|1.0-1.5| + |2.0-2.5| + |3.0-2.5| + |4.0-4.5|) / 4
        # = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        expected_mae = 0.5

        # check expected_mae matches computed_mae.compute()
        assert computed_mae.compute().item() == pytest.approx(expected_mae, abs=1e-5)

    def test_calculates_mean_mae_daily_correctly(self) -> None:
        """Test that MAE daily is calculated correctly."""
        # Convert 2D tensor to 5D tensor: (batch, channels, height, width, time)
        preds_2d = torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets_2d = torch.tensor(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_mae = MAEDaily()
        computed_mae.update(preds, targets)
        daily_result = computed_mae.compute()
        # Expected MAE per day:
        # Day 1: (|1.0-1.5| + |1.0-0.5| + |2.0-2.0| + |2.0-2.5|) / 4 = 0.375
        # Day 2: (|2.0-2.5| + |3.0-3.5| + |3.0-4.0| + |4.0-3.0|) / 4 = 0.75
        # Day 3: (|4.0-4.0| + |4.0-4.0| + |5.0-5.0| + |6.0-6.0|) / 4 = 0.0
        expected_mae = torch.tensor([0.375, 0.75, 0.0])

        assert torch.allclose(daily_result, expected_mae, atol=1e-5)

    def test_calculates_mean_rmse_correctly(self) -> None:
        """Test that RMSE is calculated correctly."""
        preds = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        targets = torch.tensor([[1.5], [2.5], [2.5], [4.5]])

        # Expected RMSE:
        # Errors: [0.5, 0.5, 0.5, 0.5] -> MSE = 0.25 -> RMSE = 0.5
        expected_rmse = 0.5
        computed_rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

        assert computed_rmse == pytest.approx(expected_rmse, abs=1e-5)

    def test_calculates_mean_rmse_daily_correctly(self) -> None:
        """Test that RMSE daily is calculated correctly."""
        preds_2d = torch.tensor(
            [[1.0, 2.0, 4.0], [1.0, 3.0, 4.0], [2.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        targets_2d = torch.tensor(
            [[1.5, 2.5, 4.0], [0.5, 3.5, 4.0], [2.0, 4.0, 5.0], [2.5, 3.0, 6.0]]
        )

        # Reshape to 5D: (batch=1, time=3, channels=1, height=2, width=2)
        preds = preds_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        targets = targets_2d.view(2, 2, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)

        computed_rmse = RMSEDaily()
        computed_rmse.update(preds, targets)
        daily_result = computed_rmse.compute()

        # Expected RMSE per day:
        # Day 1: sqrt(mean([0.25, 0.25, 0.0, 0.25])) = sqrt(0.1875) = 0.4330127
        # Day 2: sqrt(mean([0.25, 0.25, 1.0, 1.0])) = sqrt(0.625) = 0.7905694
        # Day 3: sqrt(mean([0.0, 0.0, 0.0, 0.0])) = 0.0
        expected_rmse = torch.tensor([0.4330127, 0.7905694, 0.0])

        assert torch.allclose(daily_result, expected_rmse, atol=1e-5)

    def test_calculates_sieerror_new_daily_correctly(self) -> None:
        """Test that SIEErrorNew is calculated correctly per day."""
        metric = SIEErrorNew(pixel_size=1)

        # Shape: (batch=1, time=3, channels=1, height=1, width=2)
        preds = torch.tensor(
            [
                [
                    [[[0.2, 0.2]]],  # day 1 -> 2 ice
                    [[[0.0, 0.2]]],  # day 2 -> 1 ice
                    [[[0.0, 0.2]]],  # day 3 -> 1 ice
                ]
            ]
        )

        targets = torch.tensor(
            [
                [
                    [[[0.2, 0.0]]],  # day 1 -> 1 ice
                    [[[0.2, 0.2]]],  # day 2 -> 2 ice
                    [[[0.0, 0.0]]],  # day 3 -> 0 ice
                ]
            ]
        )

        metric.update(preds, targets)
        result = metric.compute()

        # Errors per day: [2-1, 1-2, 1-0] -> abs: [1, 1, 1]
        expected = torch.tensor([1.0, 1.0, 1.0])

        assert torch.allclose(result, expected, atol=1e-5)

    def test_calculates_sieerror_new_scaled_by_pixel_size(self) -> None:
        """Test that SIEErrorNew scales by pixel_size^2."""
        metric = SIEErrorNew(pixel_size=5)

        preds = torch.tensor(
            [
                [
                    [[[0.2, 0.2]]],  # day 1 -> 2 ice
                    [[[0.0, 0.2]]],  # day 2 -> 1 ice
                    [[[0.0, 0.2]]],  # day 3 -> 1 ice
                ]
            ]
        )

        targets = torch.tensor(
            [
                [
                    [[[0.2, 0.0]]],  # day 1 -> 1 ice
                    [[[0.2, 0.2]]],  # day 2 -> 2 ice
                    [[[0.0, 0.0]]],  # day 3 -> 0 ice
                ]
            ]
        )

        metric.update(preds, targets)
        result = metric.compute()

        # Base abs errors: [1, 1, 1], scaled by pixel_size^2 = 25
        expected = torch.tensor([25.0, 25.0, 25.0])

        assert torch.allclose(result, expected, atol=1e-5)
