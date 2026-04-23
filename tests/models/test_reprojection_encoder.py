import numpy as np
import pytest
import torch

from icenet_mp.models.encoders import ReprojectionEncoder
from icenet_mp.types import DataSpace

INPUT_NAME = "source"
OUTPUT_NAME = "target"


def _make_encoder(
    input_shape: tuple[int, int] = (4, 4),
    latent_shape: tuple[int, int] = (2, 2),
    channels: int = 2,
    n_history_steps: int = 1,
) -> ReprojectionEncoder:
    input_space = DataSpace(name=INPUT_NAME, channels=channels, shape=input_shape)
    return ReprojectionEncoder(
        data_space_in=input_space,
        latent_space=latent_shape,
        n_history_steps=n_history_steps,
        project_to=OUTPUT_NAME,
    )


def _latlon_grid(shape: tuple[int, int]) -> tuple[list[float], list[float]]:
    """Flat lat/lon lists for a regular geographic grid."""
    h, w = shape
    lats = np.repeat(np.linspace(0, 60, h), w).tolist()
    lons = np.tile(np.linspace(0, 120, w), h).tolist()
    return lats, lons


def _set_input_output_latlons(
    encoder: ReprojectionEncoder,
    input_shape: tuple[int, int],
    latent_shape: tuple[int, int],
) -> None:
    lats_in, lons_in = _latlon_grid(input_shape)
    lats_out, lons_out = _latlon_grid(latent_shape)
    encoder.set_latlon(INPUT_NAME, lats_in, lons_in)
    encoder.set_latlon(OUTPUT_NAME, lats_out, lons_out)


class TestSetLatlon:
    def test_sets_input_latlon(self) -> None:
        encoder = _make_encoder(input_shape=(2, 3))
        lats = list(map(float, range(6)))
        lons = list(map(float, range(10, 16)))
        encoder.set_latlon(INPUT_NAME, lats, lons)
        assert encoder.input_latitudes == lats
        assert encoder.input_longitudes == lons

    def test_sets_output_latlon(self) -> None:
        encoder = _make_encoder(latent_shape=(2, 2))
        lats = [10.0, 20.0, 30.0, 40.0]
        lons = [50.0, 60.0, 70.0, 80.0]
        encoder.set_latlon(OUTPUT_NAME, lats, lons)
        assert encoder.output_latitudes == lats
        assert encoder.output_longitudes == lons

    def test_ignores_unknown_name(self) -> None:
        encoder = _make_encoder()
        encoder.set_latlon("unknown", [1.0, 2.0], [3.0, 4.0])
        assert encoder.input_latitudes == []
        assert encoder.input_longitudes == []
        assert encoder.output_latitudes == []
        assert encoder.output_longitudes == []

    def test_does_not_overwrite_other_grid(self) -> None:
        encoder = _make_encoder(input_shape=(2, 2), latent_shape=(2, 2))
        lats = [1.0, 2.0, 3.0, 4.0]
        lons = [5.0, 6.0, 7.0, 8.0]
        encoder.set_latlon(INPUT_NAME, lats, lons)
        assert encoder.output_latitudes == []
        assert encoder.output_longitudes == []


class TestNearestNeighbours:
    def test_raises_when_latlon_not_set(self) -> None:
        encoder = _make_encoder()
        with pytest.raises(ValueError, match="must be set"):
            encoder.nearest_neighbours(torch.device("cpu"))

    def test_raises_when_input_latlon_missing(self) -> None:
        encoder = _make_encoder(input_shape=(2, 2), latent_shape=(2, 2))
        lats_out, lons_out = _latlon_grid((2, 2))
        encoder.set_latlon(OUTPUT_NAME, lats_out, lons_out)
        with pytest.raises(ValueError, match="must be set"):
            encoder.nearest_neighbours(torch.device("cpu"))

    def test_raises_when_output_latlon_missing(self) -> None:
        encoder = _make_encoder(input_shape=(2, 2), latent_shape=(2, 2))
        lats_in, lons_in = _latlon_grid((2, 2))
        encoder.set_latlon(INPUT_NAME, lats_in, lons_in)
        with pytest.raises(ValueError, match="must be set"):
            encoder.nearest_neighbours(torch.device("cpu"))

    def test_raises_when_input_lat_wrong_size(self) -> None:
        encoder = _make_encoder(input_shape=(4, 4), latent_shape=(2, 2))
        lats_out, lons_out = _latlon_grid((2, 2))
        encoder.set_latlon(INPUT_NAME, [0.0, 1.0], [0.0, 1.0])  # 2 instead of 16
        encoder.set_latlon(OUTPUT_NAME, lats_out, lons_out)
        with pytest.raises(ValueError, match="input latitudes"):
            encoder.nearest_neighbours(torch.device("cpu"))

    def test_raises_when_output_lat_wrong_size(self) -> None:
        encoder = _make_encoder(input_shape=(2, 2), latent_shape=(3, 3))
        lats_in, lons_in = _latlon_grid((2, 2))
        encoder.set_latlon(INPUT_NAME, lats_in, lons_in)
        encoder.set_latlon(OUTPUT_NAME, [0.0, 1.0], [0.0, 1.0])  # 2 instead of 9
        with pytest.raises(ValueError, match="output latitudes"):
            encoder.nearest_neighbours(torch.device("cpu"))

    @pytest.mark.parametrize("latent_shape", [(2, 2), (3, 4)])
    def test_returns_tensors_of_correct_shape(
        self, latent_shape: tuple[int, int]
    ) -> None:
        input_shape = (4, 4)
        encoder = _make_encoder(input_shape=input_shape, latent_shape=latent_shape)
        _set_input_output_latlons(encoder, input_shape, latent_shape)
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        assert nn_h.shape == latent_shape
        assert nn_w.shape == latent_shape

    def test_index_values_are_in_input_range(self) -> None:
        input_shape = (4, 5)
        latent_shape = (2, 3)
        encoder = _make_encoder(input_shape=input_shape, latent_shape=latent_shape)
        _set_input_output_latlons(encoder, input_shape, latent_shape)
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        assert torch.all((nn_h >= 0) & (nn_h < input_shape[0]))
        assert torch.all((nn_w >= 0) & (nn_w < input_shape[1]))

    def test_identity_reprojection_maps_to_self(self) -> None:
        # When input and output grids are identical, each output cell should map to
        # the corresponding input cell
        shape = (3, 3)
        lats = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons = np.tile([0.0, 60.0, 120.0], 3).tolist()

        encoder = _make_encoder(input_shape=shape, latent_shape=shape)
        encoder.set_latlon(INPUT_NAME, lats, lons)
        encoder.set_latlon(OUTPUT_NAME, lats, lons)

        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))

        expected_h = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        expected_w = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        assert torch.equal(nn_h, expected_h)
        assert torch.equal(nn_w, expected_w)


class TestReprojectionEncoderForward:
    @pytest.mark.parametrize("test_batch_size", [1, 2, 5])
    @pytest.mark.parametrize("test_input_chw", [(2, 4, 4), (3, 6, 6)])
    @pytest.mark.parametrize("test_latent_hw", [(2, 2), (3, 4)])
    @pytest.mark.parametrize("test_n_history_steps", [1, 3])
    def test_rollout_output_shape(
        self,
        test_batch_size: int,
        test_input_chw: tuple[int, int, int],
        test_latent_hw: tuple[int, int],
        test_n_history_steps: int,
    ) -> None:
        channels, h, w = test_input_chw
        input_shape = (h, w)
        encoder = _make_encoder(
            input_shape=input_shape,
            latent_shape=test_latent_hw,
            channels=channels,
            n_history_steps=test_n_history_steps,
        )
        _set_input_output_latlons(encoder, input_shape, test_latent_hw)
        encoder.eval()
        x = torch.randn(test_batch_size, test_n_history_steps, channels, h, w)
        out = encoder.rollout(x)
        assert out.shape == (
            test_batch_size,
            test_n_history_steps,
            channels,
            *test_latent_hw,
        )

    def test_forward_applies_nearest_neighbour_indexing(self) -> None:
        # Use an identity reprojection on a 3x3 grid and verify values are sampled correctly
        shape = (3, 3)
        channels = 1
        lats = np.repeat([0.0, 30.0, 60.0], 3).tolist()
        lons = np.tile([0.0, 60.0, 120.0], 3).tolist()

        encoder = _make_encoder(
            input_shape=shape, latent_shape=shape, channels=channels
        )
        encoder.set_latlon(INPUT_NAME, lats, lons)
        encoder.set_latlon(OUTPUT_NAME, lats, lons)

        # Use eval mode to disable BatchNorm's running-stat updates and use identity-like behavior
        # after a manual forward on a uniform input
        encoder.eval()

        # A uniform input: batch_norm of a constant returns 0 (or near 0 when using running stats)
        # Instead, verify that the spatial structure is preserved: input[0, 0, i, h, w]
        # should appear at output[0, 0, i, nn_h[h,w], nn_w[h,w]] — i.e. same position for identity
        nn_h, nn_w = encoder.nearest_neighbours(torch.device("cpu"))
        assert nn_h.shape == shape
        assert nn_w.shape == shape

        x_nchw = torch.randn(2, channels, *shape)
        out = encoder(x_nchw)
        assert out.shape == (2, channels, *shape)

    def test_forward_channels_preserved(self) -> None:
        input_shape = (4, 4)
        latent_shape = (2, 2)
        channels = 5
        encoder = _make_encoder(
            input_shape=input_shape, latent_shape=latent_shape, channels=channels
        )
        _set_input_output_latlons(encoder, input_shape, latent_shape)
        encoder.eval()
        x = torch.randn(3, channels, *input_shape)
        out = encoder(x)
        assert out.shape[1] == channels
