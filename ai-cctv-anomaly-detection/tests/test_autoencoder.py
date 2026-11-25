"""Autoencoder tests (skipped automatically if PyTorch is not installed)."""
import pytest

torch = pytest.importorskip("torch")

from src.autoencoder import ConvAutoencoder, load_checkpoint, reconstruction_scores, save_checkpoint  # noqa: E402


@pytest.mark.parametrize("size,ch", [(64, 3), (128, 3), (64, 1)])
def test_forward_shapes(size, ch):
    m = ConvAutoencoder(ch, size, 8, 16)
    x = torch.rand(4, ch, size, size)
    xh, z = m(x)
    assert xh.shape == x.shape and z.shape == (4, 16)
    assert float(xh.min()) >= 0 and float(xh.max()) <= 1
    assert m.encode(x).shape == (4, 16)


def test_invalid_size_rejected():
    with pytest.raises(ValueError):
        ConvAutoencoder(3, 100)


def test_scores_and_methods():
    x = torch.zeros(2, 3, 32, 32)
    xh = x.clone()
    xh[1, :, :4, :4] = 1.0  # local error in sample 1 only
    for method in ("mse", "topk"):
        s = reconstruction_scores(x, xh, method, 0.05)
        assert s.shape == (2,) and s[0] == 0 and s[1] > 0
    assert reconstruction_scores(x, xh, "topk", 0.05)[1] > reconstruction_scores(x, xh, "mse")[1]
    with pytest.raises(ValueError):
        reconstruction_scores(x, xh, "nope")


def test_checkpoint_roundtrip_identical_outputs(tmp_path):
    m = ConvAutoencoder(3, 64, 8, 16).eval()
    p = save_checkpoint(m, tmp_path / "m.pt", {"k": 1})
    m2, meta = load_checkpoint(p)
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6) and meta == {"k": 1}


def test_missing_checkpoint_gives_helpful_error(tmp_path):
    from src.utils import ProjectError
    with pytest.raises(ProjectError, match="Train first"):
        load_checkpoint(tmp_path / "nope.pt")


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = ConvAutoencoder(3, 32, 8, 16)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    x = torch.rand(8, 3, 32, 32)
    first = None
    for _ in range(60):
        loss = torch.nn.functional.mse_loss(m(x)[0], x)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.7
