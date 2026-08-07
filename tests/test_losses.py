import torch

from cellvision.models.losses import masked_cross_entropy


def test_non_cell_viability_loss_can_be_masked():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    targets = torch.tensor([0, 1])
    valid = torch.tensor([1, 0])
    loss = masked_cross_entropy(logits, targets, valid)
    expected = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
    assert torch.allclose(loss, expected)

