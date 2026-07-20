import torch

from auto_margin_teacher_anchor import (
    BoundaryConstraint,
    factors_from_dual,
    solve_minimum_norm_dual,
)


def test_minimum_norm_dual_satisfies_boundaries():
    hidden = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.5, 1.0]])
    constraints = [
        BoundaryConstraint(0, 1, 2, 2.0, 0.0),
        BoundaryConstraint(1, 1, 3, 1.0, 0.0),
        BoundaryConstraint(2, 4, 2, 0.5, 0.0),
    ]
    alpha, meta = solve_minimum_norm_dual(constraints, hidden)
    patch, active = factors_from_dual(constraints, hidden, alpha, vocab_size=5)
    assert patch is not None
    a, b = patch
    delta = b.double() @ a.double()
    for item in constraints:
        change = (delta[item.teacher_id] - delta[item.competitor_id]) @ hidden[item.position].double()
        assert float(change) >= item.rhs - 1.0e-5
    assert active <= len(constraints)
    assert meta["kkt_residual"] < 1.0e-5


def test_empty_constraint_system():
    alpha, meta = solve_minimum_norm_dual([], torch.empty(0, 2))
    assert alpha.numel() == 0
    assert meta["kkt_residual"] == 0.0
