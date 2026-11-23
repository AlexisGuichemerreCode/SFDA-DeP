"""
Numerical checks for the ESFDA forget-loss probability redistribution
(dlib/losses/unlearning_sfda.py). Run:
    python -m pytest tests/test_forget_redistribution.py -q
or  python tests/test_forget_redistribution.py
"""
import sys
from os.path import dirname, abspath

import torch

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from dlib.losses.unlearning_sfda import (  # noqa: E402
    ForgetClassSuppressionLoss,
    redistribution_forget_target,
    redistribution_forget_loss,
)

KS = (2, 3, 5)  # binary and multi-class
ALPHAS = (0.0, 0.2, 0.5, 0.8, 1.0)


def _batch(k, b=16, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, k, generator=g, dtype=torch.float64) * scale
    d = torch.randint(0, k, (b,), generator=g)
    return logits, d


def _cases():
    for k in KS:
        yield k, 'uniform', 1.0
        for a in ALPHAS:
            yield k, 'hybrid', a


def test_target_zero_on_forgotten_class_and_sums_to_one():
    for k, mode, a in _cases():
        logits, d = _batch(k)
        q = redistribution_forget_target(logits, d, mode, a)
        assert torch.all(q.gather(1, d[:, None]) == 0), (k, mode, a)
        assert torch.allclose(q.sum(1), torch.ones(len(d), dtype=q.dtype)), \
            (k, mode, a)
        assert torch.all(q >= 0), (k, mode, a)


def test_binary_puts_all_mass_on_other_class():
    logits, d = _batch(2)
    for mode, a in (('uniform', 1.0), ('hybrid', 0.0), ('hybrid', 0.5)):
        q = redistribution_forget_target(logits, d, mode, a)
        assert torch.allclose(q.gather(1, (1 - d)[:, None]).squeeze(1),
                              torch.ones(len(d), dtype=q.dtype))


def _grad(fn, logits, d):
    z = logits.clone().requires_grad_(True)
    fn(z, d).mean().backward()
    return z.grad


def test_hybrid_alpha0_matches_original_forget_gradient():
    # original forgetting: -log(1 - p_d)
    base = ForgetClassSuppressionLoss(eps=0.0)
    for k in KS:
        logits, d = _batch(k, scale=1.0)
        g_base = _grad(base, logits, d)
        g_h0 = _grad(lambda z, c: redistribution_forget_loss(
            z, c, 'hybrid', 0.0).double(), logits, d)
        assert torch.allclose(g_base, g_h0, atol=1e-10), k


def test_saturated_forget_class_keeps_valid_target_and_gradient():
    # p_d rounds to 1 in float32: q must still sum to 1 and the push on
    # z_d must stay ~ p_d ~ 1 (no vanishing gradient).
    for k in KS:
        for gap in (20.0, 50.0, 100.0):
            z = torch.zeros(1, k)
            z[0, 0] = gap
            d = torch.tensor([0])
            for mode, a in (('uniform', 1.0), ('hybrid', 0.0),
                            ('hybrid', 0.5)):
                q = redistribution_forget_target(z, d, mode, a)
                assert torch.allclose(q.sum(1), torch.ones(1)), (k, gap, mode)
                g = _grad(lambda x, c: redistribution_forget_loss(
                    x, c, mode, a), z, d)
                assert g[0, 0] > 0.99, (k, gap, mode, a, g)


def test_hybrid_alpha1_equals_uniform():
    for k in KS:
        logits, d = _batch(k)
        l_u = redistribution_forget_loss(logits, d, 'uniform')
        l_h1 = redistribution_forget_loss(logits, d, 'hybrid', 1.0)
        assert torch.allclose(l_u, l_h1), k
        assert torch.allclose(
            _grad(lambda z, c: redistribution_forget_loss(z, c, 'uniform'),
                  logits, d),
            _grad(lambda z, c: redistribution_forget_loss(z, c, 'hybrid', 1.),
                  logits, d)), k


def test_backward_finite_including_saturated_logits():
    for k, mode, a in _cases():
        for scale in (1.0, 50.0):  # 50 -> p_d ~ 1 for some samples
            logits, d = _batch(k, scale=scale)
            # force the forgotten class to be saturated in half the batch
            logits[::2].scatter_(1, d[::2, None], 100.0)
            for dtype in (torch.float32, torch.float16):
                z = logits.to(dtype).clone().requires_grad_(True)
                loss = redistribution_forget_loss(z, d, mode, a).mean()
                loss.backward()
                assert torch.isfinite(loss), (k, mode, a, scale, dtype)
                assert torch.isfinite(z.grad).all(), (k, mode, a, scale,
                                                       dtype)


def test_loss_lowers_p_forgotten_class():
    for k, mode, a in _cases():
        logits, d = _batch(k, scale=1.0)
        z = logits.clone().requires_grad_(True)
        opt = torch.optim.SGD([z], lr=0.5)
        p0 = z.softmax(1).gather(1, d[:, None]).mean().item()
        for _ in range(20):
            opt.zero_grad()
            redistribution_forget_loss(z, d, mode, a).mean().backward()
            opt.step()
        p1 = z.softmax(1).gather(1, d[:, None]).mean().item()
        assert p1 < p0, (k, mode, a, p0, p1)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'ok  {name}')
