"""Quick GPU verification that chunked cross-entropy matches the original."""
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT, GPTConfig


def build_model(cfg, device):
    torch.manual_seed(0)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    return model


def run():
    device = "cuda"

    cfg = GPTConfig(
        sequence_len=512,
        vocab_size=32768,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
        window_pattern="SSSL",
    )

    B, T, V = 2, 512, cfg.vocab_size

    # Build two identical models
    model = build_model(cfg, device)
    model_ref = build_model(cfg, device)

    # Verify weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_ref.named_parameters()):
        assert torch.equal(p1, p2), f"Mismatch at {n1}"

    torch.manual_seed(42)
    idx = torch.randint(0, V, (B, T), device=device)
    targets = torch.roll(idx, -1, dims=1)
    # Insert some -1 entries to test ignore_index
    targets[0, 0] = -1
    targets[1, 5:10] = -1
    targets[0, 100] = -1

    # Compute reference logits (softcapped fp32) from reference model (with grad for backward checks)
    logits_ref = model_ref(idx, targets=None)  # returns softcapped fp32 logits

    results = {}

    # --- Mean reduction ---
    model.zero_grad()
    model_ref.zero_grad()
    loss_chunked = model(idx, targets=targets, loss_reduction='mean')
    loss_ref = F.cross_entropy(logits_ref.view(-1, V), targets.view(-1), ignore_index=-1, reduction='mean')
    mean_diff = float((loss_chunked - loss_ref).abs().max().detach())
    mean_match = torch.allclose(loss_chunked.detach(), loss_ref.detach(), atol=1e-6, rtol=1e-6)
    results['mean'] = (mean_match, mean_diff)

    # Gradients
    loss_chunked.backward()
    loss_ref.backward()
    grad_lh_c = model.lm_head.weight.grad.clone()
    grad_lh_r = model_ref.lm_head.weight.grad.clone()
    grad_lh_diff = float((grad_lh_c - grad_lh_r).abs().max())
    grad_lh_match = torch.allclose(grad_lh_c, grad_lh_r, atol=1e-4, rtol=1e-3)
    results['lm_head_grad'] = (grad_lh_match, grad_lh_diff)

    # Check a deeper layer grad too
    grad_deep_c = model.transformer.h[0].attn.c_proj.weight.grad.clone()
    grad_deep_r = model_ref.transformer.h[0].attn.c_proj.weight.grad.clone()
    grad_deep_diff = float((grad_deep_c - grad_deep_r).abs().max())
    grad_deep_match = torch.allclose(grad_deep_c, grad_deep_r, atol=1e-4, rtol=1e-3)
    results['deep_grad'] = (grad_deep_match, grad_deep_diff)

    # --- Sum reduction ---
    model.zero_grad()
    model_ref.zero_grad()
    loss_sum = model(idx, targets=targets, loss_reduction='sum')
    loss_sum_ref = F.cross_entropy(logits_ref.view(-1, V), targets.view(-1), ignore_index=-1, reduction='sum')
    sum_diff = float((loss_sum.detach() - loss_sum_ref.detach()).abs().max())
    sum_match = torch.allclose(loss_sum.detach(), loss_sum_ref.detach(), atol=1e-4, rtol=1e-4)
    results['sum'] = (sum_match, sum_diff)

    # --- None reduction ---
    model.zero_grad()
    model_ref.zero_grad()
    loss_none = model(idx, targets=targets, loss_reduction='none')
    loss_none_ref = F.cross_entropy(logits_ref.view(-1, V), targets.view(-1), ignore_index=-1, reduction='none')
    none_diff = float((loss_none.detach() - loss_none_ref.detach()).abs().max())
    none_match = torch.allclose(loss_none.detach(), loss_none_ref.detach(), atol=1e-6, rtol=1e-6)
    results['none'] = (none_match, none_diff)
    none_shape_ok = loss_none.shape == (B * T,)
    results['none_shape'] = (none_shape_ok, str(loss_none.shape))

    # --- Print results ---
    all_pass = True
    for key, (ok, detail) in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {key}: {status}  (detail={detail})")

    print()
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")


if __name__ == "__main__":
    run()
