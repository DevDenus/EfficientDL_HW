from init import ROOT, seed_all

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from cut_cross_entropy.cce_utils import LinearCrossEntropyImpl
from transformers import AutoModelForCausalLM

torch.cuda.set_per_process_memory_fraction(0.95, 0)

OOM_EXIT_CODE = 42
IMPLEMENTATIONS = ("ce",) + tuple(
    impl.name.lower() for impl in LinearCrossEntropyImpl
)


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / ".cache" / "model", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    ).cuda().train()
    model.config.use_cache = False
    return model


def make_forward(model, impl):
    def backbone(batch):
        return model.model(batch, use_cache=False).last_hidden_state

    def criterion(hidden, weight, batch):
        if impl != "ce":
            return linear_cross_entropy(hidden, weight, batch, shift=1, impl=impl)
        logits = F.linear(hidden[:, :-1], weight)
        return F.cross_entropy(logits.float().reshape(-1, weight.shape[0]),
                               batch[:, 1:].reshape(-1))

    def forward(batch):
        return criterion(backbone(batch), model.get_output_embeddings().weight, batch)

    return forward


def gradient_error(reference, candidate):
    reference, candidate = reference.float().flatten(), candidate.float().flatten()
    diff = candidate - reference
    return {
        "max_abs": diff.abs().max().item(),
        "relative_l2": (diff.norm() / reference.norm().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(reference, candidate, dim=0).item(),
    }


def check_accuracy(model, forward, batch, seed):
    weight = model.get_output_embeddings().weight
    values, gradients = [], []
    for index, compute in enumerate((make_forward(model, "ce"), forward)):
        seed_all(seed)
        model.zero_grad(set_to_none=True)
        try:
            loss = compute(batch)
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            if index != 0:
                raise
            model.zero_grad(set_to_none=True)
            print("CE accuracy reference OOM: loss/gradient comparison unavailable.", flush=True)
            return dict.fromkeys(("loss_abs_error", "loss_relative_error",
                                  "head_grad_max_abs", "head_grad_relative_l2", "head_grad_cosine"))
        values.append(loss.item())
        gradients.append(weight.grad.detach().cpu())
        del loss
    model.zero_grad(set_to_none=True)
    error = abs(values[1] - values[0])
    print(f"ce_loss={values[0]:.6f}, impl_loss={values[1]:.6f}", flush=True)
    return {
        "loss_abs_error": error,
        "loss_relative_error": error / max(abs(values[0]), 1e-12),
        **{f"head_grad_{key}": value for key, value in gradient_error(*gradients).items()},
    }


def weight_error(model, reference):
    if reference is None:
        return dict.fromkeys(("weights_max_abs", "weights_relative_l2", "weights_cosine"))
    error_sq = reference_sq = candidate_sq = dot = max_abs = 0.0
    for name, candidate in model.named_parameters():
        candidate, ref = candidate.detach().cpu().float(), reference[name].float()
        diff = candidate - ref
        max_abs = max(max_abs, diff.abs().max().item())
        error_sq += diff.square().sum(dtype=torch.float64).item()
        reference_sq += ref.square().sum(dtype=torch.float64).item()
        candidate_sq += candidate.square().sum(dtype=torch.float64).item()
        dot += (candidate * ref).sum(dtype=torch.float64).item()
    return {
        "weights_max_abs": max_abs,
        "weights_relative_l2": (error_sq / max(reference_sq, 1e-30)) ** 0.5,
        "weights_cosine": dot / max((reference_sq * candidate_sq) ** 0.5, 1e-30),
    }


def train_steps(forward, batches, optimizer, measure=False):
    forward_ms, backward_ms = [], []
    if measure:
        start, forward_done, backward_done = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    for batch in batches:
        batch = batch.cuda()
        optimizer.zero_grad(set_to_none=True)
        if measure:
            torch.cuda.synchronize()
            start.record()
        loss = forward(batch)
        if measure:
            forward_done.record()
        loss.backward()
        if measure:
            backward_done.record()
            backward_done.synchronize()
            forward_ms.append(start.elapsed_time(forward_done))
            backward_ms.append(forward_done.elapsed_time(backward_done))
        optimizer.step()
        del loss, batch
    if measure:
        torch.cuda.synchronize()
        return {
            "forward_mean_ms": statistics.mean(forward_ms),
            "forward_median_ms": statistics.median(forward_ms),
            "backward_mean_ms": statistics.mean(backward_ms),
            "backward_median_ms": statistics.median(backward_ms),
            "peak_train_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_train_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }


def reference_weights(batches, args):
    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    try:
        seed_all(args.seed)
        train_steps(make_forward(model, "ce"), batches, optimizer)
        return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()}
    finally:
        del optimizer, model


def clear_cache():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def write_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=IMPLEMENTATIONS, default="cce")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--csv", type=Path, default=ROOT / "results.csv")
    args = parser.parse_args()

    seed_all(args.seed)
    tokens = torch.load(ROOT / ".cache" / "tokens.pt", weights_only=True)
    blocks = tokens[:tokens.numel() // args.seq_len * args.seq_len].view(-1, args.seq_len)
    assert len(blocks) >= args.steps * args.batch_size
    order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(args.seed))
    batches = blocks[order[:args.steps * args.batch_size]].view(args.steps, args.batch_size, args.seq_len)

    print("Phase: CE reference training", flush=True)
    try:
        reference = reference_weights(batches, args)
    except torch.cuda.OutOfMemoryError:
        reference = None
        print("CE training reference OOM: weight comparison unavailable.", flush=True)
    clear_cache()

    print(f"Phase: {args.impl} accuracy", flush=True)
    model = load_model()
    forward = make_forward(model, args.impl)
    accuracy = check_accuracy(model, forward, batches[0].cuda(), args.seed)
    print(json.dumps(accuracy, indent=2), flush=True)
    clear_cache()

    print(f"Phase: {args.impl} training for weight comparison", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    seed_all(args.seed)
    train_steps(forward, batches, optimizer)
    optimizer.zero_grad(set_to_none=True)

    weights = weight_error(model, reference)
    del reference
    clear_cache()

    print(f"Phase: {args.impl} performance measurement", flush=True)
    seed_all(args.seed)
    performance = train_steps(forward, batches, optimizer, measure=True)
    row = {
        "impl": args.impl, "seed": args.seed, "steps": args.steps,
        "batch_size": args.batch_size, "seq_len": args.seq_len, "lr": args.lr,
        "vocab_size": model.config.vocab_size,
        **accuracy, **weights, **performance,
    }
    write_csv(args.csv, row)
    print(f"Saved {args.impl}: {args.csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as error:
        print(f"CUDA OOM: implementation skipped; no CSV row written.\n{error}", flush=True)
        raise SystemExit(OOM_EXIT_CODE)
