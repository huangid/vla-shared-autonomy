"""Train the residual correction network on cached features (no Isaac needed).

    python scripts/residual_train.py --data logs/data/tc_residual.npz \
        --output outputs/residual/tc_residual_v1.pt

Held-out EPISODES (not steps — neighbouring steps of one episode are near-duplicates)
measure whether the network learned anything beyond "output nothing", which is the
trivial solution when 80% of the labels are zero. The number to read is the error on
held-out corrective steps against the error of predicting zero.

The epoch count is chosen on those held-out episodes; the saved model is then retrained
on every episode for that many epochs, since 100 episodes is too few to leave 20 unused.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "pilot_models"))
from residual_policy import ResidualNet, low_dim_inputs  # noqa: E402


def fit_norm(net, fw, fs, low, y, corrective):
    """Normalisation statistics from the training split only."""
    f = net.pack_feats(fw, fs)
    if f.shape[1]:                       # a state-only network has no image features
        net.feat_mean.copy_(f.mean(0))
        net.feat_std.copy_(f.std(0).clamp_min(1e-3))
    net.low_mean.copy_(low.mean(0))
    net.low_std.copy_(low.std(0).clamp_min(1e-4))
    # Scale targets by the spread of the non-zero labels so the loss is O(1).
    net.out_scale.fill_(float(y[corrective].std().clamp_min(1e-4)))


def loss_fn(pred, y_scaled, w):
    per = F.smooth_l1_loss(pred, y_scaled, reduction="none").sum(-1)
    return (per * w).sum() / w.sum()


@torch.no_grad()
def metrics(net, fw, fs, low, y, corrective, w):
    net.eval()
    pred = net(fw, fs, low)
    out = {"loss": float(loss_fn(pred, y / net.out_scale, w))}
    pm = pred * net.out_scale                                  # metres
    c = corrective
    err = torch.linalg.vector_norm(pm[c] - y[c], dim=-1)
    zero = torch.linalg.vector_norm(y[c], dim=-1)
    out["corr_err_cm"] = float(err.mean() * 100)
    out["corr_zero_err_cm"] = float(zero.mean() * 100)          # error of predicting nothing
    out["corr_cos"] = float(F.cosine_similarity(pm[c], y[c], dim=-1).mean())
    out["corr_pred_norm_cm"] = float(torch.linalg.vector_norm(pm[c], dim=-1).mean() * 100)
    out["idle_pred_norm_cm"] = float(torch.linalg.vector_norm(pm[~c], dim=-1).mean() * 100)
    return out


def train(net, tr, epochs, lr, wd, batch, seed, val=None, log_every=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    fw, fs, low, y, c, w = tr
    ys = y / net.out_scale
    n, history = len(y), []
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g).to(y.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(net(fw[idx], fs[idx], low[idx]), ys[idx], w[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val is not None:
            m = metrics(net, *val)
            history.append(m)
            if log_every and (ep % log_every == 0 or ep == 1):
                print(f"  epoch {ep:3d}  val loss {m['loss']:.4f} | corrective err {m['corr_err_cm']:.2f} cm "
                      f"(zero {m['corr_zero_err_cm']:.2f}) cos {m['corr_cos']:+.2f} | "
                      f"idle |pred| {m['idle_pred_norm_cm']:.2f} cm", flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="logs/data/tc_residual.npz")
    p.add_argument("--output", default="outputs/residual/tc_residual_v1.pt")
    p.add_argument("--views", default="both", choices=["both", "wrist", "side", "none"],
                   help="Camera features to use. 'none' = robot state + base action only.")
    p.add_argument("--pool", type=int, default=1, choices=[1, 2],
                   help="2 = 2x2 feature cells per view (keeps coarse position), 1 = one vector per view.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--corr_weight", type=float, default=5.0,
                   help="Loss weight on corrective steps; the other ~80% of steps have a zero label.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--val_every", type=int, default=5, help="Every Nth episode is held out.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_final_all", action="store_true",
                   help="Save the model trained on the training split only.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.data, allow_pickle=True)
    dev = args.device
    fw = torch.from_numpy(d["feat_wrist"]).to(dev).float()
    fs = torch.from_numpy(d["feat_side"]).to(dev).float()
    state = torch.from_numpy(d["state"]).to(dev)
    a_r = torch.from_numpy(d["base_action"]).to(dev)
    low = low_dim_inputs(state, a_r)
    y = torch.from_numpy(d["resid"]).to(dev)
    c = torch.from_numpy(d["corrective"]).to(dev)
    w = torch.where(c, torch.full_like(y[:, 0], args.corr_weight), torch.ones_like(y[:, 0]))
    ep = torch.from_numpy(d["episode"]).to(dev)

    is_val = (ep % args.val_every) == (args.val_every - 1)
    pick = lambda m: (fw[m], fs[m], low[m], y[m], c[m], w[m])  # noqa: E731
    tr, va = pick(~is_val), pick(is_val)
    n_ep = int(ep.max()) + 1
    print(f"{n_ep} episodes | train {int((~is_val).sum())} steps ({int(c[~is_val].sum())} corrective) | "
          f"held out {int(is_val.sum())} steps ({int(c[is_val].sum())} corrective) from "
          f"{len(torch.unique(ep[is_val]))} episodes")

    def new_net():
        torch.manual_seed(args.seed)
        return ResidualNet(views=args.views, pool=args.pool, hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout).to(dev)

    net = new_net()
    fit_norm(net, tr[0], tr[1], tr[2], tr[3], tr[4])
    hist = train(net, tr, args.epochs, args.lr, args.wd, args.batch, args.seed, val=va, log_every=10)
    best = int(np.argmin([h["loss"] for h in hist]))
    m = hist[best]
    gain = 100 * (1 - m["corr_err_cm"] / m["corr_zero_err_cm"])
    print(f"\nbest epoch {best + 1} of {args.epochs} on held-out episodes:")
    print(f"  corrective steps : error {m['corr_err_cm']:.2f} cm vs {m['corr_zero_err_cm']:.2f} cm for "
          f"predicting nothing ({gain:+.0f}%), direction cosine {m['corr_cos']:+.2f}")
    print(f"  size of output   : {m['corr_pred_norm_cm']:.2f} cm on corrective steps, "
          f"{m['idle_pred_norm_cm']:.2f} cm on uncorrected ones")

    if args.no_final_all:
        net = new_net()
        fit_norm(net, tr[0], tr[1], tr[2], tr[3], tr[4])
        train(net, tr, best + 1, args.lr, args.wd, args.batch, args.seed)
        trained_on = "training split"
    else:
        net = new_net()
        allm = torch.ones_like(is_val)
        full = pick(allm)
        fit_norm(net, full[0], full[1], full[2], full[3], full[4])
        train(net, full, best + 1, args.lr, args.wd, args.batch, args.seed)
        trained_on = "all episodes"
    net.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    config = {"views": args.views, "pool": args.pool, "hidden": args.hidden, "layers": args.layers,
              "corr_weight": args.corr_weight, "epochs": best + 1, "lr": args.lr, "wd": args.wd,
              "dropout": args.dropout, "seed": args.seed, "data": args.data,
              "trained_on": trained_on, "heldout": m, "heldout_gain_pct": gain}
    torch.save({"state_dict": net.state_dict(), "config": config}, args.output)
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved {args.output}  ({best + 1} epochs on {trained_on})")


if __name__ == "__main__":
    main()
