"""
Minimal BatchTopK Sparse Autoencoder (inference + a place to load checkpoints).

Adapted from ``saes.py`` in goodfire-ai/sae-manifold (MIT; accompanies Bhalla
et al. 2026, "Do Sparse Autoencoders Capture Concept Manifolds?"). Trimmed to the
pieces this project needs: the module, a flexible checkpoint loader, and helpers
to read decoder atoms / encode activations. ``train.py`` trains this same module
so checkpoints round-trip.

A TopK SAE keeps the k largest post-ReLU features per sample:
    z = TopK_k( ReLU(W_enc x + b_enc) ),    x_hat = W_dec z + b_dec
The decoder rows are the "atoms" (feature directions) in activation space.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopKSAE(nn.Module):
    """Minimal (Batch)TopK SAE with per-sample top-k at inference.

    Args:
        d_in:  input (activation) dimension
        d_sae: number of SAE features (dictionary size)
        k:     features kept active per sample
    """

    def __init__(self, d_in, d_sae, k, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = int(k)
        self.encoder = nn.Linear(d_in, d_sae)
        self.decoder = nn.Linear(d_sae, d_in)
        # Optional JumpReLU-style per-feature threshold. If any entry > 0,
        # encode() thresholds instead of taking top-k.
        self.register_buffer("threshold", torch.zeros(d_sae))
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def encode(self, x):
        """Sparse codes ``[N, d_sae]`` for inputs ``x`` ``[N, d_in]``."""
        pre = F.relu(self.encoder(x))
        if torch.any(self.threshold > 0):
            return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))
        if self.k >= self.d_sae:
            return pre
        top = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, top.indices, top.values)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def load_sae(path, d_in=128, d_sae=None, k=None, expansion_factor=None,
             device=None, dtype=torch.float32):
    """Load a BatchTopK SAE checkpoint (bare state_dict, or a dict with
    ``state_dict`` + ``model_config``). Infers d_sae from the encoder weight if
    not given."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        cfg = ckpt.get("model_config", ckpt.get("config", {}))
        d_in = cfg.get("d_in", d_in)
        d_sae = cfg.get("d_sae", d_sae)
        k = cfg.get("k", cfg.get("target_l0", k))
        if d_sae is None and "expansion_factor" in cfg:
            expansion_factor = cfg["expansion_factor"]
    else:
        state_dict = ckpt

    if d_sae is None:
        if expansion_factor is None:
            for key in ("encoder.weight", "encoder_linear.weight", "W_enc"):
                if key in state_dict:
                    d_sae = state_dict[key].shape[0]
                    break
        else:
            d_sae = d_in * expansion_factor
    if d_sae is None:
        raise ValueError("Could not infer d_sae; pass d_sae or expansion_factor.")
    if k is None:
        raise ValueError("Could not infer k; pass k=...")

    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)
    renamed = {kk.replace("encoder_linear.", "encoder.")
                 .replace("decoder_linear.", "decoder."): v
               for kk, v in state_dict.items()}
    missing, unexpected = sae.load_state_dict(renamed, strict=False)
    if missing:
        print(f"load_sae: missing keys {missing}")
    if unexpected:
        print(f"load_sae: unexpected keys {unexpected}")
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    return sae


@torch.no_grad()
def encode_sae(sae, activations, device=None):
    """Encode ``[N, d_in]`` activations to ``[N, d_sae]`` codes (numpy)."""
    if device is None:
        device = next(sae.parameters()).device
    x = torch.as_tensor(activations).to(device=device,
                                        dtype=next(sae.parameters()).dtype)
    return sae.encode(x).detach().cpu().float().numpy()


def get_decoder(sae):
    """Decoder weight as ``[d_sae, d_in]`` numpy (atom i = row i)."""
    W = sae.decoder.weight.detach().float().cpu()  # nn.Linear stores [d_in, d_sae]
    return (W.numpy() if W.shape[0] == sae.d_sae else W.T.numpy())


def get_decoder_bias(sae):
    b = getattr(sae.decoder, "bias", None)
    return None if b is None else b.detach().float().cpu().numpy()
