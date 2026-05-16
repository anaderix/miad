"""DMF-CSPNet: a one-shot (z -> (L, F)) generator built from MiAD's CSPNet.

Differences from MiAD's `lib/models/cspnet/cspnet.py`:
  * No time conditioning: `t, t_emb` removed from forward.
  * Latent noise `z_per_graph ∈ R^{latent_dim}` is concatenated with per-atom
    embeddings via `atom_latent_emb` (same Linear shape, repurposed).
  * Inputs `frac_coords_in` and `lattices_in` come from the noise prior
    (uniform on torus and Gaussian respectively); message-passing happens on
    these as a starting guess. Output is interpreted as ABSOLUTE (L̂, F̂):
    F̂ is mod-1'd to live on torus.

The architecture is byte-for-byte the same as MiAD's CSPNet otherwise — same
CSPLayer block, ip/ln/smooth/pred_type options — so a parameter-count and
capacity match is preserved.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.utils import dense_to_sparse

MAX_ATOMIC_NUM = 100


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 10, n_space: int = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class CSPLayer(nn.Module):
    """CSPLayer with optional FiLM composition conditioning."""

    def __init__(self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True, film=False):
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        # FiLM: per-graph (γ, β) modulation of node features from composition signal.
        self.film = film
        if film:
            # input: pooled composition embedding (hidden_dim), output: 2*hidden_dim for (γ, β)
            self.film_mlp = nn.Linear(hidden_dim, 2 * hidden_dim)

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        return self.edge_mlp(edges_input)

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter(edge_features, edge_index[0], dim=0, reduce="mean", dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        return self.node_mlp(agg)

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None, comp_emb=None, node2graph=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        # FiLM modulation: each atom is modulated by its crystal's composition embedding
        if self.film and comp_emb is not None:
            gamma_beta = self.film_mlp(comp_emb)  # (B, 2*hidden_dim)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            # broadcast per-atom via node2graph
            gamma_per_atom = gamma[node2graph]
            beta_per_atom = beta[node2graph]
            node_output = gamma_per_atom * node_output + beta_per_atom
        return node_input + node_output


class CSPNetDMF(nn.Module):
    """DMF generator: (z, atom_types, N) -> (lattice, frac)."""

    def __init__(
        self,
        hidden_dim=512,
        latent_dim=256,
        num_layers=6,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=128,
        edge_style="fc",
        cutoff=7.0,
        max_neighbors=20,
        ln=True,
        ip=True,
        smooth=True,
        pred_type=False,
        film=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.smooth = smooth
        self.ln = ln
        self.ip = ip
        self.edge_style = edge_style
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type

        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise NotImplementedError(act_fn)
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        else:
            raise NotImplementedError(dis_emb)

        self.num_layers = num_layers
        self.film = film
        for i in range(num_layers):
            self.add_module(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip, film=film),
            )
        if film:
            # composition embedding: pooled atom-type one-hot → hidden_dim
            self.comp_proj = nn.Linear(MAX_ATOMIC_NUM, hidden_dim)

        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    # ----- edge construction (full graph) -----
    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style != "fc":
            raise NotImplementedError(self.edge_style)
        lis = [torch.ones(int(n), int(n), device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        frac_diff = frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]
        return fc_edges, frac_diff

    def forward(self, z_graph, atom_types, frac_in, lattices_in, num_atoms, node2graph):
        """
        z_graph:    (B, latent_dim)        per-crystal noise.
        atom_types: (N_total,) or (N_total, max_atoms) if smooth=True (one-hot soft).
        frac_in:    (N_total, 3)           initial frac coords (e.g. uniform sample).
        lattices_in:(B, 3, 3)              initial lattice basis (e.g. Gaussian sample).
        num_atoms:  (B,)
        node2graph: (N_total,)
        Returns:    L_hat (B, 3, 3),  F_hat (N_total, 3)  (F_hat in [0,1)^3)
        """
        edges, frac_diff = self.gen_edges(num_atoms, frac_in, lattices_in, node2graph)
        edge2graph = node2graph[edges[0]]

        # node features from atomic types
        if self.smooth:
            # atom_types expected as one-hot or soft over max_atoms
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        # z input: either (B, latent_dim) per-graph or (N_total, latent_dim) per-atom
        if z_graph.shape[0] == num_atoms.shape[0]:
            # per-graph
            z_per_atom = z_graph.repeat_interleave(num_atoms, dim=0)
        else:
            # per-atom — assume already shaped (N_total, latent_dim)
            assert z_graph.shape[0] == int(num_atoms.sum()), \
                f"z shape {z_graph.shape[0]} doesn't match per-graph ({num_atoms.shape[0]}) or per-atom ({int(num_atoms.sum())})"
            z_per_atom = z_graph
        node_features = torch.cat([node_features, z_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        # FiLM: pool composition per crystal (mean-pool of one-hot atom_types)
        comp_emb = None
        if self.film:
            pooled = scatter(atom_types, node2graph, dim=0, reduce="mean")  # (B, MAX_ATOMIC_NUM)
            comp_emb = self.comp_proj(pooled)  # (B, hidden_dim)

        for i in range(self.num_layers):
            node_features = self._modules[f"csp_layer_{i}"](
                node_features, frac_in, lattices_in, edges, edge2graph,
                frac_diff=frac_diff, comp_emb=comp_emb, node2graph=node2graph,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        # Per-atom coord output: interpret as ABSOLUTE frac, mod 1.
        coord_out = self.coord_out(node_features)
        F_hat = coord_out % 1.0

        # Per-graph lattice output via scatter-mean readout.
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        L_hat = self.lattice_out(graph_features).view(-1, 3, 3)
        if self.ip:
            # Same trick as MiAD: contract with input lattice basis. This keeps
            # the network in the rotation-equivariant regime defined by ip=True.
            L_hat = torch.einsum("bij,bjk->bik", L_hat, lattices_in)

        if self.pred_type:
            type_out = self.type_out(node_features)
            return L_hat, F_hat, type_out
        return L_hat, F_hat
