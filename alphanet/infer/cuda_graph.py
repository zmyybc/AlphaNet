"""Whole-step CUDA Graph capture for AlphaNet ASE inference.

A replayed step executes the entire edge-geometry -> forward -> autograd
backward kernel sequence with a single graph launch, eliminating per-kernel
launch latency and eager-autograd Python overhead (the dominant cost for
small/medium systems).

Design points:
- Static shapes come from the masked edge mode (``static_shapes=True`` in
  ``graph_from_neighbor_topology``): every cached edge (cutoff+skin) is kept
  and out-of-cutoff edges are zeroed by ``edge_mask`` — mathematically
  equivalent to hard filtering.
- The edge buffers are over-allocated (``capacity_factor``) and padded with
  self-edges carrying a huge cell offset, so their distance is far beyond the
  cutoff: they are masked to exactly zero and produce no NaNs. A neighbor-list
  rebuild therefore only refreshes the static buffers (the captured kernels
  read the new indices on the next replay) — no recapture. Recapture happens
  only when the edge count outgrows the capacity, the atom count changes, or
  the stress mode changes.
- ``torch.autograd.grad`` is captured inside the graph (official PyTorch
  pattern: warmup iterations on a side stream, then capture); forces/stress
  land in static output tensors.
"""
import numpy as np
import torch

from alphanet.models.graph import NeighborTopology, graph_from_neighbor_topology

# Padding edges are (0, 0) self-pairs shifted by this many cells along the
# first lattice vector: for any physically meaningful cell (|a1| > 0.5 A)
# the padded distance is far beyond cutoff + skin, so the edge mask zeroes
# them exactly and every edge quantity stays finite.
_PAD_SHIFT = 16


class CapturedStep:
    """One CUDA graph per (topology window capacity, stress mode)."""

    def __init__(self, model, topology, z, natoms, batch, n_atoms, cutoff,
                 skin, precision, device, with_stress, pool=None,
                 capacity_factor=1.1, warmup=3, positions=None, cell=None):
        self.model = model
        self.with_stress = with_stress
        self.n_atoms = n_atoms
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.precision = precision
        self.device = device
        self.topology_ref = None

        n_edges = int(topology.edge_index.shape[1])
        self.capacity = max(int(np.ceil(n_edges * capacity_factor)), n_edges + 8)

        # --- static buffers (inputs the captured kernels read) ---
        self.pos_in = torch.zeros(n_atoms, 3, dtype=precision, device=device,
                                  requires_grad=True)
        self.cell_in = torch.zeros(3, 3, dtype=precision, device=device)
        self._edge_index = torch.zeros(2, self.capacity, dtype=torch.long,
                                       device=device)
        self._cell_offsets = torch.zeros(self.capacity, 3, dtype=torch.int32,
                                         device=device)
        self._z = z.detach().clone()
        self._natoms = natoms.detach().clone()
        self._batch = batch.detach().clone()
        self._pos_host = torch.empty(n_atoms, 3, dtype=precision).pin_memory()
        self._cell_host = torch.empty(3, 3, dtype=precision).pin_memory()

        self._static_topology = NeighborTopology(
            edge_index=self._edge_index,
            cell_offsets=self._cell_offsets,
            neighbors=topology.neighbors,
            reference_positions=topology.reference_positions,
            reference_cell=topology.reference_cell,
            cutoff=self.cutoff,
            skin=self.skin,
        )
        self.update_topology(topology)
        # Preload real inputs so warmup/capture run on physical values
        # (kernels recorded are value-independent, but this keeps warmup free
        # of NaN propagation from all-zero positions and makes any subsequent
        # timing representative).
        if positions is not None and cell is not None:
            self._load_inputs(positions, cell)

        def _run():
            with torch.enable_grad():
                data = graph_from_neighbor_topology(
                    pos=self.pos_in, z=self._z, natoms=self._natoms,
                    batch=self._batch, topology=self._static_topology,
                    cell=self.cell_in, cutoff=self.cutoff, dtype=self.precision,
                    compute_stress=self.with_stress, static_shapes=True)
                return self.model.forward_graph(
                    data, prefix="infer", compute_forces=True,
                    compute_stress=self.with_stress)

        # warmup on a side stream: cuBLAS workspaces, allocator, autograd
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                _run()
        torch.cuda.current_stream(device).wait_stream(side)
        torch.cuda.synchronize(device)
        # Release the warmup's cached blocks back to the driver: the capture
        # pool cannot reuse the regular allocator's cache, and on large
        # systems the leftover reservation fragments the device enough to
        # OOM the capture itself.
        torch.cuda.empty_cache()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.energy, self.forces, self.stress = _run()
        self.pool = self.graph.pool()

    def update_topology(self, topology) -> bool:
        """Refresh the static edge buffers after a neighbor-list rebuild.

        Returns False when the new edge count exceeds the captured capacity
        (caller must recapture). No recapture is needed otherwise: the
        captured gather/scatter kernels read these buffers at replay time.
        """
        n_edges = int(topology.edge_index.shape[1])
        if n_edges > self.capacity or int(topology.edge_index.max().item() if n_edges else 0) >= self.n_atoms:
            return False
        with torch.no_grad():
            self._edge_index[:, :n_edges] = topology.edge_index
            self._edge_index[:, n_edges:] = 0
            self._cell_offsets[:n_edges] = topology.cell_offsets
            self._cell_offsets[n_edges:] = 0
            self._cell_offsets[n_edges:, 0] = _PAD_SHIFT
        self.topology_ref = topology
        return True

    def refresh_z(self, z: torch.Tensor) -> None:
        """Refresh the static atomic-number buffer read by the captured
        kernels: species can change while the atom count stays the same
        (composition screening, MC swaps), which rebuilds the topology but
        does not trigger a recapture."""
        with torch.no_grad():
            self._z.copy_(z)

    def _load_inputs(self, positions: np.ndarray, cell: np.ndarray):
        self._pos_host.copy_(torch.from_numpy(np.ascontiguousarray(positions)))
        self._cell_host.copy_(torch.from_numpy(np.ascontiguousarray(cell)))
        with torch.no_grad():
            self.pos_in.copy_(self._pos_host, non_blocking=True)
            self.cell_in.copy_(self._cell_host, non_blocking=True)

    def run(self, positions: np.ndarray, cell: np.ndarray):
        """Copy inputs into the static buffers, replay, return static outputs.

        The returned tensors are overwritten by the next replay — consume
        (``.cpu()``/``.item()``) before calling ``run`` again.
        """
        self._load_inputs(positions, cell)
        self.graph.replay()
        return self.energy, self.forces, self.stress
