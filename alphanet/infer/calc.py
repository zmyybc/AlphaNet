import gc
import warnings

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from alphanet.infer.cuda_graph import CapturedStep
from alphanet.models.graph import build_neighbor_topology, graph_from_neighbor_topology
from alphanet.models.model import AlphaNetWrapper

torch.set_float32_matmul_precision('high')

class AlphaNetCalculator(Calculator):
    """
    ASE Calculator for AlphaNet models.

    This calculator wraps an AlphaNet model to perform energy, force, and stress
    calculations within the ASE framework. It automatically handles both periodic
    and non-periodic systems. For non-periodic systems, it creates a large
    supercell with vacuum padding to simulate an isolated molecule.
    """
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    def __init__(
        self,
        ckpt_path,
        config,
        device='cpu',
        precision='32',
        reuse_neighbors=True,
        skin=0.5,
        use_cuda_graph=None,
        sticky_stress=True,
        use_fused_ops=None,
        **kwargs,
    ):
        """
        Initializes the AlphaNetCalculator.

        Args:
            ckpt_path (str): Path to the model checkpoint file (.ckpt or .pt).
            config (object): Model configuration object.
            device (str): Device to run the model on ('cpu' or 'cuda').
            precision (str): Precision for calculations ('32' for float, '64' for double).
            reuse_neighbors (bool): Whether to cache and reuse the neighbor list.
            skin (float): Skin distance (Angstrom) for neighbor list caching.
            use_cuda_graph (bool | None): Capture the whole step (edge geometry,
                forward, autograd backward) into a CUDA graph and replay it.
                None (default) = auto: after the first capture, replay and
                eager are timed once per (system size, stress mode) and the
                faster path is kept — graphs win on launch-bound hardware
                (fast server GPUs), eager wins where kernel execution
                dominates. True forces graphs, False disables. Capture
                failure always falls back to eager with a warning.
            sticky_stress (bool): Once stress has been requested, compute
                energy+forces+stress in every subsequent calculate() call.
                ASE optimizers/NPT drivers request E, F and S as three
                separate calls per step; with sticky stress the first call
                fills the results cache and the other two are free, so a
                step costs one forward+backward instead of ~2.5. Pure-MD
                runs that never ask for stress are unaffected. Call
                reset_neighbor_cache() to clear the mode.
            use_fused_ops (bool | None): Use the fused Triton kernels
                (inference-only, fp32+CUDA). None (default) enables them
                automatically when Triton and a CUDA device are available;
                unsupported configurations silently use the reference path.
            **kwargs: Additional arguments for the base ASE Calculator.
        """
        Calculator.__init__(self, **kwargs)
        
        # --- Model Loading ---
        if precision == "64":
           config.dtype = '64'
        if ckpt_path.endswith('ckpt'):
          self.model = AlphaNetWrapper(config).to(torch.device(device))
          # Load state dict, ignoring mismatches if any
          self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)), strict=False)        
        elif ckpt_path.endswith('pt'):
           self.model = torch.load(ckpt_path, map_location=torch.device(device))
        else:
          raise ValueError(f"Unknown checkpoint format for file: {ckpt_path}") 
        
        self.device = torch.device(device)
        self.precision = torch.float32 if precision == "32" else torch.float64
        
        if precision == "64":
          self.model.double()
        
        self.model.eval() # Set model to evaluation mode
        self.model.to(self.device)
        self.config = config
        self.reuse_neighbors = reuse_neighbors
        self.supports_neighbor_cache = hasattr(self.model, "forward_graph")
        self.skin = max(float(skin), 0.0)
        self._neighbor_topology = None
        self._reference_positions = None
        self._reference_cell = None
        self._reference_numbers = None
        self._reference_pbc = None
        self._neighbor_cache_stats = {"rebuilds": 0, "reuses": 0}

        # --- CUDA graph state ---
        # None = auto: capture, then time replay vs eager once per
        # (n_atoms, stress-mode) and keep whichever is faster. The static-shape
        # graph computes ~(1+skin/cutoff)^3 more (masked) edges than the
        # filtered eager path, so which side wins is hardware- and
        # size-dependent: graphs win where per-kernel latency dominates.
        self._graph_auto = use_cuda_graph is None
        use_graph = True if use_cuda_graph is None else bool(use_cuda_graph)
        self.use_cuda_graph = use_graph and self.device.type == "cuda"
        if (self.use_cuda_graph and not self._graph_auto
                and getattr(config, "reduce_mode", "sum") != "sum"):
            warnings.warn(
                "use_cuda_graph=True ignored: the static-shape (masked) mode "
                "is only exact for reduce_mode='sum'; using the eager path.")
        self._captured_step = None
        self._graph_disabled = False
        self._graph_choice = {}  # (n_atoms, with_stress) -> bool (auto mode)
        self._graph_stats = {"captures": 0, "replays": 0, "topo_updates": 0}

        # Sticky stress mode (shared by the eager and CUDA-graph paths): once
        # stress is requested, every calculate() produces E+F+S so the ASE
        # E/F/S call battery of optimizers and NPT drivers costs a single
        # forward+backward per step. reset_neighbor_cache() clears it.
        self.sticky_stress = bool(sticky_stress)
        self._stress_sticky = False

        # --- fused Triton ops (inference-only, fp32+CUDA) ---
        if use_fused_ops is None:
            try:
                from alphanet import ops as _ops
                use_fused_ops = (_ops.is_available()
                                 and self.device.type == "cuda"
                                 and self.precision == torch.float32)
            except Exception:
                use_fused_ops = False
        inner_model = getattr(self.model, "model", None)
        if inner_model is not None:
            inner_model.use_fused_ops = bool(use_fused_ops)
            for message_layer in getattr(inner_model, "message_layers", []):
                message_layer.use_fused_ops = bool(use_fused_ops)
        self.use_fused_ops = bool(use_fused_ops)

    @property
    def neighbor_cache_stats(self):
        return dict(self._neighbor_cache_stats)

    @property
    def cuda_graph_stats(self):
        return dict(self._graph_stats)

    def reset_neighbor_cache(self):
        self._neighbor_topology = None
        self._reference_positions = None
        self._reference_cell = None
        self._reference_numbers = None
        self._reference_pbc = None
        self._stress_sticky = False
        self._release_captured_step()

    def _release_captured_step(self):
        if self._captured_step is not None:
            # Fully destroy the old graph (and its private memory pool) before
            # any new capture: on torch 2.1 capturing into a pool that still
            # holds live blocks trips a CUDACachingAllocator assert, and a
            # failed capture poisons the allocator state for the whole process.
            self._captured_step = None
            gc.collect()
            torch.cuda.synchronize(self.device)
            # Return the dead graph pool's blocks to the driver — otherwise
            # they stay reserved and the eager path (or the next capture)
            # can be pushed into shared-memory spill on small GPUs.
            torch.cuda.empty_cache()

    def _graph_step_for(self, topology, needs_stress, z, natoms, batch,
                        n_atoms, positions, cell_array):
        """Return a ready CapturedStep for this topology/request, or None to
        use the eager path. Handles (re)capture, cheap topology refresh and
        (in auto mode) the one-off replay-vs-eager timing decision."""
        if (not self.use_cuda_graph or self._graph_disabled or self.skin <= 0.0
                or getattr(self.config, "reduce_mode", "sum") != "sum"):
            # Masked static shapes zero out-of-cutoff messages, which is only
            # equivalent to hard edge filtering under sum aggregation.
            return None
        # needs_stress already reflects the sticky policy applied in
        # calculate(); a capture therefore keeps serving the whole E/F/S
        # battery of an optimizer step with a single replay. Even with the
        # sticky policy disabled, never downgrade an existing stress capture
        # to a forces-only one — that would thrash recaptures in optimizer
        # loops (extra stress outputs are stored and simply go unused).
        want_stress = needs_stress or (self._captured_step is not None
                                       and self._captured_step.with_stress)
        key = (n_atoms, want_stress)
        if self._graph_auto and self._graph_choice.get(key) is False:
            return None
        step = self._captured_step
        if step is not None and (step.n_atoms != n_atoms
                                 or step.with_stress != want_stress):
            self._release_captured_step()
            step = None
        if step is not None:
            if step.topology_ref is not topology:
                if step.update_topology(topology):
                    self._graph_stats["topo_updates"] += 1
                else:
                    # edge count outgrew the captured capacity -> recapture
                    self._release_captured_step()
                    step = None
        if step is None:
            try:
                step = CapturedStep(
                    model=self.model, topology=topology, z=z, natoms=natoms,
                    batch=batch, n_atoms=n_atoms, cutoff=self.config.cutoff,
                    skin=self.skin, precision=self.precision,
                    device=self.device, with_stress=want_stress,
                    positions=positions, cell=cell_array,
                )
            except Exception as exc:
                warnings.warn(
                    f"CUDA graph capture failed ({exc!r}); falling back to "
                    f"the eager execution path.")
                self._graph_disabled = True
                # salvage whatever the aborted capture left behind so the
                # eager fallback is not pushed into OOM
                gc.collect()
                torch.cuda.empty_cache()
                return None
            self._captured_step = step
            self._graph_stats["captures"] += 1
            if self._graph_auto and key not in self._graph_choice:
                # Time replay, release the graph (its private pool holds the
                # full activation memory — timing eager alongside would double
                # the footprint and spill on small GPUs), then time eager.
                t_graph = self._time_path(lambda: step.run(positions, cell_array))
                # drop every reference before releasing, or the graph pool
                # survives gc and its reserved blocks skew the eager timing
                step = None
                self._release_captured_step()
                t_eager = self._time_path(self._eager_runner(
                    topology, z, natoms, batch, want_stress, positions, cell_array))
                keep = t_graph <= t_eager
                self._graph_choice[key] = keep
                self._graph_stats["auto_choice"] = (
                    f"n={n_atoms} stress={want_stress} "
                    f"graph={t_graph*1e3:.1f}ms eager={t_eager*1e3:.1f}ms -> "
                    f"{'graph' if keep else 'eager'}")
                if not keep:
                    return None
                # graph wins: capture again (one-off cost) and keep using it
                step = CapturedStep(
                    model=self.model, topology=topology, z=z, natoms=natoms,
                    batch=batch, n_atoms=n_atoms, cutoff=self.config.cutoff,
                    skin=self.skin, precision=self.precision,
                    device=self.device, with_stress=want_stress,
                    positions=positions, cell=cell_array,
                )
                self._captured_step = step
                self._graph_stats["captures"] += 1
        step.refresh_z(z)
        return step

    def _eager_runner(self, topology, z, natoms, batch, want_stress,
                      positions, cell_array):
        cell = torch.tensor(cell_array, dtype=self.precision, device=self.device)

        def eager_once():
            with torch.enable_grad():
                pos = torch.tensor(positions, dtype=self.precision,
                                   device=self.device, requires_grad=True)
                graph_data = graph_from_neighbor_topology(
                    pos=pos, z=z, natoms=natoms, batch=batch,
                    topology=topology, cell=cell, cutoff=self.config.cutoff,
                    dtype=self.precision, compute_stress=want_stress)
                self.model.forward_graph(
                    graph_data, prefix="infer", compute_forces=True,
                    compute_stress=want_stress)

        return eager_once

    def _time_path(self, fn, repeats=5):
        import time as _time
        fn()  # warmup
        torch.cuda.synchronize(self.device)
        samples = []
        for _ in range(repeats):
            t0 = _time.perf_counter()
            fn()
            torch.cuda.synchronize(self.device)
            samples.append(_time.perf_counter() - t0)
        return float(np.median(samples))

    def _prepare_atoms(self):
        if not self.atoms.pbc.any():
            print("Non-periodic system detected. Automatically adding a large vacuum box for calculation.")
            calc_atoms = self.atoms.copy()
            padding = 20.0
            new_cell_dims = calc_atoms.get_positions().ptp(axis=0) + padding
            calc_atoms.set_cell(np.diag(new_cell_dims))
            calc_atoms.center()
            calc_atoms.pbc = True
            return calc_atoms
        return self.atoms

    def _pair_distance_change_bound(self, positions, cell, pbc):
        """Conservative upper bound on how much any pair distance can have
        changed since the cached neighbor list was built, including cell
        deformation (NPT / variable-cell optimization).

        With f/s the current/reference fractional coordinates, C/C_ref the
        current/reference cells and S the cached integer offsets, any pair
        vector change decomposes as
            dv = [(f_j - s_j) - (f_i - s_i)] @ C + (s_j + S - s_i) @ (C - C_ref)
        so ||dv|| <= 2 * max_i ||aligned_i - s_i @ C||
                    + (cutoff + skin) * ||inv(C_ref)||_2 * ||C - C_ref||_2,
        where the second term uses ||v_ref|| <= cutoff + skin for every cached
        pair (and monotonicity in ||v_ref|| for excluded pairs). The cached
        list stays valid while this bound does not exceed the skin.
        """
        if self._reference_positions is None or self._reference_cell is None:
            return float("inf")
        aligned = self._align_positions_to_reference(positions, cell, pbc)
        reference_frac = self._reference_positions @ np.linalg.inv(self._reference_cell)
        internal = aligned - reference_frac @ cell
        internal_max = float(np.max(np.linalg.norm(internal, axis=1))) if len(internal) else 0.0
        cell_delta = float(np.linalg.norm(cell - self._reference_cell, ord=2))
        inv_ref_norm = float(np.linalg.norm(np.linalg.inv(self._reference_cell), ord=2))
        radius = float(self.config.cutoff) + self.skin
        return 2.0 * internal_max + radius * inv_ref_norm * cell_delta

    def _align_positions_to_reference(self, positions, cell, pbc):
        """Map each atom onto the periodic image closest to its reference
        position. Fractional coordinates are taken in each configuration's own
        cell, so the mapping remains exact under cell deformation (NPT)."""
        if self._reference_positions is None or self._reference_cell is None:
            return positions.copy()
        current_frac = positions @ np.linalg.inv(cell)
        reference_frac = self._reference_positions @ np.linalg.inv(self._reference_cell)
        delta_frac = current_frac - reference_frac
        periodic_axes = np.asarray(pbc, dtype=bool)
        delta_frac[:, periodic_axes] -= np.round(delta_frac[:, periodic_axes])
        return (reference_frac + delta_frac) @ cell

    def _should_rebuild_topology(self, positions, cell, numbers, pbc):
        if not self.reuse_neighbors or self.skin <= 0.0:
            return True
        if self._neighbor_topology is None:
            return True
        if self._reference_numbers is None or numbers.shape != self._reference_numbers.shape:
            return True
        if not np.array_equal(numbers, self._reference_numbers):
            return True
        if self._reference_pbc is None or not np.array_equal(np.asarray(pbc, dtype=bool), self._reference_pbc):
            return True
        if self._reference_cell is None:
            return True
        return self._pair_distance_change_bound(positions, cell, pbc) > self.skin

    def _build_or_reuse_topology(self, positions, cell_array, numbers, pbc, pos, natoms):
        if self._should_rebuild_topology(positions, cell_array, numbers, pbc):
            self._neighbor_topology = build_neighbor_topology(
                pos=pos.detach(),
                natoms=natoms,
                cell=torch.tensor(cell_array, dtype=self.precision, device=self.device).detach(),
                cutoff=self.config.cutoff,
                skin=self.skin,
                precision=self.precision,
                numbers=numbers,
            )
            self._reference_positions = positions.copy()
            self._reference_cell = cell_array.copy()
            self._reference_numbers = numbers.copy()
            self._reference_pbc = np.asarray(pbc, dtype=bool).copy()
            self._neighbor_cache_stats["rebuilds"] += 1
        else:
            self._neighbor_cache_stats["reuses"] += 1
        return self._neighbor_topology

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Performs the calculation of energy, forces, and stress.

        Args:
            atoms (ase.Atoms): The atoms object to calculate properties for.
            properties (list of str): List of properties to calculate.
            system_changes (list of str): List of changes since the last calculation.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        properties = properties or ['energy']
        calc_atoms = self._prepare_atoms()
        needs_stress = 'stress' in properties
        if self.sticky_stress:
            if needs_stress:
                self._stress_sticky = True
            # sticky mode: keep producing E+F+S so the remaining property
            # calls of this optimizer/NPT step are served from the cache
            needs_stress = needs_stress or self._stress_sticky
        needs_forces = needs_stress or ('forces' in properties)
        grad_enabled = needs_forces or needs_stress

        # --- Prepare Tensors for the Model ---
        atomic_numbers = calc_atoms.get_atomic_numbers()
        wrapped_positions = calc_atoms.get_positions(wrap=True)
        cell_array = np.array(calc_atoms.get_cell(complete=True))
        z = torch.tensor(atomic_numbers, dtype=torch.long, device=self.device)
        natoms = torch.tensor(
            [len(calc_atoms)], 
            dtype=torch.int64, 
            device=self.device
        )
        batch = torch.zeros_like(z).to(self.device)

        # --- Run Model Inference ---
        # The stress path also uses the cached topology: the symmetric
        # displacement is injected in graph_from_neighbor_topology, so
        # NPT / variable-cell runs no longer rebuild the neighbor list
        # from scratch on every step.
        use_neighbor_cache = (
            self.reuse_neighbors
            and self.supports_neighbor_cache
            and calc_atoms.pbc.any()
        )
        positions_for_model = wrapped_positions
        if use_neighbor_cache:
            positions_for_model = self._align_positions_to_reference(
                wrapped_positions,
                cell_array,
                calc_atoms.pbc,
            )
        pos = torch.tensor(
            positions_for_model,
            dtype=self.precision,
            device=self.device,
            requires_grad=grad_enabled,
        )
        cell = torch.tensor(
            cell_array,
            dtype=self.precision,
            device=self.device
        ) if calc_atoms.pbc.any() else None
        with torch.set_grad_enabled(grad_enabled):
            if use_neighbor_cache:
                topology = self._build_or_reuse_topology(
                    positions=positions_for_model,
                    cell_array=cell_array,
                    numbers=atomic_numbers,
                    pbc=calc_atoms.pbc,
                    pos=pos,
                    natoms=natoms,
                )
                captured = self._graph_step_for(
                    topology, needs_stress, z, natoms, batch, len(calc_atoms),
                    positions_for_model, cell_array)
                if captured is not None:
                    energy, forces, stress = captured.run(
                        positions_for_model, cell_array)
                    self._graph_stats["replays"] += 1
                    return self._store_results(energy, forces, stress)
                graph_data = graph_from_neighbor_topology(
                    pos=pos,
                    z=z,
                    natoms=natoms,
                    batch=batch,
                    topology=topology,
                    cell=cell,
                    cutoff=self.config.cutoff,
                    dtype=self.precision,
                    compute_stress=needs_stress,
                )
                energy, forces, stress = self.model.forward_graph(
                    graph_data,
                    prefix="infer",
                    compute_forces=needs_forces,
                    compute_stress=needs_stress,
                )
            else:
                energy, forces, stress = self.model(
                    pos,
                    z,
                    batch,
                    natoms,
                    cell,
                    "infer",
                    compute_forces=needs_forces,
                    compute_stress=needs_stress,
                )
        
        self._store_results(energy, forces, stress)

    def _store_results(self, energy, forces, stress):
        """Copy model outputs into ASE results (device->host copies detach the
        values from any static CUDA-graph buffers before the next replay)."""
        self.results['energy'] = energy.detach().cpu().item()
        self.results['free_energy'] = self.results['energy']

        if forces is not None:
            self.results['forces'] = forces.detach().cpu().numpy()

        if stress is not None:
            # Convert the model's 3x3 stress tensor to ASE's Voigt notation (6-element vector)
            stress_matrix = stress.detach().cpu().numpy()
            self.results['stress'] = np.array([
                stress_matrix[0, 0],  # xx
                stress_matrix[1, 1],  # yy
                stress_matrix[2, 2],  # zz
                stress_matrix[1, 2],  # yz
                stress_matrix[0, 2],  # xz
                stress_matrix[0, 1]   # xy
            ])
