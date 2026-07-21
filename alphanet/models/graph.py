from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from ase import Atoms
from matscipy.neighbours import neighbour_list
from torch import Tensor


class GraphData(NamedTuple):
    pos: Tensor
    batch: Tensor
    z: Tensor
    natoms: Tensor
    edge_index: Tensor
    edge_attr: Tensor
    edge_vec: Tensor
    cell: Tensor = None
    cell_offsets: Tensor = None
    displacement: Optional[Tensor] = None
    pbc: Optional[Tensor] = None
    # Static-shape mode (CUDA-graph friendly): edge_index keeps every cached
    # edge (cutoff+skin) and edge_mask[e] = 1.0 iff 0 < dist <= cutoff.
    # None means edges were hard-filtered as before.
    edge_mask: Optional[Tensor] = None


@dataclass
class NeighborTopology:
    edge_index: Tensor
    cell_offsets: Tensor
    neighbors: Tensor
    reference_positions: Tensor
    reference_cell: Tensor
    cutoff: float
    skin: float


def _to_numpy_array(data) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _build_single_image_topology_matscipy(
    positions: np.ndarray,
    cell: np.ndarray,
    numbers: np.ndarray,
    radius: float,
    pbc: np.ndarray,
    edge_source_first: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    atoms = Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    index_i, index_j, shift = neighbour_list(
        quantities="ijS",
        atoms=atoms,
        cutoff=radius,
    )

    index_i = np.asarray(index_i, dtype=np.int64)
    index_j = np.asarray(index_j, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int32)

    if index_i.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        cell_offsets = np.empty((0, 3), dtype=np.int32)
        return edge_index, cell_offsets, 0

    if edge_source_first:
        edge_index = np.stack((index_j, index_i), axis=0)
    else:
        edge_index = np.stack((index_i, index_j), axis=0)

    return edge_index, shift, int(index_i.shape[0])


def check_and_reshape_cell(cell: Optional[torch.Tensor]) -> torch.Tensor:
    if cell is None:
        return torch.eye(3, dtype=torch.float32).unsqueeze(0)

    if cell.dim() == 2 and cell.size(0) % 3 == 0 and cell.size(1) == 3:
        batch_size = cell.size(0) // 3
        cell = cell.reshape(batch_size, 3, 3)
    elif cell.dim() != 3 or cell.size(1) != 3 or cell.size(2) != 3:
        raise ValueError(f"Invalid cell shape. Expected (batch_size, 3, 3), but got {cell.size()}")

    return cell


def build_neighbor_topology(
    pos: Tensor,
    natoms: Tensor,
    cell: Tensor,
    cutoff: float,
    skin: float = 0.0,
    pbc: Optional[List[bool]] = None,
    precision: torch.dtype = torch.float32,
    numbers: Optional[Tensor] = None,
    edge_source_first: bool = True,
) -> NeighborTopology:
    cell = check_and_reshape_cell(cell)
    radius = cutoff + max(skin, 0.0)
    device = pos.device
    pbc_array = np.asarray(
        [True, True, True] if pbc is None else pbc,
        dtype=bool,
    )
    natoms_np = _to_numpy_array(natoms).astype(np.int64)
    pos_np = _to_numpy_array(pos).astype(np.float64, copy=False)
    cell_np = _to_numpy_array(cell).astype(np.float64, copy=False)

    if numbers is None:
        numbers_np = np.ones(pos_np.shape[0], dtype=np.int32)
    else:
        numbers_np = _to_numpy_array(numbers).astype(np.int32, copy=False)

    edge_indices = []
    cell_offsets = []
    num_neighbors_image = []

    atom_offset = 0
    for image_index, image_natoms in enumerate(natoms_np):
        image_natoms = int(image_natoms)
        image_slice = slice(atom_offset, atom_offset + image_natoms)
        image_edge_index, image_offsets, image_neighbors = _build_single_image_topology_matscipy(
            positions=pos_np[image_slice],
            cell=cell_np[image_index],
            numbers=numbers_np[image_slice],
            radius=radius,
            pbc=pbc_array,
            edge_source_first=edge_source_first,
        )
        if image_neighbors > 0:
            image_edge_index = image_edge_index + atom_offset
            edge_indices.append(torch.from_numpy(image_edge_index))
            cell_offsets.append(torch.from_numpy(image_offsets))
        num_neighbors_image.append(image_neighbors)
        atom_offset += image_natoms

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1).to(
            device=device,
            dtype=torch.long,
        )
        cell_offsets_tensor = torch.cat(cell_offsets, dim=0).to(
            device=device,
            dtype=torch.int32,
        )
    else:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        cell_offsets_tensor = torch.empty((0, 3), device=device, dtype=torch.int32)
    neighbors = torch.tensor(
        num_neighbors_image,
        device=device,
        dtype=torch.long,
    )
    return NeighborTopology(
        edge_index=edge_index,
        cell_offsets=cell_offsets_tensor,
        neighbors=neighbors,
        reference_positions=pos.detach().clone(),
        reference_cell=cell.detach().clone(),
        cutoff=cutoff,
        skin=max(skin, 0.0),
    )


def _update_edge_geometry(
    pos: Tensor,
    batch: Tensor,
    edge_index: Tensor,
    cell: Tensor,
    cell_offsets: Tensor,
    precision: torch.dtype = torch.float32,
    cutoff: Optional[float] = None,
    static_shapes: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    row = edge_index[0]
    col = edge_index[1]
    edge_batch = batch[col]
    cell_per_edge = cell[edge_batch]
    offsets = (
        cell_offsets.to(precision)
        .view(-1, 1, 3)
        .bmm(cell_per_edge.to(precision))
        .view(-1, 3)
    )
    distance_vectors = pos[row] - pos[col] + offsets
    distances = distance_vectors.norm(dim=-1, p=2)
    valid_mask = distances > 0
    if cutoff is not None:
        valid_mask = torch.logical_and(valid_mask, distances <= cutoff)
    if static_shapes:
        # Keep every cached edge so tensor shapes stay constant within a
        # topology window (CUDA-graph capturable); validity is expressed as a
        # multiplicative mask instead of index filtering. Mathematically
        # equivalent: out-of-cutoff edge contributions are zeroed in the model.
        edge_mask = valid_mask.to(precision).unsqueeze(-1)
        return edge_index, cell_offsets, distances, distance_vectors, edge_mask
    edge_index = edge_index[:, valid_mask]
    cell_offsets = cell_offsets[valid_mask]
    distances = distances[valid_mask]
    distance_vectors = distance_vectors[valid_mask]
    return edge_index, cell_offsets, distances, distance_vectors, None


def graph_from_neighbor_topology(
    pos: Tensor,
    z: Tensor,
    natoms: Tensor,
    batch: Tensor,
    topology: NeighborTopology,
    cell: Optional[Tensor] = None,
    displacement: Optional[Tensor] = None,
    cutoff: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    compute_stress: bool = False,
    static_shapes: bool = False,
) -> GraphData:
    precision = dtype
    pos = pos.to(precision)
    z = z.long()
    cell = check_and_reshape_cell(cell)
    if compute_stress and displacement is None:
        # Inject the symmetric-displacement trick on top of the cached
        # topology: the displacement tensor is numerically zero, so positions,
        # cell and hence the cached neighbor list are unchanged; the edge
        # geometry below is recomputed from the displaced pos/cell and is
        # therefore differentiable w.r.t. the displacement (stress via
        # autograd), exactly as in process_positions_and_edges.
        pos, cell, displacement = get_symmetric_displacement(
            pos, cell, num_graphs=int(natoms.numel()), batch=batch
        )
        cell = check_and_reshape_cell(cell)
    edge_index, cell_offsets, dist, vecs, edge_mask = _update_edge_geometry(
        pos=pos,
        batch=batch,
        edge_index=topology.edge_index,
        cell=cell,
        cell_offsets=topology.cell_offsets,
        precision=precision,
        cutoff=topology.cutoff if cutoff is None else cutoff,
        static_shapes=static_shapes,
    )
    return GraphData(
        pos=pos,
        z=z,
        natoms=natoms,
        batch=batch,
        edge_index=edge_index,
        edge_attr=dist,
        edge_vec=vecs,
        cell=cell,
        cell_offsets=cell_offsets,
        displacement=displacement,
        edge_mask=edge_mask,
    )


# Borrowed from MACE
def get_symmetric_displacement(
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        num_graphs: int,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cell is None:
            cell = torch.zeros(
                num_graphs * 3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            )

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )

        displacement.requires_grad_(True)
        symmetric_displacement = 0.5 * (
            displacement + displacement.transpose(-1, -2)
        )

        positions = positions + torch.einsum(
            "be,bec->bc", positions, symmetric_displacement[batch]
        )
        cell = cell.view(-1, 3, 3)
        cell = cell + torch.matmul(cell, symmetric_displacement)
        cell.view(-1, 3)
        return positions, cell, displacement


def process_positions_and_edges(
    pos: Tensor,
    z: Tensor,
    natoms: Tensor,
    batch: Tensor,
    cell: Optional[Tensor] = None,
    compute_stress: bool = False,
    compute_forces: bool = False,
    use_pbc: bool = False,
    cutoff: float = 5.0,
    dtype: torch.dtype = torch.float32
) -> GraphData:
    """
    Process atomic positions and compute edges with PBC support.

    Non-PBC mode is not supported directly; create a large vacuum cell instead.
    """
    precision = dtype
    pos = pos.to(precision)
    z = z.long()

    if compute_stress:
        pos, cell, displacement = get_symmetric_displacement(
            pos, cell, num_graphs=int(torch.max(batch)) + 1, batch=batch
        )
    else:
        displacement = None

    cell = check_and_reshape_cell(cell)

    if not use_pbc or cell is None:
        raise ValueError(
            "Non-PBC mode is not supported; please create a large vacuum cell."
        )

    topology = build_neighbor_topology(
        pos=pos,
        natoms=natoms,
        cell=cell,
        cutoff=cutoff,
        skin=0.0,
        precision=precision,
        numbers=z,
    )
    return graph_from_neighbor_topology(
        pos=pos,
        z=z,
        natoms=natoms,
        batch=batch,
        topology=topology,
        cell=cell,
        displacement=displacement,
        cutoff=cutoff,
        dtype=precision,
    )
