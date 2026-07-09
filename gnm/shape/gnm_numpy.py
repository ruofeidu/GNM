# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NumPy implementation of the GNM model .

Example usage:
  ```
  gnm = gnm_numpy.from_local(version=GNMVersion.V3, variant=GNMVariant.HEAD)

  # Generate random identity, expression, rotations, translation parameters
  identity = np.random.normal(size=gnm.identity_dim)
  expression = np.zeros(gnm.expression_dim)
  rotations = np.random.uniform(-1, 1, size=(gnm.num_joints, 3)) * 0.15
  translation = np.random.uniform(-1, 1, size=(3,)) * 0.15

  vertices = gnm(identity, expression, rotations, translation)
  ```
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
from typing import Any, TYPE_CHECKING

from absl import logging
from etils import epath
from gnm.shape import gnm_base
from gnm.shape import gnm_landmarks
from gnm.shape.data.versions import gnm_specs
import numpy as np
import numpy.typing as npt
import opt_einsum as oe

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType

_NONZERO_THRESHOLD = 1e-4
_EPSILON = 1e-8


@dataclasses.dataclass(init=False, kw_only=True)
class GNM(gnm_base.GNMBase):
  """NumPy implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  The GNM class also surfaces useful data for down-stream users, e.g. the
  names of each expression dimension, and the topology of the mesh.

  Shape dimensions are denoted:
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.
  * Q: The number of quads in the mesh topology.
  * T: The number of triangles, in a triangulated version of the mesh topology.
  * G: Number of vertex groups.
  * P: Number of separate parts of the mesh.

  Attributes:
    version: The version of the loaded GNM model.
    variant: The variant of the loaded GNM model.
    template_vertex_positions: Vertex positions in the template mesh, (V, 3).
    template_joint_positions: Joint positions in the template GNM, (J, 3).
    vertex_identity_basis: The vertex identity basis of the model, (I, V, 3).
    joint_identity_basis: The joint identity basis of the model, (I, J, 3).
    expression_basis: The vertex expression basis, aka blend-shapes, (E, V, 3).
    identity_names: The name of each identity in the identity basis.
    joint_names: The name of each joint in the skeleton.
    expression_names: The name of each expression in the expression basis.
    joint_parent_indices: Parent's index for each joint in the skeleton, (J).
    skinning_weights: The model's skinning weights, (J, V).
    quads: The mesh topology as quads, (Q, 4).
    triangles: The mesh topology as triangles, (T, 3).
    quad_uvs: Texture coordinates per quad, (Q, 4, 2).
    triangle_uvs: Texture coordinates per triangle, (T, 3, 2).
    mesh_component_names: The vertex group name corresponding to each separate
      mesh part.
    mirror_indices: The index of each vertex on the other side of the mesh.
    joint_regressor: Mapping from vertices to joints, (J, V).
    pose_correctives_regressor: Matrix for pose correctives, (9*J, 3*V).
    bone_aligned_template_joint_orientations: The bone-aligned rotations for
      each joint, (J, 3, 3). If they do not exist in the GNM npz, they are set
      to the identity matrix. Note that these are not used to compute the GNM
      joint and vertex positions.
    vertex_groups: The weights in each vertex group, (G, V).
    vertex_group_names: The name of each vertex group, (G,).
    joint_connections: The joint connections of the GNM rig.
    joint_children_indices: A dictionary that contains the joint indices of the
      children of each joint.
    skinning_segmentation: A decomposition of the vertices into separate parts
      based on skinning weights.
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
    edge_list: The quad topology represented as a list of directed edges (E, 2).
    vertex_uvs: Per-vertex UV texture coordinates shaped (V, 2).
  """

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant
  template_vertex_positions: np.ndarray  # (V, 3)
  template_joint_positions: np.ndarray  # (J, 3)
  vertex_identity_basis: np.ndarray  # (I, V, 3)
  joint_identity_basis: np.ndarray  # (I, J, 3)
  expression_basis: np.ndarray  # (E, V, 3)
  identity_names: Sequence[str]  # (I,)
  joint_names: Sequence[str]  # (J,)
  expression_names: Sequence[str]  # (E,)
  joint_parent_indices: np.ndarray  # (J,)
  skinning_weights: np.ndarray  # (J, V)
  quads: np.ndarray  # (Q, 4)
  triangles: np.ndarray  # (T, 3)
  quad_uvs: np.ndarray  # (Q, 4, 2)
  triangle_uvs: np.ndarray  # (T, 3, 2)
  mesh_component_names: Sequence[str]  # (P,)
  mirror_indices: np.ndarray  # (V,)
  joint_regressor: np.ndarray  # (J, V)
  pose_correctives_regressor: np.ndarray  # (9*J, 3*V)
  bone_aligned_template_joint_orientations: np.ndarray  # (J, 3, 3)
  vertex_groups: np.ndarray  # (G, V)
  vertex_group_names: Sequence[str]  # (G,)

  if TYPE_CHECKING:
    _vertex_group_names_lookup: dict[str, int]
    _landmarks: dict[
        gnm_landmarks.GNMLandmarksType, gnm_landmarks.LandmarksConfiguration
    ]

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a GNM instance from a model data."""
    instance = super().__new__(cls)  # pylint: disable=no-value-for-parameter

    # Set the data fields.
    for field in dataclasses.fields(cls):
      setattr(instance, field.name, model_data[field.name])

    # Store the vertex group names lookup.
    instance._vertex_group_names_lookup = _name_index_lookup(
        instance.vertex_group_names
    )

    return instance

  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

  def __call__(
      self,
      identity: npt.NDArray[np.floating] | None = None,
      expression: npt.NDArray[np.floating] | None = None,
      rotations: npt.NDArray[np.floating] | None = None,
      translation: npt.NDArray[np.floating] | None = None,
  ):
    """Evaluates the GNM mesh-generating function.

    Input parameters have optional batch dimensions [A1, ..., An]. Parameters
    may be omitted - in this case default values will be substituted.

    Args:
      identity: Identity coefficients ([A1, ..., An], I).
      expression: Expression coefficients ([A1, ..., An], E).
      rotations: Joint rotations ([A1, ..., An], J, 3),
      translation: Root-joint translation ([A1, ..., An], 3).

    Returns:
      Mesh vertices ([A1, ..., An], V, 3).

    Raises:
      ValueError if an input argument has the wrong format.
    """
    batch_dims = _get_batch_dims(identity, expression, rotations, translation)

    # Fill in missing parameter values with zeros.
    if identity is None:
      identity = np.zeros((*batch_dims, self.identity_dim), dtype=np.float32)
    if expression is None:
      expression = np.zeros(
          (*batch_dims, self.expression_dim), dtype=np.float32
      )
    if rotations is None:
      rotations = np.zeros((*batch_dims, self.num_joints, 3), dtype=np.float32)
    if translation is None:
      translation = np.zeros((*batch_dims, 3), dtype=np.float32)

    _check_batch_dims(identity, expression, rotations, translation)

    output_type = identity.dtype

    # Joint positions in bind pose, identity applied, ([A0, ..., An], J, 3).
    joints_bind = self.joint_positions_bind_pose(identity)

    # The local-to-world transforms of each joint ([A0, ..., An], J, 4, 4).
    joint_transforms_world = self.joint_transforms_world(
        joints_bind, rotations, translation
    )

    # Skinning requires we compute the transform from the bind pose to the final
    # pose for each joint. To do this, we first transform vertices from the bind
    # space to the joint space, then apply joints local-to-world transforms.
    # Since rotations in bind pose are identity, we can skip the rotation
    # part and compute RT_bind_to_world = RT_joint_to_world @ T(-joints_bind).
    bind_to_joint_transforms = np.tile(
        np.eye(4, dtype=output_type),
        (*batch_dims, self.num_joints, 1, 1),
    )
    bind_to_joint_transforms[..., :3, 3] = -joints_bind
    bind_to_world_transforms = joint_transforms_world @ bind_to_joint_transforms

    # Vertex positions in the bind pose, with identity, expression, and
    # pose-correctives applied. Already batched ([A0, ..., An], V, 3).
    vertices_bind = self.vertex_positions_bind_pose(
        identity, expression
    ) + self.compute_pose_correctives(rotations)

    # Convert bind vertices to homogeneous coordinates and apply LBS.
    vertices_bind_h = np.concatenate(
        [
            vertices_bind,
            np.ones_like(vertices_bind[..., :1], dtype=output_type),
        ],
        -1,
    )

    return oe.contract(
        'jv,...jmn,...vn->...vm',
        self.skinning_weights,
        bind_to_world_transforms,
        vertices_bind_h,
    )[..., :3]

  def vertices_and_landmarks(
      self,
      landmarks_type: gnm_landmarks.GNMLandmarksType,
      identity: npt.NDArray[np.floating] | None = None,
      expression: npt.NDArray[np.floating] | None = None,
      rotations: npt.NDArray[np.floating] | None = None,
      translation: npt.NDArray[np.floating] | None = None,
  ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Evaluates the GNM mesh-generating function and extracts 3D landmarks.

    Args:
      landmarks_type: The type of landmarks to extract.
      identity: Identity coefficients ([A1, ..., An], I).
      expression: Expression coefficients ([A1, ..., An], E).
      rotations: Joint rotations ([A1, ..., An], J, 3).
      translation: Root-joint translation ([A1, ..., An], 3).

    Returns:
      A tuple of (vertices, landmarks) where vertices has shape
      ([A1, ..., An], V, 3) and landmarks has shape ([A1, ..., An], L, 3).
    """
    gnm_landmarks.check_body_part_compatibility(landmarks_type, self.body_part)
    if not hasattr(self, '_landmarks'):
      self._landmarks = {}
    if landmarks_type not in self._landmarks:
      self._landmarks[landmarks_type] = gnm_landmarks.load_landmarks(
          landmarks_type
      )
    config = self._landmarks[landmarks_type]

    vertices = self(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    face_vertices = vertices[..., config.indices, :]
    landmarks = np.sum(face_vertices * config.weights[..., None], axis=-2)
    return vertices, landmarks

  @property
  def num_vertices(self) -> int:
    """The number of vertices in the mesh (V)."""
    return len(self.template_vertex_positions)

  @property
  def num_joints(self) -> int:
    """The number of joints in the skeleton (J)."""
    return len(self.joint_names)

  @property
  def identity_dim(self) -> int:
    """The dimensionality of the linear identity basis (I)."""
    return len(self.vertex_identity_basis)

  @property
  def expression_dim(self) -> int:
    """The dimensionality of the linear expression basis (E)."""
    return len(self.expression_names)

  @functools.cached_property
  def edge_list(self) -> npt.NDArray[np.integer]:
    """The quad topology represented as a list of directed edges."""

    # Get edge-vertex pairs for all quads.
    e1 = self.quads.ravel()
    e2 = np.roll(self.quads, -1, axis=1).ravel()

    # Sort each edge so we can get unique edges.
    edges = np.stack([np.minimum(e1, e2), np.maximum(e1, e2)], axis=1)

    edge_keys = e1 + self.num_vertices * e2  # Key for faster unique-finding.
    _, unique_indices = np.unique(edge_keys, return_index=True)
    unique_undirected_edges = edges[unique_indices]

    # Combine with flipped edges to get directed edges.
    directed_edges = np.vstack(
        [unique_undirected_edges, np.fliplr(unique_undirected_edges)]
    )

    return directed_edges

  @functools.cached_property
  def joint_connections(self) -> npt.NDArray[np.int32]:
    """The bones of the GNM rig."""
    return np.stack(
        [self.joint_parent_indices, list(range(self.num_joints))], axis=1
    )[1:]

  @functools.cached_property
  def joint_children_indices(self) -> Mapping[int, Sequence[int]]:
    children = collections.defaultdict(list)
    for joint, parent in enumerate(self.joint_parent_indices):
      if joint == 0:
        continue
      children[parent].append(joint)
    return dict(children)

  @functools.cached_property
  def vertex_uvs(self) -> npt.NDArray[np.floating]:
    """Per-vertex UV texture coordinates shaped (V, 2).

    Warning: these do not allow correct texturing across UV-seams. Please use
    per-face UV coordinates when possible. This property is exposed to support
    certain renderers or packages that do not support per-face UVs.

    With per-face UVs, each vertex may have multiple UV coordinates which must
    be reduced to a single one. We assign each vertex the last UV coordinate
    associated with it in the order given by the quad topology.
    """

    logging.warning(
        'These do not allow correct texturing across UV-seams. Please use '
        'per-quad or per-triangle UV coordinates when possible.'
    )

    indices = self.quads.ravel()
    uvs = self.quad_uvs.reshape(-1, 2)

    # Find the index of the last occurrence for each unique vertex index.
    _, last_indices_reversed = np.unique(indices[::-1], return_index=True)
    last_indices = len(indices) - 1 - last_indices_reversed

    # Use the last occurrence indices to select the UV coordinates.
    vertex_uvs = np.zeros((self.num_vertices, 2), dtype=uvs.dtype)
    vertex_uvs[indices[last_indices]] = uvs[last_indices]

    return vertex_uvs

  def add_vertex_group(
      self, name: str, value: npt.NDArray[np.floating]
  ) -> None:
    """Adds a new vertex group."""
    if name in self.vertex_group_names:
      raise ValueError(f'Vertex group {name} already exists.')

    if value.ndim != 1 or len(value) != self.num_vertices:
      raise ValueError(
          f'Vertex group {name} must be a 1D array of length'
          f' {self.num_vertices}.'
      )
    self.vertex_groups = np.vstack([self.vertex_groups, value])
    self.vertex_group_names = list(self.vertex_group_names) + [name]
    self._vertex_group_names_lookup[name] = len(self.vertex_group_names) - 1

  @property
  def skinning_segmentation(self) -> npt.NDArray[np.int32]:
    """A part segmentation of the vertices based on skinning weights."""
    return self.skinning_weights.argmax(axis=0)

  def vertex_positions_bind_pose(
      self,
      identity: npt.NDArray[np.floating],
      expression: npt.NDArray[np.floating],
  ) -> npt.NDArray[np.floating]:
    """Vertex positions in the bind pose ([A0, ..., An], V, 3).

    Mesh vertices in the bind pose with identity and expression bases applied.

    Args:
      identity: Identity coefficients, ([A0, ..., An], I).
      expression: Expression coefficients, ([A0, ..., An], E).

    Returns:
      Vertex positions in the bind pose, ([A0, ..., An], V, 3).
    """
    _check_batched_shape(identity, 'identity', (self.identity_dim,))
    _check_batched_shape(expression, 'expression', (self.expression_dim,))

    identity_deltas = oe.contract(
        '...i,ijk->...jk', identity, self.vertex_identity_basis
    )
    expression_deltas = oe.contract(
        '...i,ijk->...jk', expression, self.expression_basis
    )

    return self.template_vertex_positions + identity_deltas + expression_deltas

  def joint_positions_bind_pose(
      self, identity: npt.NDArray[np.floating]
  ) -> npt.NDArray[np.floating]:
    """Joint positions in the bind pose, ([A0, ..., An], J, 3).

    Joint positions in the bind pose with identity basis applied.

    Args:
      identity: Identity coefficients, ([A0, ..., An], I).

    Returns:
      Joint positions in the bind pose, ([A0, ..., An], J, 3).
    """
    _check_batched_shape(identity, 'identity', (self.identity_dim,))

    identity_deltas = oe.contract(
        '...i,ijk->...jk', identity, self.joint_identity_basis
    )
    return self.template_joint_positions + identity_deltas

  def compute_pose_correctives(
      self, rotations: npt.NDArray[np.floating]
  ) -> npt.NDArray[np.floating]:
    """Compute per-vertex pose corrective offsets ([A0, ..., An], V, 3).

    If `self.pose_correctives_regressor` is `None`, then this GNM instance
    has no correctives and this function returns a zero array of the same shape.

    Args:
      rotations: Joint rotations in axis-angle format, ([A0, ..., An], J, 3).

    Returns:
      Pose correctives as vertex offsets, ([A0, ..., An], V, 3).
    """
    _check_batched_shape(rotations, 'rotations', (self.num_joints, 3))

    if self.pose_correctives_regressor is None:
      return np.zeros_like(self.template_vertex_positions)

    # Required to ensure pose_feature does not truncate to integer.
    output_type = np.float32 if rotations.itemsize == 4 else np.float64

    batch_dims = rotations.shape[:-2]
    pose_feature = _rotation_matrix(rotations) - np.identity(
        3, dtype=output_type
    )
    pose_feature = pose_feature.reshape(*batch_dims, self.num_joints * 9)

    return oe.contract(
        '...f,fv->...v', pose_feature, self.pose_correctives_regressor
    ).reshape(*batch_dims, self.num_vertices, 3)

  def joint_transforms_world(
      self,
      joints: npt.NDArray[np.floating],
      rotations: npt.NDArray[np.floating],
      translation: npt.NDArray[np.floating],
  ) -> npt.NDArray[np.floating]:
    """Gets the world-space transforms of each joint, ([A0, ..., An], J, 4, 4).

    Args:
      joints: Joint locations in the bind pose, ([A0, ..., An], J, 3).
      rotations: Per-joint rotations as axis-angle, ([A0, ..., An], J, 3).
      translation: The translation of the root joint ([A0, ..., An], 3).

    Returns:
      Local-to-world transforms of each joint, ([A0, ..., An], J, 4, 4).
    """
    _check_batched_shape(joints, 'joints', (self.num_joints, 3))
    _check_batched_shape(rotations, 'rotations', (self.num_joints, 3))
    _check_batched_shape(translation, 'translation', (3,))

    output_type = joints.dtype
    batch_dims = joints.shape[:-2]

    # Gather the parent's joints position for each joint.
    joint_parents = joints[..., self.joint_parent_indices[1:], :]

    # Assemble joints local transforms (in parent-space).
    world_transforms = np.tile(
        np.eye(4, dtype=output_type), (*batch_dims, self.num_joints, 1, 1)
    )

    # Set local rotations.
    world_transforms[..., :3, :3] = _rotation_matrix(rotations)

    # Set local translations.
    world_transforms[..., 0, :3, 3] = joints[..., 0, :] + translation
    world_transforms[..., 1:, :3, 3] = joints[..., 1:, :] - joint_parents

    # Compute forward-kinematics iterating from the root joint. Each joint's
    # transform is the product of its local transform and its parent's world
    # transform.
    for j in range(1, self.num_joints):
      world_transforms[..., j, :, :] = (
          world_transforms[..., self.joint_parent_indices[j], :, :]
          @ world_transforms[..., j, :, :]
      )

    return world_transforms

  def get_posed_joint_transforms(
      self,
      identity: npt.NDArray[np.floating],
      rotations: npt.NDArray[np.floating],
      translation: npt.NDArray[np.floating],
  ) -> npt.NDArray[np.floating]:
    """Gets the world-space transforms of each joint, ([A0, ..., An], J, 4, 4).

    Args:
      identity: Identity coefficients ([A0, ..., An], I).
      rotations: Joint rotations ([A0, ..., An], J, 3),
      translation: Root-joint translation ([A0, ..., An], 3).

    Returns:
      Local-to-world joint transformations ([A0, ..., An], J, 4, 4).

    Raises:
      ValueError if an input argument has the wrong format.
    """
    _check_batched_shape(identity, 'identity', (self.identity_dim,))
    _check_batched_shape(rotations, 'rotations', (self.num_joints, 3))
    _check_batched_shape(translation, 'translation', (3,))

    return self.joint_transforms_world(
        joints=self.joint_positions_bind_pose(identity),
        rotations=rotations,
        translation=translation,
    )

  def vertex_group(self, name: str) -> npt.NDArray[np.floating]:
    """Returns the scalar value of each vertex in the vertex group, (V,).

    Args:
      name: Vertex group name.

    Raises:
      KeyError: If the vertex group name is not found.

    Returns:
      1D array of float32 values, one for each vertex.
    """
    try:
      return self.vertex_groups[self._vertex_group_names_lookup[name]]
    except KeyError as exc:
      raise KeyError(
          f'Vertex group {name} not found in {self.vertex_group_names}.'
      ) from exc

  def vertex_group_mask(
      self, *names: str, threshold: float = _NONZERO_THRESHOLD
  ) -> npt.NDArray[bool]:
    """Returns the mask of vertices with non-zero value in the group(s).

    Multiple groups may be provided.
    Example 1: Seeking vertices that intersect two groups:
      `gnm_np.vertex_group_mask('group_a', '&group_b')`.
    Example 2: Negation of a group:
      `gnm_np.vertex_group_mask('~group_a')`.

    Warning: operations are applied consecutively regardless of precedence.

    Args:
      *names: Vertex group name(s) optionally prepended with an operator and
        negation `[op][~]group_name`. Operators: '|' (default), '&', '-'.
      threshold: Threshold for non-zero values.

    Returns:
      1D array representing vertex mask (np.bool_).
    """
    result_mask = np.zeros(self.num_vertices, dtype=bool)
    for name in names:
      operator, inverse = '|', False  # Default to logical OR (union).
      if name[0] in '|&-':
        operator, name = name[0], name[1:]
      if name[0] == '~':
        inverse, name = True, name[1:]
      group_mask = self.vertex_group(name) > threshold
      if inverse:
        group_mask = ~group_mask
      match operator:
        case '|':  # Logical OR.
          result_mask |= group_mask
        case '&':  # Logical AND.
          result_mask &= group_mask
        case '-':  # Logical AND NOT.
          result_mask &= ~group_mask
    return result_mask

  def vertex_group_indices(
      self, *names: str, threshold: float = _NONZERO_THRESHOLD
  ) -> npt.NDArray[np.integer]:
    """Returns the indices of vertices with non-zero value in the group(s).

    Multiple groups may be provided. See `vertex_group_mask` for examples.

    Args:
      *names: Vertex group name(s) optionally prepended with an operator and
        negation `[op][~]group_name`. Operators: '|' (default), '&', '-'.
      threshold: Threshold for non-zero values.

    Returns:
      1D array of vertex indices (np.int64).
    """
    return np.where(self.vertex_group_mask(*names, threshold=threshold))[0]

  def quad_indices_for_group(self, *names: str) -> npt.NDArray[np.integer]:
    """Quad indices for which all vertices belong to a group, (Q')."""
    vertex_indices = self.vertex_group_indices(*names)
    return np.where(np.all(np.isin(self.quads, vertex_indices), axis=-1))[0]

  def triangle_indices_for_group(self, *names: str) -> npt.NDArray[np.integer]:
    """Triangle indices for which all vertices belong to a group, (T')."""
    vertex_indices = self.vertex_group_indices(*names)
    return np.where(np.all(np.isin(self.triangles, vertex_indices), axis=-1))[0]

  def quads_group(self, *names: str) -> npt.NDArray[np.integer]:
    """Quads for which all vertices belong to a vertex group, (Q', 4)."""
    return self.quads[self.quad_indices_for_group(*names)]

  def triangles_group(self, *names: str) -> npt.NDArray[np.integer]:
    """Triangles for which all vertices belong to a vertex group, (T', 3)."""
    return self.triangles[self.triangle_indices_for_group(*names)]

  def quad_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    """Per-quad UV texture coordinates for a vertex group, (Q', 4, 2)."""
    return self.quad_uvs[self.quad_indices_for_group(*names)]

  def triangle_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    """Per-triangle UV texture coordinates for a vertex group, (T', 3, 2)."""
    return self.triangle_uvs[self.triangle_indices_for_group(*names)]

  def vertex_uvs_group(self, *names: str) -> npt.NDArray[np.floating]:
    """Per-vertex UV texture coordinates for a vertex group, (V', 2)."""
    return self.vertex_uvs[self.vertex_group_indices(*names)]

  def compute_vertex_normals(
      self, vertices: npt.NDArray[np.floating]
  ) -> npt.NDArray[np.floating]:
    """Compute vertex normals for the provided GNM vertex positions.

    This calculation averages the adjacent face normals per-vertex, weighting
    by face area.

    Args:
      vertices: The GNM vertices in world space, ([A0, ..., An], V, 3).

    Returns:
      The vertex normals, ([A0, ..., An], V, 3).
    """
    batch_dims = vertices.shape[:-2]
    non_batch_dims = vertices.shape[-2:]  # (V, 3)
    vertices = vertices.reshape(-1, *non_batch_dims)  # (B, V, 3)

    triangles = self.triangles
    face_vertices = vertices[..., triangles, :]  # (B, T, 3, 3)

    v0, v1, v2 = np.moveaxis(face_vertices, -2, 0)
    face_normals_area = np.cross(v1 - v0, v2 - v0, axis=-1)  # (B, T, 3)

    vertex_normals = np.zeros_like(vertices)
    # Sum the face normals across all triangles and assign sums per-vertex.
    np.add.at(
        vertex_normals,
        (slice(None), triangles, slice(None)),
        face_normals_area[:, :, None, :],  # Broadcast to (B, T, 3, 3)
    )

    # Normalize the vertex normals.
    normal_magnitudes = np.linalg.norm(vertex_normals, axis=-1, keepdims=True)
    if np.any(np.isclose(normal_magnitudes, 0.0)):
      logging.warning(
          'Some vertex normals have zero magnitude. This is unexpected and'
          ' likely indicates triangle collapse.'
      )
    vertex_normals = vertex_normals / np.maximum(normal_magnitudes, _EPSILON)
    return vertex_normals.reshape(*batch_dims, *non_batch_dims)

  def _check_inputs(
      self,
      identity: npt.NDArray[np.floating],
      expression: npt.NDArray[np.floating],
      rotations: npt.NDArray[np.floating],
      translation: npt.NDArray[np.floating],
  ) -> None:
    """Checks the shape of inputs to the mesh-generating function."""
    _check_shape(identity, 'identity', (self.identity_dim,))
    _check_shape(expression, 'expression', (self.expression_dim,))
    _check_shape(rotations, 'rotations', (self.num_joints, 3))
    _check_shape(translation, 'translation', (3,))


def _name_index_lookup(
    names: Sequence[str],
) -> dict[str, int]:
  """Returns a lookup for the index of each name in the sequence."""
  return {name: i for i, name in enumerate(names)}


def _rotation_matrix(
    axis_angle: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
  """Builds 3x3 rotation matrices from axis-angle vectors.

  The rotation matrix is computed using Rodrigues' rotation formula.  See here
  for more information:
  https://en.wikipedia.org/wiki/Rodrigues_rotation_formula

  Args:
    axis_angle: The axis of rotation is the direction of this vector and its
      norm is the angle of rotation (in radians), with shape (..., 3).

  Returns:
    3x3 rotation matrices with shape (..., 3, 3).
  """
  norm_squared = np.sum(np.square(axis_angle), axis=-1, keepdims=True)
  angle = np.where(
      norm_squared < _EPSILON,
      0.0,
      np.sqrt(norm_squared),
  )

  axis = axis_angle / (angle + _EPSILON)

  # Compute the sine and cosine of the angle and convert the shapes to
  # [..., 1, 1] to enable broadcasting.
  sin_a, cos_a = np.sin(angle), np.cos(angle)
  sin_a, cos_a = sin_a[..., np.newaxis], cos_a[..., np.newaxis]

  matrix = np.broadcast_to(
      np.eye(3, dtype=angle.dtype), (*axis_angle.shape[:-1], 3, 3)
  ).copy()

  # Form the skew-symmetric matrix for the axis.
  skew_symmetric_axis_matrix = np.zeros(
      (*axis_angle.shape[:-1], 3, 3), dtype=angle.dtype
  )
  rows = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
  cols = np.array([1, 2, 0, 2, 0, 1], dtype=np.int32)
  values = axis[..., [2, 1, 2, 0, 1, 0]] * np.array(
      [-1, 1, 1, -1, -1, 1], dtype=angle.dtype
  )
  skew_symmetric_axis_matrix[..., rows, cols] += values

  matrix += (
      sin_a * skew_symmetric_axis_matrix
      + (1 - cos_a) * skew_symmetric_axis_matrix @ skew_symmetric_axis_matrix
  )

  return matrix


def _get_batch_dims(
    identity: npt.NDArray[np.floating] | None = None,
    expression: npt.NDArray[np.floating] | None = None,
    rotations: npt.NDArray[np.floating] | None = None,
    translation: npt.NDArray[np.floating] | None = None,
) -> tuple[int, ...]:
  """Returns the leading batch dimensions [A1, ..., An] for GNM inputs.

  Args:
    identity: Identity coefficients ([A1, ..., An], I).
    expression: Expression coefficients ([A1, ..., An], E).
    rotations: Joint rotations ([A1, ..., An], J, 3),
    translation: Root-joint translation ([A1, ..., An], 3).

  Returns:
    A tuple of leading batch dimensions (A1, ..., An). If the input does not
    include batch dimensions, return empty tuple ().
  """
  if identity is not None:
    return identity.shape[:-1]
  if expression is not None:
    return expression.shape[:-1]
  if rotations is not None:
    return rotations.shape[:-2]
  if translation is not None:
    return translation.shape[:-1]
  return ()


def _check_batch_dims(
    identity: npt.NDArray[np.floating],
    expression: npt.NDArray[np.floating],
    rotations: npt.NDArray[np.floating],
    translation: npt.NDArray[np.floating],
) -> None:
  """Ensures that the leading batch dimensions of all inputs are the same."""
  if not (
      identity.shape[:-1]
      == expression.shape[:-1]
      == rotations.shape[:-2]
      == translation.shape[:-1]
  ):
    raise ValueError(
        f'Mismatched batch dimensions: ({identity.shape[:-1]},'
        f'{expression.shape[:-1]},{rotations.shape[:-2]},'
        f'{translation.shape[:-1]}).'
    )


def _check_shape(
    data: npt.NDArray[np.floating], name: str, shape: Sequence[int]
) -> None:
  """Checks that given data has the expected shape.

  Args:
    data: A NumPy array, e.g. for GNM joint angle rotation coefficients.
    name: The name that will be shown if we raise an error.
    shape: The expected shape, e.g. (5, 3).

  Raises:
    ValueError: The data does not have the expected shape.
  """
  if data.shape != shape:
    raise ValueError(f'Expecting shape {shape} for {name}. Got {data.shape}.')


def _check_batched_shape(
    data: npt.NDArray[np.floating], name: str, shape: Sequence[int]
) -> None:
  """Checks that given data has the expected shape.

  Args:
    data: A NumPy array, e.g. for GNM joint angle rotation coefficients.
    name: The name that will be shown if we raise an error.
    shape: The expected shape, e.g. (5, 3) expects shape to be (..., 5, 3).

  Raises:
    ValueError: The data does not have the expected shape.
  """
  if data.shape[-len(shape) :] != shape:
    raise ValueError(
        f'Expecting batched shape {shape} for {name}. Got {data.shape}.'
    )
