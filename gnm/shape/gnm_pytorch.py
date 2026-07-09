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

"""PyTorch implementation of the GNM face model.

Usage:
  ```
  gnm = gnm_pytorch.from_local(version=GNMVersion.V3, variant=GNMVariant.HEAD)

  # Generate batches of parameters.
  n_batch = 5
  identity = torch.rand(size=(n_batch, gnm.identity_dim))
  expression = torch.rand(size=(n_batch, gnm.expression_dim))
  rotations = torch.rand(size=[n_batch, gnm.num_joints, 3])
  translation = torch.rand(size=(n_batch, 3))

  vertices = gnm(identity, expression, rotations, translation)
  ```
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, TYPE_CHECKING

from gnm.shape import gnm_base
from gnm.shape import gnm_landmarks
from gnm.shape.data.versions import gnm_specs
import numpy as np
import torch

_EPSILON = 1.0e-09


GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType


def _as_torch_float32_tensor(data: np.ndarray) -> torch.Tensor:
  return torch.tensor(data, dtype=torch.float32)


def _as_torch_int32_tensor(data: np.ndarray) -> torch.Tensor:
  return torch.tensor(data, dtype=torch.int32)


def _as_original(data: Any) -> Any:
  return data


@dataclasses.dataclass(init=False, kw_only=True)
class GNM(gnm_base.GNMBase, torch.nn.Module):
  """PyTorch batched implementation of the GNM parametric face model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  This PyTorch implementation evaluates a batch of N parameters, and produces
  a batch of vertex positions (N, V, 3).

  The GNM class also surfaces useful data for down-stream users, e.g. the
  names of each expression dimension, and the topology of the mesh.

  Shape dimensions are denoted:
  * N: Size of batch.
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.
  * Q: The number of quads in the mesh topology.
  * T: The number of triangles, in a triangulated version of the mesh topology.
  * G: Number of vertex groups.

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
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
  """

  _FIELD_NAMES_DTYPE_MAP = {
      'version': _as_original,
      'variant': _as_original,
      'template_vertex_positions': _as_torch_float32_tensor,
      'template_joint_positions': _as_torch_float32_tensor,
      'vertex_identity_basis': _as_torch_float32_tensor,
      'joint_identity_basis': _as_torch_float32_tensor,
      'expression_basis': _as_torch_float32_tensor,
      'identity_names': _as_original,
      'joint_names': _as_original,
      'expression_names': _as_original,
      'joint_parent_indices': _as_original,
      'skinning_weights': _as_torch_float32_tensor,
      'quads': _as_torch_int32_tensor,
      'triangles': _as_torch_int32_tensor,
      'quad_uvs': _as_torch_float32_tensor,
      'triangle_uvs': _as_torch_float32_tensor,
      'mesh_component_names': _as_original,
      'mirror_indices': _as_torch_int32_tensor,
      'joint_regressor': _as_torch_float32_tensor,
      'pose_correctives_regressor': _as_torch_float32_tensor,
      'bone_aligned_template_joint_orientations': _as_torch_float32_tensor,
      'vertex_groups': _as_torch_float32_tensor,
      'vertex_group_names': _as_original,
  }

  # Attributes must be set to a value, otherwise they cannot be found by
  # GNMBase, due to the interaction with torch.nn.Module.
  version: gnm_specs.GNMVersion = ...
  variant: gnm_specs.GNMVariant = ...
  template_vertex_positions: torch.Tensor = ...
  template_joint_positions: torch.Tensor = ...
  vertex_identity_basis: torch.Tensor = ...
  joint_identity_basis: torch.Tensor = ...
  expression_basis: torch.Tensor = ...
  identity_names: Sequence[str] = ...
  joint_names: Sequence[str] = ...
  expression_names: Sequence[str] = ...
  joint_parent_indices: Sequence[int] = ...
  skinning_weights: torch.Tensor = ...
  quads: torch.Tensor = ...
  triangles: torch.Tensor = ...
  quad_uvs: torch.Tensor = ...
  triangle_uvs: torch.Tensor = ...
  mesh_component_names: Sequence[str] = ...
  mirror_indices: torch.Tensor = ...
  joint_regressor: torch.Tensor = ...
  pose_correctives_regressor: torch.Tensor = ...
  bone_aligned_template_joint_orientations: torch.Tensor = ...
  vertex_groups: torch.Tensor = ...
  vertex_group_names: Sequence[str] = ...

  if TYPE_CHECKING:
    _landmarks: dict[gnm_landmarks.GNMLandmarksType, Any]

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a GNM instance from a model data."""
    instance = super().__new__(cls)  # pylint: disable=no-value-for-parameter
    super(GNM, instance).__init__()

    # Set the data fields.
    for key in model_data.keys():
      if key not in cls._FIELD_NAMES_DTYPE_MAP:
        raise ValueError(f"Field '{key}' not found in _FIELD_NAMES_DTYPE_MAP")
      extract_fn = cls._FIELD_NAMES_DTYPE_MAP[key]
      assert model_data[key] is not None, f"Field '{key}' is None"
      field = extract_fn(model_data[key])
      if isinstance(field, torch.Tensor):
        # We have to delete the attribute first, because torch.nn.Module
        # registers buffers based on the attribute name. If the attribute is
        # already set, torch.nn.Module will refuse to register the buffer.
        if hasattr(cls, key):
          delattr(cls, key)
        instance.register_buffer(key, field, persistent=False)
      else:
        setattr(instance, key, field)

    return instance

  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    result = {}
    for field in dataclasses.fields(self):
      val = getattr(self, field.name)
      if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
      result[field.name] = val
    return result

  @property
  def num_vertices(self) -> int:
    """The number of vertices in the mesh (V)."""
    return self.template_vertex_positions.shape[0]

  @property
  def num_joints(self) -> int:
    """The number of joints in the skeleton (J)."""
    return self.skinning_weights.shape[0]

  @property
  def identity_dim(self) -> int:
    """The dimensionality of the linear identity basis (I)."""
    return self.vertex_identity_basis.shape[0]

  @property
  def expression_dim(self) -> int:
    """The dimensionality of the linear expression basis (E)."""
    return self.expression_basis.shape[0]

  def __call__(
      self,
      identity: torch.Tensor | None = None,
      expression: torch.Tensor | None = None,
      rotations: torch.Tensor | None = None,
      translation: torch.Tensor | None = None,
  ):
    """Evaluates the GNM mesh-generating function.

    Input parameters have optional batch dimensions [A1, ..., An]. Parameters
    may be omitted - in this case default values will be substituted.



    Args:
      identity: Identity coefficients ([A1, ..., An], I).
      expression: Expression coefficients ([A1, ..., An], E,).
      rotations: Joint rotations ([A1, ..., An], J, 3),
      translation: Root-joint translation ([A1, ..., An], 3,).

    Returns:
      Mesh vertices ([A1, ..., An], V, 3).

    Raises:
      ValueError if an input argument has the wrong format.
    """

    batch_dims = _get_batch_dims(identity, expression, rotations, translation)

    device = self.template_vertex_positions.device

    # Fill in missing parameter values with zeros.
    if identity is None:
      identity = torch.zeros(
          list(batch_dims) + [self.identity_dim], device=device
      )
    if expression is None:
      expression = torch.zeros(
          list(batch_dims) + [self.expression_dim], device=device
      )
    if rotations is None:
      rotations = torch.zeros(
          list(batch_dims) + [self.num_joints, 3], device=device
      )
    if translation is None:
      translation = torch.zeros(list(batch_dims) + [3], device=device)

    # Flatten inputs to have a single batch dimension of A1 * ... An = N.
    batch_size = int(np.prod(batch_dims))
    identity_flat = torch.reshape(identity, [batch_size, self.identity_dim])
    expression_flat = torch.reshape(
        expression, [batch_size, self.expression_dim]
    )
    rotations_flat = torch.reshape(rotations, [batch_size, self.num_joints, 3])
    translation_flat = torch.reshape(translation, [batch_size, 3])

    vertices = self._forward(
        identity_flat, expression_flat, rotations_flat, translation_flat
    )

    # Reshape computed vertices to match original batch dimensions.
    output_shape = list(batch_dims) + [self.num_vertices, 3]
    return torch.reshape(vertices, output_shape)

  def vertices_and_landmarks(
      self,
      landmarks_type: gnm_landmarks.GNMLandmarksType,
      identity: torch.Tensor | None = None,
      expression: torch.Tensor | None = None,
      rotations: torch.Tensor | None = None,
      translation: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates the GNM mesh function and extracts 3D landmarks.

    Args:
      landmarks_type: The type of landmarks to extract.
      identity: Identity coefficients ([A1, ..., An], I).
      expression: Expression coefficients ([A1, ..., An], E).
      rotations: Joint rotations ([A1, ..., An], J, 3).
      translation: Root-joint translation ([A1, ..., An], 3).

    Returns:
      A tuple of (vertices, landmarks).
    """
    gnm_landmarks.check_body_part_compatibility(landmarks_type, self.body_part)
    if not hasattr(self, '_landmarks'):
      self._landmarks = {}
    if landmarks_type not in self._landmarks:
      config = gnm_landmarks.load_landmarks(landmarks_type)
      self._landmarks[landmarks_type] = (
          torch.from_numpy(config.indices).long(),
          torch.from_numpy(config.weights).float(),
      )
    indices, weights = self._landmarks[landmarks_type]

    vertices = self(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    indices = indices.to(device=vertices.device)
    weights = weights.to(device=vertices.device, dtype=vertices.dtype)

    face_vertices = vertices[..., indices, :]
    landmarks = (face_vertices * weights.unsqueeze(-1)).sum(dim=-2)
    return vertices, landmarks

  def _forward(
      self,
      identity: torch.Tensor,
      expression: torch.Tensor,
      rotations: torch.Tensor,
      translation: torch.Tensor,
  ) -> torch.Tensor:
    """Evaluates the GNM function for a flat batch of N parameters.

    Args:
      identity: A batch of identity coefficients (N, I,).
      expression: A batch of expression coefficients (N, E,).
      rotations: A batch of joint rotations (N, J, 3),
      translation: A batch of root-joint translations (N, 3,).

    Returns:
      A batch of Mesh vertices (N, V, 3).

    Raises:
      ValueError: The inputs do not have a consistent batch size.
      InvalidArgumentError: An input has the wrong dimensionality.
    """
    identity = _ensure_shape(identity, [None, self.identity_dim])
    expression = _ensure_shape(expression, [None, self.expression_dim])
    rotations = _ensure_shape(rotations, [None, self.num_joints, 3])
    translation = _ensure_shape(translation, [None, 3])

    # Bind pose vertex positions with identity and expression, (N, V, 3).
    vertices = self.vertex_positions_bind_pose(identity, expression)

    # Bind pose joint positions in the bind pose, with identity, (N, J, 3).
    joints = self.joint_positions_bind_pose(identity)

    # Apply pose correctives:
    pose_correctives = self.compute_pose_correctives(rotations)
    vertices += pose_correctives

    vertices_skinned = self.vertex_positions_world(
        vertices, joints, rotations, translation
    )

    return vertices_skinned

  def vertex_positions_world(
      self,
      vertices: torch.Tensor,
      joints: torch.Tensor,
      rotations: torch.Tensor,
      translation: torch.Tensor,
  ) -> torch.Tensor:
    """Applies linear blend skinning to the input GNM vertices.

    Args:
      vertices: The vertices in the bind-pose, (N, V, 3).
      joints: The joints in the bind-pose, (N, J, 3).
      rotations: GNM joint rotations in axis-angle format, (N, J, 3).
      translation: GNM root translation, (N,  3).

    Returns:
      The posed vertices after applying linear blend skinning about the input
        joints, (N, V, 3).
    """
    # The local-to-world transforms of each joint, after posing, (N, J, 4, 4).
    joint_transforms_world = self.joint_transforms_world(
        joints, rotations, translation
    )
    n_batch = vertices.shape[0]
    n_joints = joints.shape[1]

    # Skinning requires we compute the transform from the bind pose to the final
    # pose for each joint. To do this, we pre-apply the inverse joint-to-world
    # transforms for the bind pose. Fortunately, our convention is that joints
    # have zero rotation in the bind pose, so these transforms are simply
    # translation matrices.
    deltas = torch.einsum(
        'njik,njk->nji', joint_transforms_world[:, :, :3, :3], joints
    )
    deltas = torch.reshape(deltas, (n_batch, n_joints, 3, 1))
    offset = torch.concat(
        [torch.zeros((n_batch, n_joints, 3, 3), device=deltas.device), deltas],
        dim=-1,
    )
    offset = torch.concat(
        [offset, torch.zeros((n_batch, n_joints, 1, 4), device=offset.device)],
        dim=-2,
    )
    joint_transforms = joint_transforms_world - offset

    # Convert the vertices to homogeneous coordinates.
    vertices_h = torch.concat(
        [vertices, torch.ones_like(vertices[:, :, :1])], -1
    )

    # Perform Linear Blend Skinning: accumulate the result of each joint's
    # posing transform on the mesh according to skinning weights.
    vertices_skinned = torch.einsum(
        'jv,bjmn,bvn->bvm', self.skinning_weights, joint_transforms, vertices_h
    )[..., :3]

    return vertices_skinned

  def vertex_positions_bind_pose(
      self,
      identity: torch.Tensor,
      expression: torch.Tensor,
  ) -> torch.Tensor:
    """Computes vertices in the bind pose, with identity and expression applied.

    Args:
      identity: A batch of identity coefficients (N, I,).
      expression: A batch of expression coefficients (N, E,).

    Returns:
      A batch of vertex positions in the bind pose, (N, V, 3).
    """
    identity = _ensure_shape(identity, [None, self.identity_dim])
    expression = _ensure_shape(expression, [None, self.expression_dim])

    # For formatting.
    vertex_identity_basis = self.vertex_identity_basis
    expression_basis = self.expression_basis

    # Apply linear identity and expression bases to vertices in bind pose.
    identity_deltas = torch.einsum(
        'bi,ijk->bjk', identity, vertex_identity_basis
    )
    expression_deltas = torch.einsum(
        'bi,ijk->bjk', expression, expression_basis
    )

    return self.template_vertex_positions + identity_deltas + expression_deltas

  def compute_pose_correctives(
      self,
      rotations: torch.Tensor,
  ) -> torch.Tensor:
    """Apply pose correctives.

    If self.pose_correctives_regressor is not None, return computed pose
    correctives. Otherwise, return zero tensor.

    Args:
      rotations: a batch of rotation vectors, (N, J, 3).

    Returns:
      A batch of pose correctives as vertex offsets, (N, V, 3).
    """
    _ensure_shape(rotations, [None, self.num_joints, 3])

    batch_size = rotations.shape[0]
    if self.pose_correctives_regressor is None:
      zero_tensor = torch.zeros_like(
          self.template_vertex_positions[torch.newaxis, ...]
      )
      return torch.repeat_interleave(zero_tensor, batch_size, dim=0)

    pose_feature = _rotation_matrices(rotations)
    zero_pose_feature = torch.eye(
        3, dtype=rotations.dtype, device=pose_feature.device
    )
    pose_feature -= zero_pose_feature
    pose_feature = torch.reshape(pose_feature, [batch_size, -1])

    pose_deltas_flattened = pose_feature @ self.pose_correctives_regressor
    pose_deltas = torch.reshape(
        pose_deltas_flattened, [batch_size, self.num_vertices, 3]
    )

    return pose_deltas

  def joint_positions_bind_pose(self, identity: torch.Tensor) -> torch.Tensor:
    """Joint positions in the bind pose, with identity basis applied, (N, J, 3).

    Args:
      identity: A batch of identity coefficients (N, I,).

    Returns:
      A batch of joint positions in the bind pose, (N, J, 3).
    """
    identity = _ensure_shape(identity, [None, self.identity_dim])
    deltas = torch.einsum('bi,ijk->bjk', identity, self.joint_identity_basis)
    return self.template_joint_positions + deltas

  def joint_transforms_world(
      self,
      joints: torch.Tensor,
      rotations: torch.Tensor,
      translation: torch.Tensor,
  ) -> torch.Tensor:
    """Gets the world-space transforms of each skeletal joint.

    Args:
      joints: Joint locations in the bind pose, (N, J, 3).
      rotations: Per-joint rotations as axis-angle vectors, (N, J, 3).
      translation: The translation of the root joint (N, 3,).

    Returns:
      Local-to-world transforms of each skeletal joint, (N, J, 4, 4).
    """

    n_batch = rotations.shape[0]
    n_joints = self.num_joints

    # Build 3x3 local rotation matrices for each joint, (N, J, 3, 3).
    rotation_matrices = _rotation_matrices(rotations)

    # Joint offsets for all joints except the root, (N, J-1, 3).
    joint_offsets = joints[:, 1:] - joints[:, self.joint_parent_indices[1:]]

    # Prepend with the joint offsets for the root joints, (N, J, 3).
    root_joints = torch.reshape(translation + joints[:, 0], (n_batch, 1, 3))
    joint_offsets = torch.concat([root_joints, joint_offsets], dim=1)

    # Include translation to build 4x4 transforms in parent space, (N, J, 4, 4).
    right_columns = torch.reshape(joint_offsets, (n_batch, n_joints, 3, 1))
    bottom_row = torch.tensor(
        [[[[0, 0, 0, 1]]]], dtype=torch.float32, device=joints.device
    )
    bottom_rows = torch.tile(bottom_row, [n_batch, n_joints, 1, 1])
    transforms_3x4 = torch.concat([rotation_matrices, right_columns], dim=3)
    transforms_parent = torch.concat([transforms_3x4, bottom_rows], dim=2)

    # Traverse the skeleton to compute posing transforms in world-space.
    transforms_world = [transforms_parent[:, 0]]
    for i in range(1, n_joints):
      parent_index = self.joint_parent_indices[i]
      transform_world = transforms_world[parent_index] @ transforms_parent[:, i]
      transforms_world.append(transform_world)

    return torch.stack(transforms_world, dim=1)

  def get_posed_joint_transforms(
      self,
      identity: torch.Tensor,
      rotations: torch.Tensor,
      translation: torch.Tensor,
  ) -> torch.Tensor:
    """Computes the local-to-world transformation for every joint.

    Args:
      identity: Identity coefficients (I,).
      rotations: Joint rotations (J, 3),
      translation: Root-joint translation (3,).

    Returns:
      Local-to-world joint transformations (J, 4, 4).

    Raises:
      ValueError if an input argument has the wrong format.
    """
    identity = _ensure_shape(identity, [None, self.identity_dim])
    rotations = _ensure_shape(rotations, [None, self.num_joints, 3])
    translation = _ensure_shape(translation, [None, 3])

    # Joint positions in the bind pose, with identity model applied, (J, 3).
    joints = self.joint_positions_bind_pose(identity)

    # The local-to-world transforms of each joint, after posing, (J, 4, 4).
    return self.joint_transforms_world(joints, rotations, translation)

  def prune_vertices(self, keep_vertices: torch.Tensor) -> None:
    """Prune GNM model's vertices in-place for faster evaluation.

    All GNM quads and triangles are removed.

    Args:
      keep_vertices: Indices of vertices to be kept. Shaped (V*,) of type int32
        or int64. Values must be in range [0, self.num_vertices).
    """
    # Store the original number of vertices.
    num_vertices = self.num_vertices

    with torch.no_grad():
      self.template_vertex_positions = self.template_vertex_positions[
          keep_vertices
      ]
      self.vertex_identity_basis = self.vertex_identity_basis[:, keep_vertices]
      self.expression_basis = self.expression_basis[:, keep_vertices]
      self.skinning_weights = self.skinning_weights[:, keep_vertices]

      self.quads = torch.empty(
          size=(0, 4), dtype=self.quads.dtype, device=self.quads.device
      )
      self.triangles = torch.empty(
          size=(0, 3),
          dtype=self.triangles.dtype,
          device=self.triangles.device,
      )

      if self.pose_correctives_regressor is not None:
        pose_correctives = torch.reshape(
            self.pose_correctives_regressor,
            [self.num_joints, -1, num_vertices, 3],
        )
        pose_correctives = pose_correctives[..., keep_vertices, :]

        self.pose_correctives_regressor = torch.reshape(
            pose_correctives, [-1, len(keep_vertices) * 3]
        )


def _rotation_matrices(axis_angles: torch.Tensor) -> torch.Tensor:
  """Builds 3x3 rotation matrices axis-angle vectors.

  The rotation matrix is computed using Rodrigues' rotation formula. See here
  for more information:
  https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula


  Args:
    axis_angles: A list axis-angle vectors. Shaped (N, 3).

  Returns:
    3x3 rotation matrices, (N, 3, 3).
  """
  # pylint: disable=not-callable
  axis_angles = torch.where(
      torch.linalg.norm(axis_angles, dim=-1, keepdims=True) <= _EPSILON,
      axis_angles + _EPSILON,
      axis_angles,
  )
  # pylint: enable=not-callable
  angle = torch.sqrt(torch.sum(torch.square(axis_angles), dim=-1, keepdim=True))
  axis = axis_angles / (angle + _EPSILON)

  sin_axis = torch.sin(angle) * axis
  cos_angle = torch.cos(angle)
  cos1_axis = (1.0 - cos_angle) * axis
  _, axis_y, axis_z = torch.unbind(axis, dim=-1)
  cos1_axis_x, cos1_axis_y, _ = torch.unbind(cos1_axis, dim=-1)
  sin_axis_x, sin_axis_y, sin_axis_z = torch.unbind(sin_axis, dim=-1)
  tmp = cos1_axis_x * axis_y
  m01 = tmp - sin_axis_z
  m10 = tmp + sin_axis_z
  tmp = cos1_axis_x * axis_z
  m02 = tmp + sin_axis_y
  m20 = tmp - sin_axis_y
  tmp = cos1_axis_y * axis_z
  m12 = tmp - sin_axis_x
  m21 = tmp + sin_axis_x
  diag = cos1_axis * axis + cos_angle
  diag_x, diag_y, diag_z = torch.unbind(diag, dim=-1)
  matrix = torch.stack(
      (diag_x, m01, m02, m10, diag_y, m12, m20, m21, diag_z), dim=-1
  )
  output_shape = list(axis.shape[:-1]) + [3, 3]
  return torch.reshape(matrix, shape=output_shape)


def _get_batch_dims(
    identity: torch.Tensor | None = None,
    expression: torch.Tensor | None = None,
    rotations: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
) -> torch.Size:
  """Returns the leading batch dimensions [A1, ..., An] for GNM inputs.

  Args:
    identity: Identity coefficients ([A1, ..., An], I).
    expression: Expression coefficients ([A1, ..., An], E).
    rotations: Joint rotations ([A1, ..., An], J, 3),
    translation: Root-joint translation ([A1, ..., An], 3).

  Returns:
    An int32 torch.Tensor of leading batch dimensions (A1, ..., An). If the
    input
    does not include batch dimensions, return empty tuple ().
  """
  if identity is not None:
    return identity.shape[:-1]
  if expression is not None:
    return expression.shape[:-1]
  if rotations is not None:
    return rotations.shape[:-2]
  if translation is not None:
    return translation.shape[:-1]
  return torch.Size()


def _ensure_shape(tensor: torch.Tensor, shape: Sequence[int]):
  """Emulates Tensorflow ensure_shape."""
  if len(tensor.shape) != len(shape):
    raise RuntimeError(
        f'Shape of tensor {tensor.shape} is not compatible with expected shape'
        f' {shape}.'
    )
  for expected, actual in zip(shape, tensor.shape):
    if expected is not None and actual != expected:
      raise RuntimeError(
          f'Shape of tensor {tensor.shape} is not compatible with expected'
          f' shape {shape}.'
      )
  return tensor
