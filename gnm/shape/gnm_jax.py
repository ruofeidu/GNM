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

"""JAX implementation of the GNM model.

Usage:
  ```
  gnm = gnm_jax.from_local(version=GNMVersion.V3, variant=GNMVariant.HEAD)

  # Generate batches of parameters.
  n_batch = 5
  identity = np.random.uniform(shape=(n_batch, gnm.identity_dim))
  expression = np.random.uniform(shape=(n_batch, gnm.expression_dim))
  rotations = np.random.uniform(shape=[n_batch, gnm.num_joints, 3])
  translation = np.random.uniform(shape=(n_batch, 3))
  vertices = gnm(identity, expression, rotations, translation)
  ```

  # This module is differentiable.
  grad_func = jax.grad(
     lambda *args: jnp.square(gnm(*args)).mean(),
     argnums=np.Array([0, 1, 2, 3]))
  grads = grad_func(identity, expression, rotations, translation)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any, TYPE_CHECKING

from gnm.shape import gnm_base
from gnm.shape import gnm_landmarks
from gnm.shape.data.versions import gnm_specs
import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
_typechecker = None

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType


def _as_jnp_float32_array(data: np.ndarray) -> jt.Float[jt.Array, '...']:
  return data.astype(jnp.float32)


def _as_jnp_int32_array(data: np.ndarray) -> jt.Int[jt.Array, '...']:
  return data.astype(jnp.int32)


def _as_original(data: Any) -> Any:
  return data


@dataclasses.dataclass(init=False, kw_only=True)
class GNM(gnm_base.GNMBase):
  """JAX batched implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  This JAX implementation evaluates a batch of [A1, A2, ..., An] parameters, and
  produces a batch of vertex positions ([A1, A2, ..., An], V, 3).

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
      'template_vertex_positions': _as_jnp_float32_array,
      'template_joint_positions': _as_jnp_float32_array,
      'vertex_identity_basis': _as_jnp_float32_array,
      'joint_identity_basis': _as_jnp_float32_array,
      'expression_basis': _as_jnp_float32_array,
      'identity_names': _as_original,
      'joint_names': _as_original,
      'expression_names': _as_original,
      'joint_parent_indices': _as_original,
      'skinning_weights': _as_jnp_float32_array,
      'quads': _as_jnp_int32_array,
      'triangles': _as_jnp_int32_array,
      'quad_uvs': _as_jnp_float32_array,
      'triangle_uvs': _as_jnp_float32_array,
      'mesh_component_names': _as_original,
      'mirror_indices': _as_jnp_int32_array,
      'joint_regressor': _as_jnp_float32_array,
      'pose_correctives_regressor': _as_jnp_float32_array,
      'bone_aligned_template_joint_orientations': _as_jnp_float32_array,
      'vertex_groups': _as_jnp_float32_array,
      'vertex_group_names': _as_original,
  }

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant
  template_vertex_positions: jt.Float[jt.Array, '{self.num_vertices} 3']
  template_joint_positions: jt.Float[jt.Array, '{self.num_joints} 3']
  vertex_identity_basis: jt.Float[
      jt.Array, '{self.identity_dim} {self.num_vertices} 3'
  ]
  joint_identity_basis: jt.Float[
      jt.Array, '{self.identity_dim} {self.num_joints} 3'
  ]
  expression_basis: jt.Float[
      jt.Array, '{self.expression_dim} {self.num_vertices} 3'
  ]
  identity_names: Sequence[str]  # (I,)
  joint_names: Sequence[str]  # (J,)
  expression_names: Sequence[str]  # (E,)
  joint_parent_indices: Sequence[int]  # (J,)
  skinning_weights: jt.Float[jt.Array, '{self.num_joints} {self.num_vertices}']
  quads: jt.Int[jt.Array, 'Q 4']
  triangles: jt.Int[jt.Array, 'T 3']
  quad_uvs: jt.Float[jt.Array, 'Q 4 2']
  triangle_uvs: jt.Float[jt.Array, 'T 3 2']
  mesh_component_names: Sequence[str]  # (P,)
  mirror_indices: jt.Int[jt.Array, '{self.num_vertices}']
  joint_regressor: jt.Float[jt.Array, '{self.num_joints} {self.num_vertices}']
  pose_correctives_regressor: jt.Float[
      jt.Array, '{self.num_joints}*9 {self.num_vertices}*3'
  ]
  bone_aligned_template_joint_orientations: jt.Float[
      jt.Array, '{self.num_joints} 3 3'
  ]
  vertex_groups: jt.Float[jt.Array, 'G {self.num_vertices}']
  vertex_group_names: Sequence[str]  # (G,)

  if TYPE_CHECKING:
    _landmarks: dict[gnm_landmarks.GNMLandmarksType, Any]

  @classmethod
  def _from_model_data(
      cls,
      model_data: Mapping[str, Any],
  ) -> GNM:
    """Creates a GNM instance from a model data."""
    instance = super().__new__(cls)  # pylint: disable=no-value-for-parameter

    # Set the data fields.
    for key in model_data.keys():
      if key not in cls._FIELD_NAMES_DTYPE_MAP:
        raise ValueError(f"Field '{key}' not found in _FIELD_NAMES_DTYPE_MAP")
      extract_fn = cls._FIELD_NAMES_DTYPE_MAP[key]
      assert model_data[key] is not None, f"Field '{key}' is None"
      setattr(instance, key, extract_fn(model_data[key]))

    return instance

  def to_numpy_data_dict(self) -> dict[str, Any]:
    """Returns a dictionary of the GNM data represented as NumPy arrays."""
    result = {}
    for field in dataclasses.fields(self):
      val = getattr(self, field.name)
      if isinstance(val, jax.Array):
        val = np.array(val)
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

  @jt.jaxtyped(typechecker=_typechecker)
  def __call__(
      self,
      identity: jt.Float[jt.Array, '*N {self.identity_dim}'],
      expression: jt.Float[jt.Array, '*N {self.expression_dim}'],
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      translation: jt.Float[jt.Array, '*N 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_vertices} 3']:
    """Evaluates the GNM function for batches of parameters.

    Args:
      identity: A batch of identity coefficients.
      expression: A batch of expression coefficients.
      rotations: A batch of joint rotations.
      translation: A batch of root-joint translations.
      precision: The precision of the JAX operations.

    Returns:
      A batch of GNM vertices.

    Raises:
      ValueError: The inputs do not have a consistent batch size or
        an input has the wrong dimensionality.
    """
    # Bind pose vertex positions with identity and expression, (*N, V, 3).
    vertices = self.vertex_positions_bind_pose(
        identity, expression, precision=precision
    )

    # Bind pose joint positions in the bind pose, with identity, (*N, J, 3).
    joints = self.joint_positions_bind_pose(identity, precision=precision)

    pose_correctives = self.compute_pose_correctives(
        rotations, precision=precision
    )
    vertices += pose_correctives

    return self.apply_linear_blend_skinning(
        vertices, joints, rotations, translation, precision=precision
    )

  @jt.jaxtyped(typechecker=_typechecker)
  def apply_linear_blend_skinning(
      self,
      local_vertices: jt.Float[jt.Array, '*N {self.num_vertices} 3'],
      joint_positions: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      translation: jt.Float[jt.Array, '*N 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_vertices} 3']:
    """Computes vertices in the mesh pose, with world-space transforms applied.

    Args:
      local_vertices: A batch of vertices in the bind pose.
      joint_positions: A batch of joint positions in the bind pose.
      rotations: A batch of joint rotations.
      translation: A batch of root-joint translations.
      precision: The precision of the JAX operations.

    Returns:
      A batch of Mesh vertices.
    """
    batch_dims = local_vertices.shape[:-2]

    # The local-to-world transforms of each joint, after posing, (N, J, 4, 4).
    joint_transforms_world = self.joint_transforms_world(
        joint_positions, rotations, translation, precision=precision
    )

    # Skinning requires we compute the transform from the bind pose to the final
    # pose for each joint. To do this, we pre-apply the inverse joint-to-world
    # transforms for the bind pose. Fortunately, our convention is that joints
    # have zero rotation in the bind pose, so these transforms are simply
    # translation matrices.
    deltas = jnp.einsum(
        '...jik,...jk->...ji',
        joint_transforms_world[..., :3, :3],
        joint_positions,
        precision=precision,
    )
    deltas = deltas.reshape(*batch_dims, self.num_joints, 3, 1)
    offset = jnp.concatenate(
        [
            jnp.zeros((*batch_dims, self.num_joints, 3, 3), dtype=deltas.dtype),
            deltas,
        ],
        axis=-1,
    )
    offset = jnp.concatenate(
        [
            offset,
            jnp.zeros((*batch_dims, self.num_joints, 1, 4), dtype=deltas.dtype),
        ],
        axis=-2,
    )
    joint_transforms = joint_transforms_world - offset

    # Convert the vertices to homogeneous coordinates.
    vertices_h = jnp.concatenate(
        [local_vertices, jnp.ones_like(local_vertices[..., :1])], -1
    )

    # Perform Linear Blend Skinning: accumulate the result of each joint's
    # posing transform on the mesh according to skinning weights.
    vertices_skinned = jnp.einsum(
        'jv,...jmn,...vn->...vm',
        self.skinning_weights,
        joint_transforms,
        vertices_h,
        precision=precision,
    )[..., :3]

    return vertices_skinned

  @jt.jaxtyped(typechecker=_typechecker)
  def vertices_and_landmarks(
      self,
      landmarks_type: gnm_landmarks.GNMLandmarksType,
      identity: jt.Float[jt.Array, '*N {self.identity_dim}'],
      expression: jt.Float[jt.Array, '*N {self.expression_dim}'],
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      translation: jt.Float[jt.Array, '*N 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> tuple[
      jt.Float[jt.Array, '*N {self.num_vertices} 3'],
      jt.Float[jt.Array, '*N _ 3'],
  ]:
    """Evaluates the GNM mesh function and extracts 3D landmarks.

    Args:
      landmarks_type: The type of landmarks to extract.
      identity: A batch of identity coefficients.
      expression: A batch of expression coefficients.
      rotations: A batch of joint rotations.
      translation: A batch of root-joint translations.
      precision: The precision of the JAX operations.

    Returns:
      A tuple of (vertices, landmarks).
    """
    gnm_landmarks.check_body_part_compatibility(landmarks_type, self.body_part)
    if not hasattr(self, '_landmarks'):
      self._landmarks = {}
    if landmarks_type not in self._landmarks:
      config = gnm_landmarks.load_landmarks(landmarks_type)
      self._landmarks[landmarks_type] = (
          jnp.array(config.indices, dtype=jnp.int32),
          jnp.array(config.weights, dtype=jnp.float32),
      )
    indices, weights = self._landmarks[landmarks_type]
    weights = weights.astype(identity.dtype)

    vertices = self(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
        precision=precision,
    )
    face_vertices = vertices[..., indices, :]
    landmarks = jnp.sum(face_vertices * weights[..., None], axis=-2)
    return vertices, landmarks

  @jt.jaxtyped(typechecker=_typechecker)
  def vertex_positions_bind_pose(
      self,
      identity: jt.Float[jt.Array, '*N {self.identity_dim}'],
      expression: jt.Float[jt.Array, '*N {self.expression_dim}'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_vertices} 3']:
    """Computes vertices in the bind pose, with identity and expression applied.

    Args:
      identity: A batch of identity coefficients.
      expression: A batch of expression coefficients.
      precision: The precision of the JAX operations.

    Returns:
      A batch of vertex positions in the bind pose.
    """
    # For formatting.
    vertex_identity_basis = self.vertex_identity_basis
    expression_basis = self.expression_basis

    # Apply linear identity and expression bases to vertices in bind pose.
    identity_deltas = jnp.einsum(
        '...i,ijk->...jk',
        identity,
        vertex_identity_basis,
        precision=precision,
    )
    expression_deltas = jnp.einsum(
        '...i,ijk->...jk',
        expression,
        expression_basis,
        precision=precision,
    )

    return self.template_vertex_positions + identity_deltas + expression_deltas

  @jt.jaxtyped(typechecker=_typechecker)
  def compute_pose_correctives(
      self,
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_vertices} 3']:
    """Apply pose correctives.

    If self.pose_correctives_regressor is not None, return computed pose
    correctives. Otherwise, return zero tensor.

    Args:
      rotations: a batch of rotation vectors.
      precision: The precision of the JAX operations.

    Returns:
      A batch of pose correctives as vertex offsets.
    """
    batch_dims = rotations.shape[:-2]

    if self.pose_correctives_regressor is None:
      return jnp.zeros(
          (*batch_dims, self.num_vertices, 3), dtype=rotations.dtype
      )

    rotation_matrices = axis_angle_to_rotation_matrix(
        rotations, precision=precision
    )
    rotation_matrices = rotation_matrices.reshape(
        *batch_dims, self.num_joints, 3, 3
    )

    pose_features = (
        rotation_matrices - jnp.eye(3, dtype=rotations.dtype)[None, None]
    )

    pose_features = pose_features.reshape([*batch_dims, self.num_joints * 9])
    pose_deltas_flattened = jnp.einsum(
        'jv,...j->...v',
        self.pose_correctives_regressor,
        pose_features,
        precision=precision,
    )
    return pose_deltas_flattened.reshape([*batch_dims, self.num_vertices, 3])

  @jt.jaxtyped(typechecker=_typechecker)
  def joint_positions_bind_pose(
      self,
      identity: jt.Float[jt.Array, '*N {self.identity_dim}'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_joints} 3']:
    """Joint positions in the bind pose, with identity basis applied.

    Args:
      identity: A batch of identity coefficients.
      precision: The precision of the JAX operations.

    Returns:
      A batch of joint positions in the bind pose.
    """
    deltas = jnp.einsum(
        '...i,ijk->...jk',
        identity,
        self.joint_identity_basis,
        precision=precision,
    )
    return self.template_joint_positions + deltas

  def joint_transforms_world(
      self,
      joints: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      translation: jt.Float[jt.Array, '*N 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_joints} 4 4']:
    """Gets the world-space transforms of each skeletal joint.

    Args:
      joints: Joint locations in the bind pose.
      rotations: Per-joint rotations as axis-angle vectors.
      translation: The translation of the root joint.
      precision: The precision of the JAX operations.

    Returns:
      Local-to-world transforms of each skeletal joint.
    """
    # The posing transforms in joint-local space for each joint.
    batch_dims = rotations.shape[:-2]  # 'N' and 'J'.

    # Build 3x3 local rotation matrices for each joint.
    rotation_matrices = axis_angle_to_rotation_matrix(
        rotations, precision=precision
    )

    # Joint offsets for all joints except the root, (N, J-1, 3).
    joint_offsets = (
        joints[..., 1:, :] - joints[..., self.joint_parent_indices[1:], :]
    )

    # Prepend with the joint offsets for the root joints.
    root_joints = (translation + joints[..., 0, :]).reshape(*batch_dims, 1, 3)
    joint_offsets = jnp.concatenate([root_joints, joint_offsets], axis=-2)

    # Include translation to build 4x4 transforms in parent space.
    right_columns = joint_offsets.reshape(*batch_dims, self.num_joints, 3, 1)
    bottom_rows = jnp.broadcast_to(
        jnp.array([0, 0, 0, 1], dtype=rotations.dtype),
        [*batch_dims, self.num_joints, 1, 4],
    )
    transforms_3x4 = jnp.concatenate(
        [rotation_matrices, right_columns], axis=-1
    )
    transforms_parent = jnp.concatenate([transforms_3x4, bottom_rows], axis=-2)

    # Traverse the skeleton to compute posing transforms in world-space.
    transforms_world = [transforms_parent[..., 0, :, :]]
    for i, parent_index in enumerate(self.joint_parent_indices[1:], start=1):
      transform_world = (
          transforms_world[parent_index] @ transforms_parent[..., i, :, :]
      )
      transforms_world.append(transform_world)

    output = jnp.stack(transforms_world, axis=-3)
    return output

  @jt.jaxtyped(typechecker=_typechecker)
  def get_posed_joint_transforms(
      self,
      identity: jt.Float[jt.Array, '*N {self.identity_dim}'],
      rotations: jt.Float[jt.Array, '*N {self.num_joints} 3'],
      translation: jt.Float[jt.Array, '*N 3'],
      *,
      precision: jax.lax.PrecisionLike = 'float32',
  ) -> jt.Float[jt.Array, '*N {self.num_joints} 4 4']:
    """Computes the local-to-world transformation for every joint.

    Args:
      identity: Identity coefficients.
      rotations: Joint rotations.
      translation: Root-joint translation.
      precision: The precision of the JAX operations.

    Returns:
      Local-to-world joint transformations.

    Raises:
      ValueError if an input argument has the wrong format.
    """
    # Joint positions in the bind pose, with identity model applied, (J, 3).
    joints = self.joint_positions_bind_pose(identity, precision=precision)

    # The local-to-world transforms of each joint, after posing, (J, 4, 4).
    return self.joint_transforms_world(
        joints, rotations, translation, precision=precision
    )


def axis_angle_to_rotation_matrix(
    axis_angle: jt.Float[jt.Array, '*N 3'],
    *,
    precision: jax.lax.PrecisionLike = 'float32',
) -> jt.Float[jt.Array, '*N 3 3']:
  """Builds a 3x3 rotation matrix from an axis-angle vector.

  The rotation matrix is computed using Rodrigues' rotation formula.  See here
  for more information:
  https://en.wikipedia.org/wiki/Rodrigues_rotation_formula

  Args:
    axis_angle: The axis of rotation is the direction of this vector and its
      norm is the angle of rotation (in radians), with shape (3,).
    precision: The precision of the rotation matrix computation to use for
      multiplications.

  Returns:
    A 3x3 rotation matrix.
  """
  epsilon = jnp.finfo(axis_angle.dtype).eps
  norm_squared = jnp.sum(jnp.square(axis_angle), axis=-1, keepdims=True)
  angle = jnp.sqrt(jnp.maximum(norm_squared, epsilon))
  axis = axis_angle / angle

  sin_a, cos_a = jnp.sin(angle), jnp.cos(angle)
  sin_a, cos_a = sin_a[..., jnp.newaxis], cos_a[..., jnp.newaxis]

  matrix = jnp.broadcast_to(
      jnp.eye(3, dtype=angle.dtype), (*axis_angle.shape[:-1], 3, 3)
  )

  # Form the skew-symmetric matrix for the axis.
  skew_symmetric_axis_matrix = jnp.zeros(
      (*axis_angle.shape[:-1], 3, 3), dtype=angle.dtype
  )
  rows = jnp.array([0, 0, 1, 1, 2, 2])
  cols = jnp.array([1, 2, 0, 2, 0, 1])
  values = axis[..., [2, 1, 2, 0, 1, 0]] * jnp.array(
      [-1, 1, 1, -1, -1, 1], dtype=angle.dtype
  )

  skew_symmetric_axis_matrix = skew_symmetric_axis_matrix.at[
      ..., rows, cols
  ].set(values, unique_indices=True)

  matrix += sin_a * skew_symmetric_axis_matrix + (1 - cos_a) * jnp.matmul(
      skew_symmetric_axis_matrix,
      skew_symmetric_axis_matrix,
      precision=precision,
  )

  return matrix
