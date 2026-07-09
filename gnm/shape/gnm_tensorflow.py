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

"""TensorFlow implementation of the GNM model.

Usage:
  ```
  gnm = gnm_tensorflow.from_local(
      version=GNMVersion.V3,
      variant=GNMVariant.HEAD
  )

  # Generate batches of parameters.
  n_batch = 5
  identity = tf.random.uniform(shape=(n_batch, gnm.identity_dim))
  expression = tf.random.uniform(shape=(n_batch, gnm.expression_dim))
  rotations = tf.random.uniform(shape=[n_batch, gnm.num_joints, 3])
  translation = tf.random.uniform(shape=(n_batch, 3))

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
import tensorflow as tf

GNMVersion = gnm_specs.GNMVersion
GNMMajorVersion = gnm_specs.GNMMajorVersion
GNMVariant = gnm_specs.GNMVariant
GNMBodyPart = gnm_specs.GNMBodyPart
GNMLandmarksType = gnm_landmarks.GNMLandmarksType

_EPSILON = 1.0e-09
_PACKAGE_NAME = 'GNM'

TensorOrVariable = tf.Tensor | tf.Variable


class InvalidShapeError(Exception):
  """Raised when an input has the wrong shape."""


def _as_tf_constant_float32(data: np.ndarray) -> tf.Tensor:
  return tf.constant(data, tf.float32)


def _as_tf_constant_int32(data: np.ndarray) -> tf.Tensor:
  return tf.constant(data, tf.int32)


def _as_original(data: Any) -> Any:
  return data


@dataclasses.dataclass(init=False, kw_only=True)
class GNM(gnm_base.GNMBase):
  """TensorFlow batched implementation of the GNM parametric model.

  GNM is a mesh-generating function. Given identity, expression, joint
  rotation, and translation parameters, it produces vertices of a mesh.

  This TensorFlow implementation evaluates a batch of N parameters, and produces
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
    joint_parent_indices: Parent's index for each joint in the skeleton, (J).
    template_vertex_positions: Vertex positions in the template mesh, (V, 3).
    template_joint_positions: Joint positions in the template GNM, (J, 3).
    identity_names: The name of each identity in the identity basis.
    vertex_identity_basis: The vertex identity basis (I, V, 3).
    joint_identity_basis: The joint identity basis of the model, (I, J, 3).
    expression_names: The name of each expression in the expression basis.
    expression_basis: The vertex expression basis, aka blend-shapes, (E, V, 3).
    joint_names: The name of each joint in the skeleton.
    skinning_weights: The model's skinning weights, (J, V).
    quads: The mesh topology as quads, (Q, 4).
    triangles: The mesh topology as triangles, (T, 3).
    version: The version of the loaded GNM model.
    pose_correctives_regressor: Matrix for pose correctives, (9*J, 3*V).
    joint_regressor: Mapping from vertices to joints, (J, V).
    bone_aligned_orientations: The bone-aligned rotations for each joint, (J, 3,
      3). If they do not exist in the GNM npz, they are set to the identity
      matrix. Note that these are not used to compute the GNM joint and vertex
      positions.
    num_vertices: The number of vertices in the mesh V.
    num_joints: The number of joints in the skeleton J.
    identity_dim: The dimensionality of the linear identity basis I.
    expression_dim: The dimensionality of the linear expression basis E.
  """

  _FIELD_NAMES_DTYPE_MAP = {
      'version': _as_original,
      'variant': _as_original,
      'template_vertex_positions': _as_tf_constant_float32,
      'template_joint_positions': _as_tf_constant_float32,
      'vertex_identity_basis': _as_tf_constant_float32,
      'joint_identity_basis': _as_tf_constant_float32,
      'expression_basis': _as_tf_constant_float32,
      'identity_names': _as_original,
      'joint_names': _as_original,
      'expression_names': _as_original,
      'joint_parent_indices': _as_original,
      'skinning_weights': _as_tf_constant_float32,
      'quads': _as_tf_constant_int32,
      'triangles': _as_tf_constant_int32,
      'quad_uvs': _as_tf_constant_float32,
      'triangle_uvs': _as_tf_constant_float32,
      'mesh_component_names': _as_original,
      'mirror_indices': _as_tf_constant_int32,
      'joint_regressor': _as_tf_constant_float32,
      'pose_correctives_regressor': _as_tf_constant_float32,
      'bone_aligned_template_joint_orientations': _as_tf_constant_float32,
      'vertex_groups': _as_tf_constant_float32,
      'vertex_group_names': _as_original,
  }

  version: gnm_specs.GNMVersion
  variant: gnm_specs.GNMVariant
  template_vertex_positions: tf.Tensor  # (V, 3)
  template_joint_positions: tf.Tensor  # (J, 3)
  vertex_identity_basis: tf.Tensor  # (I, V, 3)
  joint_identity_basis: tf.Tensor  # (I, J, 3)
  expression_basis: tf.Tensor  # (E, V, 3)
  identity_names: Sequence[str]  # (I,)
  joint_names: Sequence[str]  # (J,)
  expression_names: Sequence[str]  # (E,)
  joint_parent_indices: Sequence[int]  # (J,)
  skinning_weights: tf.Tensor  # (J, V)
  quads: tf.Tensor  # (Q, 4)
  triangles: tf.Tensor  # (T, 3)
  quad_uvs: tf.Tensor  # (Q, 4, 2)
  triangle_uvs: tf.Tensor  # (T, 3, 2)
  mesh_component_names: Sequence[str]  # (P,)
  mirror_indices: tf.Tensor  # (V,)
  joint_regressor: tf.Tensor  # (J, V)
  pose_correctives_regressor: tf.Tensor  # (9*J, 3*V)
  bone_aligned_template_joint_orientations: tf.Tensor  # (J, 3, 3)
  vertex_groups: tf.Tensor  # (G, V)
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
      if isinstance(val, tf.Tensor):
        val = val.numpy()
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
      identity: TensorOrVariable | None = None,
      expression: TensorOrVariable | None = None,
      rotations: TensorOrVariable | None = None,
      translation: TensorOrVariable | None = None,
  ) -> tf.Tensor:
    """Evaluates the GNM mesh-generating function.

    Input parameters have optional batch dimensions [A1, ..., An]. Parameters
    may be omitted - in this case default values will be substituted.

    Args:
      identity: Identity coefficients ([A1, ..., An], I). If None, zeros will be
        used.
      expression: Expression coefficients ([A1, ..., An], E,). If None, zeros
        will be used.
      rotations: Joint rotations ([A1, ..., An], J, 3), in axis-angle format. If
        None, zeros will be used.
      translation: Root-joint translation ([A1, ..., An], 3,). If None, zeros
        will be used.

    Returns:
      Mesh vertices ([A1, ..., An], V, 3).

    Raises:
      ValueError if an input argument has the wrong format.
    """

    batch_dims = _get_batch_dims(identity, expression, rotations, translation)

    _check_batch_dims(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )

    # Fill in missing parameter values with zeros.
    if identity is None:
      identity = tf.zeros(tf.concat([batch_dims, [self.identity_dim]], 0))
    if expression is None:
      expression = tf.zeros(tf.concat([batch_dims, [self.expression_dim]], 0))
    if rotations is None:
      rotations = tf.zeros(tf.concat([batch_dims, [self.num_joints, 3]], 0))
    if translation is None:
      translation = tf.zeros(tf.concat([batch_dims, [3]], 0))

    _maybe_check_static_suffix(
        identity, 'identity', (self.identity_dim,), start=-1
    )
    _maybe_check_static_suffix(
        expression, 'expression', (self.expression_dim,), start=-1
    )
    _maybe_check_static_suffix(
        rotations, 'rotations', (self.num_joints, 3), start=-2
    )
    _maybe_check_static_suffix(translation, 'translation', (3,), start=-1)

    # Bind pose vertex positions with identity and expression,
    # ([A1, ..., An], V, 3).
    vertices = self.vertex_positions_bind_pose(identity, expression)

    # Bind pose joint positions in the bind pose, with identity,
    # ([A1, ..., An], J, 3).
    joints = self.joint_positions_bind_pose(identity)

    # Apply pose correctives to the vertices, ([A1, ..., An], V, 3).
    pose_correctives = self.compute_pose_correctives(rotations)
    vertices += pose_correctives

    return self.vertex_positions_world(vertices, joints, rotations, translation)

  def vertices_and_landmarks(
      self,
      landmarks_type: gnm_landmarks.GNMLandmarksType,
      identity: TensorOrVariable | None = None,
      expression: TensorOrVariable | None = None,
      rotations: TensorOrVariable | None = None,
      translation: TensorOrVariable | None = None,
  ) -> tuple[tf.Tensor, tf.Tensor]:
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
          tf.convert_to_tensor(config.indices, dtype=tf.int32),
          tf.convert_to_tensor(config.weights, dtype=tf.float32),
      )
    indices, weights = self._landmarks[landmarks_type]

    vertices = self(
        identity=identity,
        expression=expression,
        rotations=rotations,
        translation=translation,
    )
    weights = tf.cast(weights, dtype=vertices.dtype)

    face_vertices = tf.gather(vertices, indices, axis=-2)
    landmarks = tf.reduce_sum(
        face_vertices * tf.expand_dims(weights, axis=-1), axis=-2
    )
    return vertices, landmarks

  def vertex_positions_world(
      self,
      vertices: tf.Tensor,
      joints: tf.Tensor,
      rotations: tf.Tensor,
      translation: tf.Tensor,
  ) -> tf.Tensor:
    """Applies linear blend skinning to the input GNM vertices.

    Args:
      vertices: The vertices in the bind-pose, ([A1, ..., An], V, 3).
      joints: The joints in the bind-pose, ([A1, ..., An], J, 3).
      rotations: GNM joint rotations in axis-angle format, ([A1, ..., An], J,
        3).
      translation: GNM root translation, ([A1, ..., An],  3).

    Returns:
      The posed vertices after applying linear blend skinning about the input
        joints, ([A1, ..., An], V, 3).
    """
    _check_batch_dims(
        vertices=vertices,
        joints=joints,
        rotations=rotations,
        translation=translation,
    )
    _maybe_check_static_suffix(
        vertices, 'vertices', (self.num_vertices, 3), start=-2
    )
    _maybe_check_static_suffix(joints, 'joints', (self.num_joints, 3), start=-2)
    _maybe_check_static_suffix(
        rotations, 'rotations', (self.num_joints, 3), start=-2
    )
    _maybe_check_static_suffix(translation, 'translation', (3,), start=-1)

    # The local-to-world transforms of each joint, after posing, (N, J, 4, 4).
    joint_transforms_world = self.joint_transforms_world(
        joints, rotations, translation
    )
    batch_dims = tf.shape(rotations)[:-2]
    # n_batch = tf.shape(vertices)[0]
    num_joints = self.num_joints

    # Skinning requires we compute the transform from the bind pose to the final
    # pose for each joint. To do this, we pre-apply the inverse joint-to-world
    # transforms for the bind pose. Fortunately, our convention is that joints
    # have zero rotation in the bind pose, so these transforms are simply
    # translation matrices.
    deltas = tf.einsum(
        '...jik,...jk->...ji', joint_transforms_world[..., :3, :3], joints
    )[..., tf.newaxis]

    offset_3x3 = tf.zeros(tf.concat([batch_dims, [num_joints, 3, 3]], 0))
    bottom_row = tf.zeros(tf.concat([batch_dims, [num_joints, 1, 4]], 0))
    offset = tf.concat([offset_3x3, deltas], axis=-1)
    offset = tf.concat([offset, bottom_row], axis=-2)
    joint_transforms = joint_transforms_world - offset

    # Convert the vertices to homogeneous coordinates.
    vertices_h = tf.concat([vertices, tf.ones_like(vertices[..., :1])], -1)

    # Perform Linear Blend Skinning: accumulate the result of each joint's
    # posing transform on the mesh according to skinning weights.
    vertices_skinned = tf.einsum(
        'jv,...jmn,...vn->...vm',
        self.skinning_weights,
        joint_transforms,
        vertices_h,
    )[..., :3]

    return vertices_skinned

  def vertex_positions_bind_pose(
      self,
      identity: TensorOrVariable,
      expression: TensorOrVariable,
  ) -> tf.Tensor:
    """Computes vertices in the bind pose, with identity and expression applied.

    Args:
      identity: A batch of identity coefficients ([A1, ..., An], I,).
      expression: A batch of expression coefficients ([A1, ..., An], E,).

    Returns:
      A batch of vertex positions in the bind pose, ([A1, ..., An], V, 3).
    """
    _check_batch_dims(identity=identity, expression=expression)
    _maybe_check_static_suffix(
        identity, 'identity', (self.identity_dim,), start=-1
    )
    _maybe_check_static_suffix(
        expression, 'expression', (self.expression_dim,), start=-1
    )

    # For formatting.
    vertex_identity_basis = self.vertex_identity_basis
    expression_basis = self.expression_basis

    # Apply linear identity and expression bases to vertices in bind pose.
    identity_deltas = tf.einsum(
        '...i,ijk->...jk', identity, vertex_identity_basis
    )
    expression_deltas = tf.einsum(
        '...i,ijk->...jk', expression, expression_basis
    )

    return self.template_vertex_positions + identity_deltas + expression_deltas

  def compute_pose_correctives(
      self, rotations: TensorOrVariable
  ) -> TensorOrVariable:
    """Apply pose correctives.

    If self.pose_correctives_regressor is not None, return computed pose
    correctives. Otherwise, return zero tensor.

    Args:
      rotations: a batch of rotation vectors, (..., J, 3).

    Returns:
      A batch of pose correctives as vertex offsets, (..., V, 3).
    """
    _maybe_check_static_suffix(
        rotations, 'rotations', (self.num_joints, 3), start=-2
    )
    batch_size = tf.shape(rotations)[:-2]
    if self.pose_correctives_regressor is None:
      # Returns a batch of zeros with the expected shape.
      zeros = tf.zeros_like(self.template_vertex_positions)
      broadcast_shape = tf.concat([batch_size, [self.num_vertices, 3]], axis=0)
      return tf.broadcast_to(zeros, broadcast_shape)

    pose_feature = _rotation_matrices(rotations) - tf.eye(
        3, dtype=rotations.dtype
    )
    pose_feature = tf.reshape(pose_feature, tf.concat([batch_size, [-1]], 0))

    pose_deltas = tf.einsum(
        '...d,dv->...v', pose_feature, self.pose_correctives_regressor
    )

    output_shape = tf.concat([batch_size, [self.num_vertices, 3]], 0)
    return tf.reshape(pose_deltas, output_shape)

  def joint_positions_bind_pose(self, identity: TensorOrVariable) -> tf.Tensor:
    """Joint positions in the bind pose, with identity basis applied, (N, J, 3).

    Args:
      identity: A batch of identity coefficients (..., I,).

    Returns:
      A batch of joint positions in the bind pose, (..., J, 3).
    """
    _maybe_check_static_suffix(
        identity, 'identity', (self.identity_dim,), start=-1
    )
    deltas = tf.einsum('...i,ijk->...jk', identity, self.joint_identity_basis)
    return self.template_joint_positions + deltas

  def joint_transforms_world(
      self,
      joints: TensorOrVariable,
      rotations: TensorOrVariable,
      translation: TensorOrVariable,
  ) -> tf.Tensor:
    """Gets the world-space transforms of each skeletal joint.

    Args:
      joints: Joint locations in the bind pose, (N, J, 3).
      rotations: Per-joint rotations as axis-angle vectors, (N, J, 3).
      translation: The translation of the root joint (N, 3,).

    Returns:
      Local-to-world transforms of each skeletal joint, (N, J, 4, 4).
    """
    _check_batch_dims(
        rotations=rotations, translation=translation, joints=joints
    )
    _maybe_check_static_suffix(
        rotations, 'rotations', (self.num_joints, 3), start=-2
    )
    _maybe_check_static_suffix(translation, 'translation', (3,), start=-1)
    _maybe_check_static_suffix(joints, 'joints', (self.num_joints, 3), start=-2)

    # Use `tf.shape()` for the batch size. Using `rotations.shape[0]` will
    # fail if the batch dimensions is None, such as when using GNM as part of
    # a Keras model.
    batch_dims = tf.shape(rotations)[:-2]
    n_joints = self.num_joints

    # Build 3x3 local rotation matrices for each joint, (..., J, 3, 3).
    rotation_matrices = _rotation_matrices(rotations)

    # Joint offsets for all joints except the root, (..., J-1, 3).
    joint_offsets = joints[..., 1:, :] - tf.gather(
        joints, self.joint_parent_indices[1:], axis=-2
    )

    # Prepend with the joint offsets for the root joints, (..., J, 3).
    root_joints = (translation + joints[..., 0, :])[..., tf.newaxis, :]
    joint_offsets = tf.concat([root_joints, joint_offsets], axis=-2)

    # Include translation to build 4x4 transforms in parent space,
    # (..., J, 4, 4).
    right_cols_shape = tf.concat([batch_dims, [n_joints, 3, 1]], axis=0)
    right_columns = tf.reshape(joint_offsets, right_cols_shape)

    bottom_row = tf.constant([0, 0, 0, 1], dtype=tf.float32)
    bottom_rows_shape = tf.concat([batch_dims, [n_joints, 1, 4]], axis=0)
    bottom_rows = tf.broadcast_to(bottom_row, bottom_rows_shape)

    # Concatenate rotation matrices, right columns, and bottom rows,
    # (..., J, 4, 4).
    transforms_3x4 = tf.concat([rotation_matrices, right_columns], axis=-1)
    transforms_parent = tf.concat([transforms_3x4, bottom_rows], axis=-2)

    # Traverse the skeleton to compute posing transforms in world-space.
    transforms_world = [transforms_parent[..., 0, :, :]]
    for i, parent_index in enumerate(
        tuple(self.joint_parent_indices[1:]), start=1
    ):
      transform_world = (
          transforms_world[parent_index] @ transforms_parent[..., i, :, :]
      )
      transforms_world.append(transform_world)

    return tf.stack(transforms_world, axis=-3)

  def get_posed_joint_transforms(
      self,
      identity: TensorOrVariable,
      rotations: TensorOrVariable,
      translation: TensorOrVariable,
  ) -> TensorOrVariable:
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
    _check_batch_dims(
        identity=identity, rotations=rotations, translation=translation
    )
    _maybe_check_static_suffix(
        identity, 'identity', (self.identity_dim,), start=-1
    )
    _maybe_check_static_suffix(
        rotations, 'rotations', (self.num_joints, 3), start=-2
    )
    _maybe_check_static_suffix(translation, 'translation', (3,), start=-1)

    # Joint positions in the bind pose, with identity model applied, (J, 3).
    joints = self.joint_positions_bind_pose(identity)

    # The local-to-world transforms of each joint, after posing, (J, 4, 4).
    return self.joint_transforms_world(joints, rotations, translation)

  def prune_vertices(self, keep_vertices: tf.Tensor) -> None:
    """Prune GNM model's vertices in-place for faster evaluation.

    Args:
      keep_vertices: Indices of vertices to be kept. Shaped (V*,) of type int32
        or int64. Values must be in range [0, self.num_vertices).
    """
    # Store the original number of vertices.
    num_vertices = self.num_vertices

    self.template_vertex_positions = tf.gather(
        self.template_vertex_positions, keep_vertices, axis=0
    )
    self.vertex_identity_basis = tf.gather(
        self.vertex_identity_basis, keep_vertices, axis=1
    )
    self.expression_basis = tf.gather(
        self.expression_basis, keep_vertices, axis=1
    )
    self.skinning_weights = tf.gather(
        self.skinning_weights, keep_vertices, axis=1
    )

    # Map the vertex indices to the new range [0, V*). Set removed vertices to
    # -1.
    mapper = (
        tf.scatter_nd(
            indices=tf.expand_dims(keep_vertices, axis=-1),
            updates=tf.range(tf.shape(keep_vertices)[0], dtype=tf.int32) + 1,
            shape=(num_vertices,),
        )
        - 1
    )
    quads = tf.gather(mapper, self.quads)
    triangles = tf.gather(mapper, self.triangles)
    # Remove quads and triangles with removed vertices (vertex index is -1).
    quads = tf.gather(
        quads,
        tf.where(tf.reduce_all(quads >= 0, axis=-1))[:, 0],
        axis=0,
    )
    triangles = tf.gather(
        triangles,
        tf.where(tf.reduce_all(triangles >= 0, axis=-1))[:, 0],
        axis=0,
    )
    self.quads = quads
    self.triangles = triangles

    if self.pose_correctives_regressor is not None:
      pose_correctives = tf.reshape(
          self.pose_correctives_regressor,
          [self.num_joints, -1, num_vertices, 3],
      )
      pose_correctives = tf.gather(pose_correctives, keep_vertices, axis=-2)

      self.pose_correctives_regressor = tf.reshape(
          pose_correctives, [-1, len(keep_vertices) * 3]
      )


def _rotation_matrices(axis_angles: TensorOrVariable) -> tf.Tensor:
  """Builds 3x3 rotation matrices axis-angle vectors.

  The rotation matrix is computed using Rodrigues' rotation formula. See here
  for more information:
  https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula


  Args:
    axis_angles: A list axis-angle vectors. Shaped (N, 3).

  Returns:
    3x3 rotation matrices, (N, 3, 3).
  """
  axis_angles = tf.where(
      tf.linalg.norm(axis_angles, axis=-1, keepdims=True) <= _EPSILON,
      axis_angles + _EPSILON,
      axis_angles,
  )
  angle = tf.sqrt(tf.reduce_sum(tf.square(axis_angles), axis=-1, keepdims=True))
  axis = axis_angles / (angle + _EPSILON)

  sin_axis = tf.sin(angle) * axis
  cos_angle = tf.cos(angle)
  cos1_axis = (1.0 - cos_angle) * axis
  _, axis_y, axis_z = tf.unstack(axis, axis=-1)
  cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
  sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
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
  diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
  matrix = tf.stack(
      (diag_x, m01, m02, m10, diag_y, m12, m20, m21, diag_z), axis=-1
  )
  output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)


def _get_batch_dims(
    identity: tf.Tensor | tf.Variable | None = None,
    expression: tf.Tensor | tf.Variable | None = None,
    rotations: tf.Tensor | tf.Variable | None = None,
    translation: tf.Tensor | tf.Variable | None = None,
    vertices: tf.Tensor | tf.Variable | None = None,
    joints: tf.Tensor | tf.Variable | None = None,
    static: bool = False,
) -> tf.Tensor | tuple[int, ...]:
  """Returns the leading batch dimensions [A1, ..., An] for GNM inputs.

  Args:
    identity: Identity coefficients ([A1, ..., An], I).
    expression: Expression coefficients ([A1, ..., An], E).
    rotations: Joint rotations ([A1, ..., An], J, 3),
    translation: Root-joint translation ([A1, ..., An], 3).
    vertices: Vertices, ([A1, ..., An], V, 3).
    joints: Joints, ([A1, ..., An], J, 3).
    static: Whether to return static batch dimensions, i.e. if True, use
      `tf.shape` instead of `tensor.shape`.

  Returns:
    An int32 tf.Tensor of leading batch dimensions (A1, ..., An). If the input
    does not include batch dimensions, return empty tuple ().
  """
  if identity is not None:
    return _shape(identity, static)[:-1]
  if expression is not None:
    return _shape(expression, static)[:-1]
  if rotations is not None:
    return _shape(rotations, static)[:-2]
  if translation is not None:
    return _shape(translation, static)[:-1]
  if vertices is not None:
    return _shape(vertices, static)[:-2]
  if joints is not None:
    return _shape(joints, static)[:-1]
  return tuple() if static else tf.zeros(shape=(0), dtype=tf.int32)


def _shape(
    tensor: tf.Tensor | tf.Variable, static: bool = False
) -> tuple[int, ...] | tf.Tensor:
  """Returns the shape of a tensor."""
  return tensor.shape if static else tf.shape(tensor)


def _check_batch_dims(
    identity: tf.Tensor | tf.Variable | None = None,
    expression: tf.Tensor | tf.Variable | None = None,
    rotations: tf.Tensor | tf.Variable | None = None,
    translation: tf.Tensor | tf.Variable | None = None,
    vertices: tf.Tensor | tf.Variable | None = None,
    joints: tf.Tensor | tf.Variable | None = None,
) -> None:
  """Checks that the batch dimensions are consistent.

  Args:
    identity: Identity coefficients ([A1, ..., An], I).
    expression: Expression coefficients ([A1, ..., An], E).
    rotations: Joint rotations ([A1, ..., An], J, 3),
    translation: Root-joint translation ([A1, ..., An], 3).
    vertices: Vertices, ([A1, ..., An], V, 3).
    joints: Joints, ([A1, ..., An], J, 3).

  Raises:
    InvalidShapeError: if the batch dimensions of the inputs are inconsistent.
  """
  batch_dims = _get_batch_dims(
      identity,
      expression,
      rotations,
      translation,
      vertices,
      joints,
      static=True,
  )

  _maybe_check_static_shape_prefix(identity, batch_dims, 'identity')
  _maybe_check_static_shape_prefix(expression, batch_dims, 'expression')
  _maybe_check_static_shape_prefix(rotations, batch_dims, 'rotations')
  _maybe_check_static_shape_prefix(translation, batch_dims, 'translation')
  _maybe_check_static_shape_prefix(vertices, batch_dims, 'vertices')
  _maybe_check_static_shape_prefix(joints, batch_dims, 'joints')


def _maybe_check_static_shape_prefix(
    tensor: tf.Tensor | tf.Variable | None,
    expected_prefix: tuple[int, ...],
    name: str,
):
  """Checks that the static shape of a tensor matches the expected prefix."""
  if tensor is None:
    return
  if tensor.shape[: len(expected_prefix)] != expected_prefix:
    raise InvalidShapeError(
        f'Tensor shape prefix for `{name}` does not match. Expected'
        f' {expected_prefix} prefix, got {tensor.shape[:len(expected_prefix)]}.'
    )


def _maybe_check_static_suffix(
    tensor: tf.Tensor | tf.Variable | None,
    name: str,
    expected_suffix: tuple[int, ...],
    start: int = 0,
):
  """Checks that the static shape of a tensor matches the expected suffix."""
  if tensor is None:
    return
  if tensor.shape[start:] != expected_suffix:
    raise InvalidShapeError(
        f'Tensor shape for `{name}` does not match. Expected suffix'
        f' {expected_suffix} starting at index {start}, got'
        f' {tensor.shape[start:]}.'
    )
