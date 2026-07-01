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

"""Tests for gnm_base."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from absl.testing import absltest
from gnm.shape import gnm_base
from gnm.shape.data.versions import gnm_specs


class DummyGNM(gnm_base.GNMBase):
  """Dummy GNM subclass for testing."""

  def __init__(
      self,
      version: gnm_specs.GNMVersion,
      variant: gnm_specs.GNMVariant,
  ) -> None:
    self.version = version
    self.variant = variant

  def to_numpy_data_dict(self) -> dict[str, Any]:
    return {"dummy": 1}

  @classmethod
  def _from_model_data(
      cls,
      data_dict: Mapping[str, Any],
  ) -> DummyGNM:
    del data_dict
    return cls(
        version=gnm_specs.GNMVersion.V0_0,
        variant=gnm_specs.GNMVariant.EXPERIMENTAL,
    )


class GNMBaseTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.gnm = DummyGNM(
        version=gnm_specs.GNMVersion.V0_0,
        variant=gnm_specs.GNMVariant.EXPERIMENTAL,
    )

  def test_properties(self) -> None:
    self.assertEqual(self.gnm.major_version, gnm_specs.GNMMajorVersion.V0)
    self.assertEqual(self.gnm.body_part, gnm_specs.GNMBodyPart.EXPERIMENTAL)

  def test_from_gnm(self) -> None:
    new_gnm = DummyGNM.from_gnm(self.gnm)
    self.assertIsInstance(new_gnm, DummyGNM)
    self.assertEqual(new_gnm.version, gnm_specs.GNMVersion.V0_0)
    self.assertEqual(new_gnm.variant, gnm_specs.GNMVariant.EXPERIMENTAL)

  def test_metaclass_enforces_schema(self) -> None:
    with self.assertRaises(TypeError):

      class GNM(gnm_base.GNMBase):  # pylint: disable=unused-variable
        pass


if __name__ == "__main__":
  absltest.main()
