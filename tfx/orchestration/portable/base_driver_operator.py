# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class to define how to operator an executor."""

import abc
from typing import Any, Dict, List, Text

import six
from tfx import orchestration as metadata
from tfx import types
from tfx.proto.orchestration import driver_output_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import abc_utils

from google.protobuf import message


class BaseDriverOperator(six.with_metaclass(abc.ABCMeta, object)):
  """The base class of all executor operators."""

  SUPPORTED_EXECUTABLE_SPEC_TYPE = abc_utils.abstract_property()

  def __init__(self, driver_spec: message.Message,
               mlmd_connection: metadata.Metadata,
               pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_node: pipeline_pb2.PipelineNode):
    """Constructor.

    Args:
      driver_spec: The specification of how to initialize the driver.
      mlmd_connection: ML metadata connection.
      pipeline_info: The information of the pipeline that this driver is in.
      pipeline_node: The specification of the node that this driver is in.

    Raises:
      RuntimeError: if the driver_spec is not supported.
    """
    if not isinstance(driver_spec,
                      tuple(t for t in self.SUPPORTED_EXECUTABLE_SPEC_TYPE)):
      raise RuntimeError('Driver spec not supported: %s' % driver_spec)
    self._driver_spec = driver_spec
    self._mlmd_connection = mlmd_connection
    self._pipeline_info = pipeline_info
    self._pipeline_node = pipeline_node

  @abc.abstractmethod
  def run_driver(
      self, input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any]) -> driver_output_pb2.DriverOutput:
    """Invokes the driver with inputs provided by the Launcher.

    Args:
      input_dict: The defult input_dict resolved by the launcher.
      output_dict: The default output_dict resolved by the launcher.
      exec_properties: The default exec_properties resolved by the launcher.

    Returns:
      An DriverOutput instance.
    """
    pass
