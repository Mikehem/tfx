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
"""Tests for tfx.utils.proto_utils."""

import tensorflow as tf
from tfx.utils import proto_utils
from tfx.utils.testdata import foo_pb2


class ProtoUtilsTest(tf.test.TestCase):

  def test_gather_file_descriptors(self):
    fd_names = set()
    for fd in proto_utils.gather_file_descriptors(foo_pb2.Foo.DESCRIPTOR):
      fd_names.add(fd.name)
    self.assertEqual(
        fd_names, {
            'tfx/utils/testdata/bar.proto',
            'tfx/utils/testdata/foo.proto'
        })

  def test_proto_to_json(self):
    proto = foo_pb2.TestProto(
        option1='hello', option2=2, option3=0.5, integer_list=[42])
    json_str = proto_utils.proto_to_json(proto)
    # Checks whether original field name is kept and fields are sorted.
    self.assertEqual(
        json_str.replace(' ', '').replace('\n', ''),
        '{"integer_list":[42],"option1":"hello","option2":2,"option3":0.5}')

  def test_proto_to_json_dict(self):
    proto = foo_pb2.TestProto(
        option1='hi', option2=4, option3=0.125, integer_list=[24])
    json_dict = proto_utils.proto_to_json_dict(proto)
    # Checks whether original field name is kept and fields are sorted.
    self.assertEqual(json_dict, {
        'integer_list': [24],
        'option1': 'hi',
        'option2': 4,
        'option3': 0.125
    })

  def test_json_to_proto(self):
    json_str = '{"obsolete_field":2,"option1":"x"}'
    self.assertEqual(
        proto_utils.json_to_proto(json_str, foo_pb2.TestProto()),
        foo_pb2.TestProto(option1='x'))

  def test_json_dict_to_proto(self):
    json_dict = {'option1': 'y', 'option2': 3, 'option3': 0.25}
    self.assertEqual(
        proto_utils.json_dict_to_proto(json_dict, foo_pb2.TestProto()),
        foo_pb2.TestProto(option1='y', option2=3, option3=0.25))


if __name__ == '__main__':
  tf.test.main()
