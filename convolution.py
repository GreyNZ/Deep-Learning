from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
  """
  Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
  """
  num_examples, in_height, in_width, input_in_channels = inputs.shape
  #print(f"num_examples:{num_examples}, in_width:{in_width}, in_width:{in_width}, input_in_channels:{input_in_channels}")
  filter_height, filter_width, filter_in_channels, filter_out_channels = filters.shape
  #print(f"filter_height:{filter_height}, filter_width:{filter_width}, filter_in_channels:{filter_in_channels}, filter_out_channels:{filter_out_channels}")
  num_examples_stride, strideY, strideX, channels_stride = strides
  #print(f"num_examples_stride:{num_examples_stride}, strideY:{strideY}, strideX:{strideX}, channels_stride:{channels_stride}")

  assert input_in_channels == filter_out_channels, f"Should be the same {input_in_channels} != {filter_out_channels}"

  # Padding
  pad_x, pad_y = 0, 0
  p_zeros = (0, 0)
  if padding == 'SAME': 
    pad_x = (filter_width - 1) // 2
    pad_y = (filter_height - 1) // 2
  output_width = (in_width - filter_width + pad_x * 2) // strideX + 1
  output_height = (in_height - filter_height + pad_y * 2) // strideY + 1
  padded = np.pad(inputs, (p_zeros, (pad_y, pad_y), (pad_x, pad_x), p_zeros))
  conv2d = np.zeros((num_examples, output_height, output_width, filter_out_channels))

  # def corr2d(X, K): 
  #   """Compute 2D cross-correlation."""
  #   h, w = K.shape
  #   Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
  #   for i in range(Y.shape[0]):
  #       for j in range(Y.shape[1]):
  #           Y[i, j].assign(tf.reduce_sum(
  #               X[i: i + h, j: j + w] * K))
  #   return Y

  for i in range(num_examples):
    for y in range(output_height):
      for x in range(output_width):
        for k in range(filter_out_channels):
          conv2d[i, y, x, k] = np.tensordot(padded[i, y: y + filter_height, x: x + filter_width, :], filters[:, :, :, k], ((0, 1, 2), (0, 1, 2)))
  
  #return conv2d # as np array
  return tf.convert_to_tensor(conv2d, tf.float32) # as tensor


