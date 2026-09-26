import numpy as np
from torch.nn.functional import relu6

def conv_output_size(input_size : np.ndarray, kernel_size : int, stride : int, padding : int) -> np.ndarray:
    # suitable for maxpool too
    return (input_size - kernel_size + 2*padding)//stride + 1


def flops(image_size : np.ndarray, batch : np.ndarray, emb_dim : int = 32, out_dim : int = 100) -> np.ndarray:
    input_conv_output_size = conv_output_size(image_size, 7, 2, 3)
    input_conv_flops = 2 * input_conv_output_size**2 * 7 * 7 * 3 * emb_dim
    max_pool_output_size = conv_output_size(input_conv_output_size, 3, 2, 1)
    max_pool_flops = emb_dim * max_pool_output_size**2 * (3*3 - 1)

    conv1_output_size = conv_output_size(max_pool_output_size, 5, 1, 2)
    conv1_flops = 2 * conv1_output_size**2 * 5 * 5 * emb_dim * 2*emb_dim
    relu1_flops = conv1_output_size**2 * 2*emb_dim

    conv2_output_size = conv_output_size(conv1_output_size, 3, 2, 1)
    conv2_flops = 2 * conv2_output_size**2 * 3 * 3 * 2*emb_dim * 4*emb_dim
    relu2_flops = conv2_output_size**2 * 4*emb_dim

    conv3_output_size = conv_output_size(conv2_output_size, 1, 1, 0)
    conv3_flops = 2 * conv3_output_size**2 * 1 * 1 * 4*emb_dim * 8*emb_dim
    relu3_flops = conv3_output_size**2 * 8*emb_dim

    conv4_output_size = conv_output_size(conv3_output_size, 3, 2, 1)
    conv4_flops = 2 * conv4_output_size**2 * 3 * 3 * 8*emb_dim * 8*emb_dim
    relu4_flops = conv4_output_size**2 * 8*emb_dim

    conv5_output_size = conv_output_size(conv4_output_size, 1, 1, 0)
    conv5_flops = 2 * conv5_output_size**2 * 1 * 1 * 8*emb_dim * 16*emb_dim
    relu5_flops = conv5_output_size**2 * 16*emb_dim

    avg_pool_flops = 16*emb_dim * conv5_output_size**2
    linear1_flops = 2 * 16*emb_dim * 8*emb_dim
    relu6_flops = 8*emb_dim
    linear2_flops = 2 * 8*emb_dim * out_dim
    result = (
        input_conv_flops + conv1_flops + conv2_flops + conv3_flops + conv4_flops + conv5_flops +
        relu1_flops + relu2_flops + relu3_flops + relu4_flops + relu5_flops + relu6_flops +
        max_pool_flops + avg_pool_flops + linear1_flops + linear2_flops
    ) * batch.T
    return result

def memory(image_size : np.ndarray, batch : np.ndarray, emb_dim : int = 32, out_dim : int = 100) ->  np.ndarray:
    input_values = 3*image_size**2
    input_conv_output_size = conv_output_size(image_size, 7, 2, 3)
    input_conv_weights = 3 * emb_dim * 7 * 7
    input_conv_activation_size = emb_dim * input_conv_output_size**2
    max_pool_output_size = conv_output_size(input_conv_output_size, 3, 2, 1)
    max_pool_activation_size = emb_dim * max_pool_output_size**2

    # As ReLU is inplace, we'll omit them in memory calculations
    conv1_output_size = conv_output_size(max_pool_output_size, 5, 1, 2)
    conv1_weights = emb_dim * 2*emb_dim * 5 * 5
    conv1_activation_size = 2*emb_dim * conv1_output_size**2

    conv2_output_size = conv_output_size(conv1_output_size, 3, 2, 1)
    conv2_weights = 2*emb_dim * 4*emb_dim * 3 * 3
    conv2_activation_size = 4*emb_dim * conv2_output_size**2

    conv3_output_size = conv_output_size(conv2_output_size, 1, 1, 0)
    conv3_weights = 4*emb_dim * 8*emb_dim * 1 * 1
    conv3_activation_size = 8*emb_dim * conv3_output_size**2

    conv4_output_size = conv_output_size(conv3_output_size, 3, 2, 1)
    conv4_weights = 8*emb_dim * 8*emb_dim * 3 * 3
    conv4_activation_size = 8*emb_dim * conv4_output_size**2

    conv5_output_size = conv_output_size(conv4_output_size, 1, 1, 0)
    conv5_weights = 8*emb_dim * 16*emb_dim * 1 * 1
    conv5_activation_size = 16*emb_dim * conv5_output_size**2

    avg_pool_size = 16*emb_dim
    linear1_weights = 16*emb_dim * 8*emb_dim
    linear1_activation_size = 8*emb_dim
    linear2_weights = 8*emb_dim * out_dim
    linear2_activation_size = out_dim

    result = (
        input_values + input_conv_activation_size + max_pool_activation_size + conv1_activation_size +
        conv2_activation_size + conv3_activation_size + conv4_activation_size + conv5_activation_size +
        avg_pool_size + linear1_activation_size + linear2_activation_size
    ) * batch.T + (
        input_conv_weights + conv1_weights + conv2_weights + conv3_weights + conv4_weights + conv5_weights +
        linear1_weights + linear2_weights
    )
    # Assuming that all the data is FP32, then multypling by 4 is bytes
    return 4*result

def _active_bytes(image_size : np.ndarray, batch : np.ndarray, emb_dim : int = 32, out_dim : int = 100):
    input_values = 3*image_size**2
    input_conv_output_size = conv_output_size(image_size, 7, 2, 3)
    input_conv_activation_size = emb_dim * input_conv_output_size**2
    max_pool_output_size = conv_output_size(input_conv_output_size, 3, 2, 1)
    max_pool_activation_size = emb_dim * max_pool_output_size**2

    # As ReLU is inplace, we'll omit them in memory calculations
    conv1_output_size = conv_output_size(max_pool_output_size, 5, 1, 2)
    conv1_activation_size = 2*emb_dim * conv1_output_size**2

    conv2_output_size = conv_output_size(conv1_output_size, 3, 2, 1)
    conv2_activation_size = 4*emb_dim * conv2_output_size**2

    conv3_output_size = conv_output_size(conv2_output_size, 1, 1, 0)
    conv3_activation_size = 8*emb_dim * conv3_output_size**2

    conv4_output_size = conv_output_size(conv3_output_size, 3, 2, 1)
    conv4_activation_size = 8*emb_dim * conv4_output_size**2

    conv5_output_size = conv_output_size(conv4_output_size, 1, 1, 0)
    conv5_activation_size = 16*emb_dim * conv5_output_size**2

    avg_pool_size = 16*emb_dim
    linear1_activation_size = 8*emb_dim
    linear2_activation_size = out_dim

    result = (
        input_values + input_conv_activation_size + max_pool_activation_size + conv1_activation_size +
        conv2_activation_size + conv3_activation_size + conv4_activation_size + conv5_activation_size +
        avg_pool_size + linear1_activation_size + linear2_activation_size
    ) * batch.T
    # Assuming that all the data is FP32, then multypling by 4 is bytes
    return 4*result

def _throughput_bytes(bytes : np.ndarray, theta : np.ndarray) -> np.ndarray:
    throughput_max, throughut_min, bytes_threshold = np.exp(theta) # all the values has to be non-negative
    return np.minimum(throughput_max, throughut_min + bytes*(throughput_max-throughut_min)/bytes_threshold)

def latency(image_size : np.ndarray, batch : np.ndarray, theta_lat : np.ndarray, emb_dim : int = 32, out_dim : int = 100) ->  np.ndarray:
    active_bytes = _active_bytes(image_size, batch, emb_dim, out_dim)
    throughput = _throughput_bytes(active_bytes, theta_lat)
    return active_bytes * throughput

def energy(image_size : np.ndarray, batch : np.ndarray, theta_energy : np.ndarray, emb_dim : int = 32, out_dim : int = 100) ->  np.ndarray:
    active_bytes = _active_bytes(image_size, batch, emb_dim, out_dim)
    throughput = _throughput_bytes(active_bytes, theta_energy)
    return active_bytes * throughput**2
