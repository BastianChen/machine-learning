import torch
import numpy as np
from net import Net_Linear
import tensorrt as trt

width, height = 28, 28
eps = 1e-05


def build_trt_network(network, weights_path, input_tensor):
    params = np.load(weights_path)

    linear1_w = np.ascontiguousarray(params['linear.0.weight'])
    linear1_b = np.ascontiguousarray(params['linear.0.bias'])
    fc1 = network.add_fully_connected(input=input_tensor, num_outputs=1024, kernel=linear1_w, bias=linear1_b)

    bn1_w = np.ascontiguousarray(params['linear.1.weight'])
    bn1_b = np.ascontiguousarray(params['linear.1.bias'])
    bn1_mean = np.ascontiguousarray(params['linear.1.running_mean'])
    bn1_var = np.ascontiguousarray(params['linear.1.running_var'])
    scale_1 = bn1_w / np.sqrt(bn1_var + eps)
    bias_1 = bn1_b - bn1_mean * scale_1
    power_1 = np.ones_like(scale_1)
    bn1 = network.add_scale(fc1.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=bias_1, scale=scale_1, power=power_1)

    prelu1_w = np.ascontiguousarray(params['linear.2.weight'])
    prelu1 = network.add_activation(bn1.get_output(0), trt.ActivationType.LEAKY_RELU)
    prelu1.alpha = prelu1_w

    linear2_w = np.ascontiguousarray(params['linear.3.weight'])
    linear2_b = np.ascontiguousarray(params['linear.3.bias'])
    fc2 = network.add_fully_connected(input=prelu1.get_output(0), num_outputs=512, kernel=linear2_w, bias=linear2_b)

    bn2_w = np.ascontiguousarray(params['linear.4.weight'])
    bn2_b = np.ascontiguousarray(params['linear.4.bias'])
    bn2_mean = np.ascontiguousarray(params['linear.4.running_mean'])
    bn2_var = np.ascontiguousarray(params['linear.4.running_var'])
    scale_2 = bn2_w / np.sqrt(bn2_var + eps)
    bias_2 = bn2_b - bn2_mean * scale_2
    power_2 = np.ones_like(scale_2)
    bn2 = network.add_scale(fc2.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=bias_2, scale=scale_2, power=power_2)

    prelu2_w = np.ascontiguousarray(params['linear.5.weight'])
    prelu2 = network.add_activation(bn2.get_output(0), trt.ActivationType.LEAKY_RELU)
    prelu2.alpha = prelu2_w

    linear3_w = np.ascontiguousarray(params['linear.6.weight'])
    linear3_b = np.ascontiguousarray(params['linear.6.bias'])
    fc3 = network.add_fully_connected(input=prelu2.get_output(0), num_outputs=256, kernel=linear3_w, bias=linear3_b)

    bn3_w = np.ascontiguousarray(params['linear.7.weight'])
    bn3_b = np.ascontiguousarray(params['linear.7.bias'])
    bn3_mean = np.ascontiguousarray(params['linear.7.running_mean'])
    bn3_var = np.ascontiguousarray(params['linear.7.running_var'])
    scale_3 = bn3_w / np.sqrt(bn3_var + eps)
    bias_3 = bn3_b - bn3_mean * scale_3
    power_3 = np.ones_like(scale_3)
    bn3 = network.add_scale(fc3.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=bias_3, scale=scale_3, power=power_3)

    prelu3_w = np.ascontiguousarray(params['linear.8.weight'])
    prelu3 = network.add_activation(bn3.get_output(0), trt.ActivationType.LEAKY_RELU)
    prelu3.alpha = prelu3_w

    linear4_w = np.ascontiguousarray(params['linear.9.weight'])
    linear4_b = np.ascontiguousarray(params['linear.9.bias'])
    fc4 = network.add_fully_connected(input=prelu3.get_output(0), num_outputs=128, kernel=linear4_w, bias=linear4_b)

    bn4_w = np.ascontiguousarray(params['linear.10.weight'])
    bn4_b = np.ascontiguousarray(params['linear.10.bias'])
    bn4_mean = np.ascontiguousarray(params['linear.10.running_mean'])
    bn4_var = np.ascontiguousarray(params['linear.10.running_var'])
    scale_4 = bn4_w / np.sqrt(bn4_var + eps)
    bias_4 = bn4_b - bn4_mean * scale_4
    power_4 = np.ones_like(scale_4)
    bn4 = network.add_scale(fc4.get_output(0), mode=trt.ScaleMode.CHANNEL, shift=bias_4, scale=scale_4, power=power_4)

    prelu4_w = np.ascontiguousarray(params['linear.11.weight'])
    prelu4 = network.add_activation(bn4.get_output(0), trt.ActivationType.LEAKY_RELU)
    prelu4.alpha = prelu4_w

    linear5_w = np.ascontiguousarray(params['linear.12.weight'])
    linear5_b = np.ascontiguousarray(params['linear.12.bias'])
    fc5 = network.add_fully_connected(input=prelu4.get_output(0), num_outputs=10, kernel=linear5_w, bias=linear5_b)

    fc5.get_output(0).name = 'predict'
    network.mark_output(tensor=fc5.get_output(0))

    return network


if __name__ == '__main__':
    net_path = r"model/model_without_conv.pth"
    weights_path = 'para.npz'
    net = Net_Linear()
    net.load_state_dict(torch.load(net_path))
    weights = net.state_dict()
    torchPara = {}
    for key, value in weights.items():
        torchPara[key] = value.detach().numpy()

    np.savez(weights_path, **torchPara)

    logger = trt.Logger(trt.Logger.ERROR)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()

    input_tensor = network.add_input(name='data', dtype=trt.float32, shape=[-1, 1, height, width])
    profile.set_shape(input_tensor.name, [1, 1, height, width], [4, 1, height, width], [8, 1, height, width])
    config.add_optimization_profile(profile)

    network = build_trt_network(network, weights_path, input_tensor)

    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open('./model_trt.plan', "wb") as f:
        f.write(engineString)
