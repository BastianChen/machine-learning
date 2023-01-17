from nets import Net1, Net2, Net3
import torch
import argparse
import onnxruntime as rt
import numpy as np


def get_onnx(args, input_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = None
    assert args.model in (1, 2, 3), 'model type must be 1 or 2 or 3'
    if args.model == 1:
        net = Net1(args.num_classes, args.channels).to(device)
    elif args.model == 2:
        net = Net2(args.num_classes, args.channels).to(device)
    elif args.model == 3:
        net = Net3(args.num_classes, args.type, args.channels).to(device)

    net.load_state_dict(torch.load(args.torch_file_path))
    net.eval()

    input_tensor = input_tensor.to(device)

    output = net(input_tensor)
    print(output)
    _ = torch.onnx.export(net, input_tensor, args.onnx_file_path, verbose=False,
                          #   training=False,
                          #   do_constant_folding=True,
                          input_names=['input'], output_names=['output'])


def check_onnx_model(args):
    sess = rt.InferenceSession(args.onnx_file_path)
    inputs_name = sess.get_inputs()[0].name
    outputs_name = sess.get_outputs()[0].name
    output = sess.run([outputs_name], {inputs_name: np.ones(shape=((1, 3, 210, 180)), dtype=np.float32)})
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_file_path", type=str,
                        default='models/resnet34_without_data_augmentation-lr8e-05-epoch30-channels3.pth',
                        help='torch_file_path')
    parser.add_argument("--onnx_file_path", type=str,
                        default='models/resnet34_without_data_augmentation-lr8e-05-epoch30-channels3.onnx',
                        help='onnx_file_path')
    parser.add_argument('-m', '--model', default=3, type=int, help="model type,1:net9layers,2:net44layers,3:resnet")
    parser.add_argument('-c', '--channels', default=3, type=int, help="channels of images")
    parser.add_argument('-t', '--type', default='34', type=str,
                        help="type of resnet,18:resnet18,34:resnet34,50:resnet50")
    parser.add_argument('-n', '--num_classes', default=2, type=int, help="num of class")
    args = parser.parse_args()

    input_tensor = torch.ones(1, 3, 210, 180)

    get_onnx(args, input_tensor)

    check_onnx_model(args)
