import torch
import time
import argparse
import tensorrt as trt
from gen_datasets import Datasets
from torch.utils.data import DataLoader


def trt_version():
    return trt.__version__


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # 创建输出tensor，并分配内存
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)  # 通过binding_name找到对应的input_id
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))  # 找到对应的数据类型
            shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))  # 找到对应的形状大小
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()  # 绑定输出数据指针

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[0].contiguous().data_ptr()

        # 执行推理
        self.context.execute_async(
            batch_size, bindings, torch.cuda.current_stream().cuda_stream
        )

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs[0]


class Detector:
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger = trt.Logger(trt.Logger.INFO)
        self.engine = None
        with open(args.trt_file_path, "rb") as f, trt.Runtime(logger) as runtime:
            # 输入trt本地文件，返回ICudaEngine对象
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.test_data = Datasets(f"{args.datasets_path}/test", False, args.channels)
        self.test_dataset = DataLoader(self.test_data, batch_size=1, shuffle=True)

    def show_info(self):
        # 查看输入输出的名字，类型，大小
        for idx in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(idx)
            name = self.engine.get_binding_name(idx)
            op_type = self.engine.get_binding_dtype(idx)
            shape = self.engine.get_binding_shape(idx)
            print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ',
                  op_type)

    def detect(self):
        trt_model = TRTModule(self.engine, ["input"], ["output"])
        correct = 0
        time_list = []
        for index, (data, label, image) in enumerate(self.test_dataset):
            data, label = data.to(self.device), label.to(self.device)
            start_time = time.time()
            outputs = trt_model(data)
            end_time = time.time()
            inference_time = end_time - start_time
            if index > 0:
                time_list.append(inference_time)
            pred = outputs.argmax(dim=1, keepdim=True)
            label = label.argmax(dim=1, keepdim=True)
            num = pred.eq(label).sum().item()
            correct += num
        acc = 100. * correct / len(self.test_dataset.dataset)

        print(
            f'accuracy: {correct}/{len(self.test_dataset.dataset)} ({acc:.3f}%) ,mean_latency:{sum(time_list) / len(time_list)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--channels', default=3, type=int, help="channels of images")
    parser.add_argument('-t', '--trt_file_path',
                        default="models/resnet34_without_data_augmentation-lr8e-05-epoch30-channels3-int8.trt",
                        type=str, help="name of trt file path")
    parser.add_argument('-dp', '--datasets_path', default="datasets",
                        type=str, help="path of Datasets")

    args = parser.parse_args()
    detector = Detector(args)
    # detector.show_info()
    detector.detect()
