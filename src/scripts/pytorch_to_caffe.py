# Some standard imports
import io
import os
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from caffe2.python.predictor import mobile_exporter

import argparse

from model_setup.train_setup_records import create_train_setup
from tools.project import proj_dir


def get_model(model_id, setup, epoch=None):
    setup = create_train_setup(setup)

    setup.create_components(load_pretrained=False)
    setup.load_checkpoint(epoch, model_id, mode="test")

    return setup.model


def check_model(model_export_path, x, torch_out):
    model = onnx.load(model_export_path)

    prepared_backend = onnx_caffe2_backend.prepare(model)

    W = {model.graph.input[0].name: x.data.cpu().numpy()}

    c2_out = prepared_backend.run(W)[0]

    np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

    print("Exported model has been executed on Caffe2 backend, and the result looks good!")

    return prepared_backend


def export_to_mobile(prepared_backend, model_export_root):
    c2_workspace = prepared_backend.workspace
    c2_model = prepared_backend.predict_net

    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    with open(os.path.join(model_export_root, 'init_net.pb'), "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open(os.path.join(model_export_root, 'predict_net.pb'), "wb") as fopen:
        fopen.write(predict_net.SerializeToString())


def pytorch_to_caffe(model_id, setup, epoch=None):
    model = get_model(model_id, setup, epoch)
    model.train(False)

    x = torch.randn(1, 4, 3, 256, 256, requires_grad=True, device="cuda")

    model_export_root = proj_dir("export", f"m{model_id}_{setup}")
    os.makedirs(model_export_root)

    model_export_path = os.path.join(model_export_root, "model.onnx")

    # Export the model
    torch_out = torch.onnx._export(model,
                                   x,
                                   model_export_path,
                                   export_params=True)

    prepared_backend = check_model(model_export_path, x, torch_out)

    export_to_mobile(prepared_backend, model_export_root)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", default="unnamed")
    parser.add_argument("--setup")

    args = parser.parse_args()

    pytorch_to_caffe(args.model_id, args.setup)


if __name__ == "__main__":
    main()
