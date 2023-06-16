# create onnx-model file to $ProjectFold/Onnxs/$model_name/$model_name.onnx

import torchvision
import torch
from Onnxs.scripts.export.support import SupportedModels
from Onnxs.scripts.export.config import Config

mymodels={
    "googlenet": torchvision.models.googlenet(pretrained=True),
    "resnet50": torchvision.models.resnet50(pretrained=True),
    "vgg19": torchvision.models.vgg19(pretrained=True),
    "squeezenetv1": torchvision.models.squeezenet1_0(pretrained=True),
}

for i in range(1,15,2):
    # for model_name in mymodels:
    model_name="vgg19"
    if model_name in SupportedModels:
        model = mymodels[model_name]
        model.eval()

        # torch.onnx._export(model, torch.rand(*SupportedModels[model_name]["input_shape"]), Config.ModelSavePathName(model_name), export_params=True,input_names=[SupportedModels[model_name]["input_name"]], output_names=["output"])

        a=list(SupportedModels[model_name]["input_shape"])
        a[0]=i
        torch.onnx._export(model, torch.rand(*a), model_name+"-"+str(a[0])+".onnx", export_params=True,input_names=[SupportedModels[model_name]["input_name"]], output_names=["output"])