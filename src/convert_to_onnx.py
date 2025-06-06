import torch
import torch.onnx
from pytorch_model import Classifier, BasicBlock
import os

def convert_to_onnx():
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(os.path.join('models', 'pytorch_model_weights.pth')))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join('models', 'model.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted to ONNX format: {onnx_path}")
    
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation successful")
    except Exception as e:
        print(f"ONNX validation error: {e}")

if __name__ == "__main__":
    convert_to_onnx() 