

import argparse
import platform
import sys
import time
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  


import export
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from segment.val import run as val_seg
from utils import notebook_init
from utils.general import LOGGER, check_yaml, file_size, print_args
from utils.torch_utils import select_device
from val import run as val_det


def run(
    weights=ROOT / "yolov5s.pt",  
    imgsz=640,  
    batch_size=1,  
    data=ROOT / "data/coco128.yaml",  
    device="",  
    half=False,  
    test=False,  
    pt_only=False,  
    hard_fail=False,  
):
    """
    Run YOLOv5 benchmarks on multiple export formats and log results for model performance evaluation.

    Args:
        weights (Path | str): Path to the model weights file (default: ROOT / "yolov5s.pt").
        imgsz (int): Inference size in pixels (default: 640).
        batch_size (int): Batch size for inference (default: 1).
        data (Path | str): Path to the dataset.yaml file (default: ROOT / "data/coco128.yaml").
        device (str): CUDA device, e.g., '0' or '0,1,2,3' or 'cpu' (default: "").
        half (bool): Use FP16 half-precision inference (default: False).
        test (bool): Test export formats only (default: False).
        pt_only (bool): Test PyTorch format only (default: False).
        hard_fail (bool): Throw an error on benchmark failure if True (default: False).

    Returns:
        None. Logs information about the benchmark results, including the format, size, mAP50-95, and inference time.

    Notes:
        Supported export formats and models include PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML,
            TensorFlow SavedModel, TensorFlow GraphDef, TensorFlow Lite, and TensorFlow Edge TPU. Edge TPU and TF.js
            are unsupported.

    Example:
        ```python
        $ python benchmarks.py --weights yolov5s.pt --img 640
        ```

    Usage:
        Install required packages:
          $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  
          $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow   
          $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  

        Run benchmarks:
          $ python benchmarks.py --weights yolov5s.pt --img 640
    """
    y, t = [], time.time()
    device = select_device(device)
    model_type = type(attempt_load(weights, fuse=False))  
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():  
        try:
            assert i not in (9, 10), "inference not supported"  
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  
            if "cpu" in device.type:
                assert cpu, "inference not supported on CPU"
            if "cuda" in device.type:
                assert gpu, "inference not supported on GPU"

            
            if f == "-":
                w = weights  
            else:
                w = export.run(
                    weights=weights, imgsz=[imgsz], include=[f], batch_size=batch_size, device=device, half=half
                )[-1]  
            assert suffix in str(w), "export failed"

            
            if model_type == SegmentationModel:
                result = val_seg(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][7]  
            else:  
                result = val_det(data, w, batch_size, imgsz, plots=False, device=device, task="speed", half=half)
                metric = result[0][3]  
            speed = result[2][1]  
            y.append([name, round(file_size(w), 1), round(metric, 4), round(speed, 2)])  
        except Exception as e:
            if hard_fail:
                assert type(e) is AssertionError, f"Benchmark --hard-fail for {name}: {e}"
            LOGGER.warning(f"WARNING ⚠️ Benchmark failure for {name}: {e}")
            y.append([name, None, None, None])  
        if pt_only and i == 0:
            break  

    
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  
    c = ["Format", "Size (MB)", "mAP50-95", "Inference time (ms)"] if map else ["Format", "Export", "", ""]
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f"\nBenchmarks complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    if hard_fail and isinstance(hard_fail, str):
        metrics = py["mAP50-95"].array  
        floor = eval(hard_fail)  
        assert all(x > floor for x in metrics if pd.notna(x)), f"HARD FAIL: mAP50-95 < floor {floor}"
    return py


def test(
    weights=ROOT / "yolov5s.pt",  
    imgsz=640,  
    batch_size=1,  
    data=ROOT / "data/coco128.yaml",  
    device="",  
    half=False,  
    test=False,  
    pt_only=False,  
    hard_fail=False,  
):
    """
    Run YOLOv5 export tests for all supported formats and log the results, including export statuses.

    Args:
        weights (Path | str): Path to the model weights file (.pt format). Default is 'ROOT / "yolov5s.pt"'.
        imgsz (int): Inference image size (in pixels). Default is 640.
        batch_size (int): Batch size for testing. Default is 1.
        data (Path | str): Path to the dataset configuration file (.yaml format). Default is 'ROOT / "data/coco128.yaml"'.
        device (str): Device for running the tests, can be 'cpu' or a specific CUDA device ('0', '0,1,2,3', etc.). Default is an empty string.
        half (bool): Use FP16 half-precision for inference if True. Default is False.
        test (bool): Test export formats only without running inference. Default is False.
        pt_only (bool): Test only the PyTorch model if True. Default is False.
        hard_fail (bool): Raise error on export or test failure if True. Default is False.

    Returns:
        pd.DataFrame: DataFrame containing the results of the export tests, including format names and export statuses.

    Examples:
        ```python
        $ python benchmarks.py --weights yolov5s.pt --img 640
        ```

    Notes:
        Supported export formats and models include PyTorch, TorchScript, ONNX, OpenVINO, TensorRT, CoreML, TensorFlow
        SavedModel, TensorFlow GraphDef, TensorFlow Lite, and TensorFlow Edge TPU. Edge TPU and TF.js are unsupported.

    Usage:
        Install required packages:
            $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  
            $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow   
            $ pip install -U nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com  
        Run export tests:
            $ python benchmarks.py --weights yolov5s.pt --img 640
    """
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():  
        try:
            w = (
                weights
                if f == "-"
                else export.run(weights=weights, imgsz=[imgsz], include=[f], device=device, half=half)[-1]
            )  
            assert suffix in str(w), "export failed"
            y.append([name, True])
        except Exception:
            y.append([name, False])  

    
    LOGGER.info("\n")
    parse_opt()
    notebook_init()  
    py = pd.DataFrame(y, columns=["Format", "Export"])
    LOGGER.info(f"\nExports complete ({time.time() - t:.2f}s)")
    LOGGER.info(str(py))
    return py


def parse_opt():
    """
    Parses command-line arguments for YOLOv5 model inference configuration.

    Args:
        weights (str): The path to the weights file. Defaults to 'ROOT / "yolov5s.pt"'.
        imgsz (int): Inference size in pixels. Defaults to 640.
        batch_size (int): Batch size. Defaults to 1.
        data (str): Path to the dataset YAML file. Defaults to 'ROOT / "data/coco128.yaml"'.
        device (str): CUDA device, e.g., '0' or '0,1,2,3' or 'cpu'. Defaults to an empty string (auto-select).
        half (bool): Use FP16 half-precision inference. This is a flag and defaults to False.
        test (bool): Test exports only. This is a flag and defaults to False.
        pt_only (bool): Test PyTorch only. This is a flag and defaults to False.
        hard_fail (bool | str): Throw an error on benchmark failure. Can be a boolean or a string representing a minimum
            metric floor, e.g., '0.29'. Defaults to False.

    Returns:
        argparse.Namespace: Parsed command-line arguments encapsulated in an argparse Namespace object.

    Notes:
        The function modifies the 'opt.data' by checking and validating the YAML path using 'check_yaml()'.
        The parsed arguments are printed for reference using 'print_args()'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="weights path")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--test", action="store_true", help="test exports only")
    parser.add_argument("--pt-only", action="store_true", help="test PyTorch only")
    parser.add_argument("--hard-fail", nargs="?", const=True, default=False, help="Exception on error or < min metric")
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 benchmark tests or main training/inference routines based on the provided command-line arguments.

    Args:
        opt (argparse.Namespace): Parsed command-line arguments including options for weights, image size, batch size, data
            configuration, device, and other flags for inference settings.

    Returns:
        None: This function does not return any value. It leverages side-effects such as logging and running benchmarks.

    Example:
        ```python
        if __name__ == "__main__":
            opt = parse_opt()
            main(opt)
        ```

    Notes:
        - For a complete list of supported export formats and their respective requirements, refer to the
          [Ultralytics YOLOv5 Export Formats](https://github.com/ultralytics/yolov5
        - Ensure that you have installed all necessary dependencies by following the installation instructions detailed in
          the [main repository](https://github.com/ultralytics/yolov5

        ```shell
        
        $ python benchmarks.py --weights yolov5s.pt --img 640
        ```
    """
    test(**vars(opt)) if opt.test else run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
