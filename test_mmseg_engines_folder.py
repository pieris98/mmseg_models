import cv2, os, ctypes
import numpy as np
import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt

# Define the number of images to process
N = 40  # or any other value you want to set dynamically

PATH_TO_MMSEG = os.environ["PATH_TO_MMSEG"]
image_dir = PATH_TO_MMSEG + f"/demo/batch{N}/"  # Directory containing batch of images
output_dir = PATH_TO_MMSEG + f"/demo/batch{N}-outs/"  # or any other directory you want to set dynamically

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)[:N]]  # Get N image paths

def preprocess_images(image_paths):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1024, 512))  # Resize to desired shape
        image = image.astype(np.float32)
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        images.append(image)
    images = np.stack(images, axis=0)
    images = np.ascontiguousarray(images)
    return images

def save_annotated_images(segmentation_masks, original_images, output_dir):
    print("\n\nSHAPE OF MASKS IS:" ,segmentation_masks.shape)
    for idx in range(len(original_images)):
        segmentation_mask = segmentation_masks[idx]
        original_image = original_images[idx]
        
        segmentation_mask = segmentation_mask.squeeze()
        num_classes = np.max(segmentation_mask) + 1
        colors = plt.cm.get_cmap('jet', num_classes)(range(num_classes))[:, :3] * 255
        colored_mask = colors[segmentation_mask.astype(np.int32)]

        original_image = np.transpose(original_image, (1, 2, 0))
        original_image = (original_image * 255).astype(np.uint8)
        colored_mask = colored_mask.astype(np.uint8)

        annotated_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)

        output_path = os.path.join(output_dir, f"output_{idx}.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated image saved at {output_path}")

def get_engine_info(engine):
    for idx in range(engine.num_bindings):
        binding_name = engine.get_binding_name(idx)
        binding_shape = engine.get_binding_shape(idx)
        binding_dtype = engine.get_binding_dtype(idx)
        binding_nptype = trt.nptype(binding_dtype)
        is_input = engine.binding_is_input(idx)
        engine.get_tensor_dtype("input")
        if is_input:
            print(f"Binding {idx} ({binding_name}): Shape = {binding_shape}, Dtype = {binding_dtype}, NumpyDtype = {binding_nptype}")
        else:
            print(f"NON_INPUT Binding {idx} ({binding_name}): Shape = {binding_shape}, Dtype = {binding_dtype}, NumpyDtype = {binding_nptype}")

def run_tensorrt_inference(images):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    def load_plugins():
        plugin_library = "/root/openmmlab/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"
        if not os.path.exists(plugin_library):
            raise RuntimeError(f"Plugin library not found at: {plugin_library}")
        ctypes.CDLL(plugin_library)
        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            raise RuntimeError(f"Failed to initialize TensorRT plugins with: {plugin_library}")

    def load_engine(trt_runtime, plan_path):
        with open(plan_path, 'rb') as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)
    
    load_plugins()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(runtime, '/root/openmmlab/work_dir_cityscapes_batch40/end2end.engine')
    if engine is None:
        raise RuntimeError("Failed to load the engine. Ensure the engine and plugins are correctly configured.")

    get_engine_info(engine)
    context = engine.create_execution_context()

    if engine.num_optimization_profiles > 0:
        profile_index = 0
        context.set_optimization_profile_async(profile_index, cuda.Stream().handle)
        min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_index, 0)
        print(f"Optimization profile shapes: min={min_shape}, opt={opt_shape}, max={max_shape}")

    input_shape = images.shape
    print(f"Setting input shape: {input_shape}")
    context.set_binding_shape(0, input_shape)
    assert context.all_binding_shapes_specified

    output_shapes = {
        "output": (input_shape[0], 1, input_shape[2], input_shape[3])
    }

    print(f"Allocating memory: input={input_shape}, outputs={output_shapes}")
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().itemsize))
    d_output = cuda.mem_alloc(int(np.prod(output_shapes["output"]) * np.int32().itemsize))
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    print("Transferring input data to GPU")
    cuda.memcpy_htod_async(d_input, images, stream)
    print("Running inference")
    
    try:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    except Exception as e:
        print(f"Error during inference: {e}")
    
    output = np.empty(output_shapes["output"], dtype=np.int32)
    print("Transferring output data from GPU")
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    print(f"Output shape: {output.shape}")
    print(f"Output data types: {output.dtype}")

    return output

images = preprocess_images(image_paths)
tensorrt_result = run_tensorrt_inference(images)
print(f"Output size {tensorrt_result.shape}")

save_annotated_images(tensorrt_result, images, output_dir)
