import cv2, os, ctypes
import numpy as np
import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt

PATH_TO_MMSEG = os.environ["PATH_TO_MMSEG"]
image_path = PATH_TO_MMSEG + "/demo/batch6/01.png"

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Static input shape 1024x 512 (W x H) (resize can be adjusted as needed)
    image = cv2.resize(image, (1024, 512))
    image = image.astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = np.ascontiguousarray(image)
    return image

def save_annotated_image(segmentation_mask, original_image, output_path):
    """
    Save the annotated image with the segmentation mask overlayed.

    Parameters:
    segmentation_mask (numpy.ndarray): The segmentation mask produced by the model with class indices.
    original_image (numpy.ndarray): The original image before preprocessing.
    output_path (str): Path to save the annotated image.
    """
    # Ensure the segmentation mask is of the expected shape
    if len(segmentation_mask.shape) != 4:
        raise ValueError(f"Expected segmentation mask shape to be len==4 (B, C, H, W), but got {segmentation_mask.shape}")
    # Ensure the segmentation mask is of the expected shape
    if segmentation_mask.shape[0] != 1 or segmentation_mask.shape[1] != 1:
        raise ValueError(f"Expected segmentation mask shape to be (1, 1, H, W), but got {segmentation_mask.shape}")
    
    # Squeeze the mask to remove the batch and channel dimensions
    segmentation_mask = segmentation_mask.squeeze()
    print("Shape of squeezed seg_mask=" ,segmentation_mask.shape)
    print("Shape of original image=",original_image.shape)
    # Ensure the segmentation mask and original image have the same width and height
    if segmentation_mask.shape != (original_image.shape[2], original_image.shape[3]):
        raise ValueError(f"Segmentation mask shape {segmentation_mask.shape} does not match original image shape {original_image.shape[1:]}")

    # Map class indices to colors
    num_classes = np.max(segmentation_mask) + 1
    colors = plt.cm.get_cmap('jet', num_classes)(range(num_classes))[:, :3] * 255
    colored_mask = colors[segmentation_mask.astype(np.int32)]

    # Convert the original image back to its original shape (H, W, C) and format
    original_image = np.transpose(original_image.squeeze(), (1, 2, 0))
    original_image = (original_image * 255).astype(np.uint8)

    # Ensure the colored_mask is in uint8 format
    colored_mask = colored_mask.astype(np.uint8)

    # Overlay the colored mask on the original image
    annotated_image = cv2.addWeighted(original_image, 0.7, colored_mask, 0.3, 0)

    # Save the annotated image
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


def run_tensorrt_inference(image):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    def load_plugins():
        plugin_library = "/home/lemon/anomaly/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"
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
    engine = load_engine(runtime, '/home/lemon/anomaly/work_dir_cityscapes/end2end.engine')
    if engine is None:
        raise RuntimeError("Failed to load the engine. Ensure the engine and plugins are correctly configured.")

    get_engine_info(engine)

    context = engine.create_execution_context()

    if engine.num_optimization_profiles > 0:
        profile_index = 0
        context.set_optimization_profile_async(profile_index, cuda.Stream().handle)

        min_shape, opt_shape, max_shape = engine.get_profile_shape(profile_index, 0)
        print(f"Optimization profile shapes: min={min_shape}, opt={opt_shape}, max={max_shape}")

    input_shape = image.shape
    print(f"Setting input shape: {input_shape}")
    context.set_binding_shape(0, input_shape)
    assert context.all_binding_shapes_specified

    output_shapes = {
        "output": (1, 1, 512, 1024),
    }

    print(f"Allocating memory: input={input_shape}, outputs={output_shapes}")
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * np.float32().itemsize))
    d_output = cuda.mem_alloc(int(np.prod(output_shapes["output"]) * np.int32().itemsize))
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    print("Transferring input data to GPU")
    cuda.memcpy_htod_async(d_input, image, stream)
    print("Running inference")
    
    try:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    except Exception as e:
        print(f"Error during inference: {e}")
    
    output = np.empty(output_shapes["output"], dtype=np.int32)
    print("Transferring output data from GPU")
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    print("Output shape: output{}".format(output.shape))
    print("Output data types: output={}".format(output.dtype))

    return output

image = preprocess_image(image_path)

tensorrt_result = run_tensorrt_inference(image.copy())
print(f"Output size {tensorrt_result.shape}")
print(f"Output values {tensorrt_result}")

save_annotated_image(tensorrt_result, image, "tensorrt_output_swin_01.jpg")

# ONNX inference
# def run_onnx_inference(image, original_image):
#     providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#     session = ort.InferenceSession('work_dir_panopt/end2end.onnx', providers=providers)

#     # Get the input name for the ONNX model
#     input_name = session.get_inputs()[0].name

#     # Run the inference
#     result = session.run(None, {input_name: image})
    
#     # Assuming result[0] is the desired output for post-processing
#     save_annotated_image(result[0], original_image, "onnx_annotated_image.jpg")
#     return result

# Run both inferences
# onnx_result = run_onnx_inference(image, original_image.copy())
# print("ONNX Inference Result:", onnx_result)
