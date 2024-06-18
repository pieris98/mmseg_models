backend_config = dict(
    common_config=dict(fp16_mode=False, max_workspace_size=23068672000),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    max_shape=[
                        32,
                        3,
                        512,
                        1024,
                    ],
                    min_shape=[
                        32,
                        3,
                        512,
                        1024,
                    ],
                    opt_shape=[
                        32,
                        3,
                        512,
                        1024,
                    ]))),
    ],
    type='tensorrt')
codebase_config = dict(task='Segmentation', type='mmseg', with_argmax=True)
onnx_config = dict(
    export_params=True,
    input_names=[
        'input',
    ],
    input_shape=[
        1024,
        512,
    ],
    keep_initializers_as_inputs=False,
    opset_version=12,
    optimize=True,
    output_names=[
        'output',
    ],
    save_file='end2end.onnx',
    type='onnx')