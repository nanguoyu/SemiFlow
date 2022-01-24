"""
@File : convert.py
@Author: Dong Wang
@Date : 1/23/2022
"""
from ..engine import backend
from ..Model import Model
from ..activations import Activation
from ..layer import InputLayer, Dense, Conv2D, RNN, MaxPooling2D, Flatten, Layer
from ..losses import Loss
from ..layer.convolutional import padding
from ..activations import Softmax, Sigmoid, Softplus, Tanh, Relu, Linear, Activation
from .. import activations

try:
    import onnx
    from onnx import shape_inference
except ImportError:
    raise ValueError("To convert SemiFlow to ONNX, you need to install ONNX")


def to_onnx(model: Model, name: str = 'model', save_file='./convented.onnx') -> None:
    # IO tensors (ValueInfoProto).
    model_input_name = "X"

    model_output_name = "Y"

    # nodes
    nodes = []

    # initializer_tensor
    initializer_tensors = []

    layer = model.first_layer
    input_shape = layer.input_shape
    if isinstance(layer, Conv2D):
        h, w, c = input_shape
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, c, h, w])
    elif isinstance(layer, MaxPooling2D):
        h, w, c = input_shape
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, c, h, w])
    elif isinstance(layer, Dense):
        num_features = input_shape[-1]
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, num_features])
    elif isinstance(layer, RNN):
        num_steps, num_features = input_shape
        X = onnx.helper.make_tensor_value_info(model_input_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, num_steps, num_features])
    else:
        raise NotImplementedError(f'Not implemented when the first layer is {type(layer)}')

    last_layer = model.last_layer
    if isinstance(last_layer, Dense):
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, last_layer.units])
    elif isinstance(last_layer, Conv2D):
        out_h, out_w, out_c = last_layer.output_shape
        Y = onnx.helper.make_tensor_value_info(model_output_name,
                                               onnx.TensorProto.FLOAT,
                                               [None, out_c, out_h, out_w])
    else:
        raise NotImplementedError(f'Not implemented when you set {type(last_layer)} as the last layer.')

    input_node_name = model_input_name

    # if not layer.outbound:
    #     return None  # There is only an input layer in the model
    # layer = layer.outbound[0]

    while layer and not isinstance(layer, InputLayer) and not isinstance(layer, Activation) \
            and not isinstance(layer, Loss):
        if isinstance(layer.outbound[0], Loss) and not hasattr(layer, 'activation'):
            output_node_name = model_output_name
        else:
            output_node_name = None
        if isinstance(layer, Dense):
            _nodes, _initializer_tensors, output_node_name = generate_onnx_dense_node(layer, input_node_name,
                                                                                      output_node_name)
        elif isinstance(layer, Conv2D):
            _nodes, _initializer_tensors, output_node_name = generate_onnx_conv2d_node(layer, input_node_name,
                                                                                       output_node_name)
        elif isinstance(layer, MaxPooling2D):
            _nodes, _initializer_tensors, output_node_name = generate_onnx_maxpooling2d_node(layer, input_node_name,
                                                                                             output_node_name)
        elif isinstance(layer, Flatten):
            _nodes, _initializer_tensors, output_node_name = generate_onnx_flatten_node(layer, input_node_name,
                                                                                        output_node_name)
        elif isinstance(layer, RNN):
            # Todo: implement `generate_onnx_rnn_node`
            # Todo: validate the parameter settings od RNN
            raise NotImplementedError("Not implemented", type(layer))
        else:
            raise NotImplementedError("Not implemented", type(layer))

        nodes += _nodes
        initializer_tensors += _initializer_tensors
        input_node_name = output_node_name
        if isinstance(layer.outbound[0], Loss) and hasattr(layer, 'activation'):
            output_node_name = model_output_name
        else:
            output_node_name = None
        if hasattr(layer, 'activation'):
            activation = layer.activation
            if hasattr(layer, 'original_activation_name'):
                activation = activations.get(layer.original_activation_name)
            _nodes, _initializer_tensors, output_node_name = generate_onnx_activation_node(activation,
                                                                                           input_node_name,
                                                                                           output_node_name)
            nodes += _nodes
            initializer_tensors += _initializer_tensors
            input_node_name = output_node_name

        if isinstance(layer.outbound[0], Loss):
            break
        layer = layer.outbound[0]

    # Create the graph (GraphProto)
    graph_def = onnx.helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=[X],  # Graph input
        outputs=[Y],  # Graph output
        initializer=initializer_tensors,
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="semiflow-onnx")
    model_def.opset_import[0].version = 13  # Todo: support multi-versions
    print(model_def)
    model_def = shape_inference.infer_shapes(model_def)
    onnx.checker.check_model(model_def)
    onnx.save(model_def, save_file)


def generate_onnx_conv2d_node(layer: Conv2D, input_node_name: str, output_node_name: str = None) -> (
[onnx.NodeProto], [onnx.TensorProto], str):
    parameters = layer.params
    output_node_name = output_node_name if output_node_name is not None else layer.name + "_onnx_node_output"

    conv_w_initializer_name = layer.name + "_onnx_initializer_tensor"
    conv_w_initializer_tensor = create_onnx_initializer_tensor(name=conv_w_initializer_name,
                                                               tensor_array=parameters['kernel'])
    conv_b_initializer_name = layer.name + "_onnx_initializer_tensor"
    conv_b_initializer_tensor = create_onnx_initializer_tensor(name=conv_b_initializer_name,
                                                               tensor_array=parameters['bias'])
    padding_width = padding(kernel_shape=layer.kernel_size,
                            padding_mode=layer.padding)
    h_pad, w_pad = padding_width[1:3]
    conv_pads = [h_pad[0], w_pad[0], h_pad[1], w_pad[1]]
    conv_node = onnx.helper.make_node(
        name=layer.name + "_onnx_node",  # Name is optional.
        op_type="Conv",
        # Must follow the order of input and output definitions.
        # https://github.com/onnx/onnx/blob/rel-1.9.0/docs/Operators.md#inputs-2---3
        inputs=[
            input_node_name, conv_w_initializer_name,
            conv_b_initializer_name
        ],
        outputs=[output_node_name],
        # The following arguments are attributes.
        kernel_shape=layer.kernel_size,
        # Default values for other attributes: dilations=[1, 1], groups=1
        pads=conv_pads,
        strides=layer.strides,
    )
    return [conv_node], [conv_w_initializer_tensor, conv_b_initializer_tensor], output_node_name


def generate_onnx_maxpooling2d_node(layer: MaxPooling2D, input_node_name: str, output_node_name: str = None) -> (
        [onnx.NodeProto], [onnx.TensorProto], str):
    output_node_name = output_node_name if output_node_name is not None else layer.name + "_onnx_node_output"

    maxpooling_node = onnx.helper.make_node(
        name=layer.name + "_onnx_node",  # Name is optional.
        op_type="MaxPooling",
        inputs=[input_node_name],
        outputs=[output_node_name],
    )
    return [maxpooling_node], [], output_node_name


def generate_onnx_dense_node(layer: Dense, input_node_name: str, output_node_name: str = None) -> (
        [onnx.NodeProto], [onnx.TensorProto], str):
    parameters = layer.params
    output_node_name = output_node_name if output_node_name is not None else layer.name + "_onnx_node_output"

    dense_w_initializer_name = layer.name + "_w_onnx_initializer_tensor"
    w_output_node_name = layer.name + "_w_onnx_node_output"
    dense_w_initializer_tensor = create_onnx_initializer_tensor(name=dense_w_initializer_name,
                                                                tensor_array=parameters['kernel'])

    dense_b_initializer_name = layer.name + "_b_onnx_initializer_tensor"
    b_output_node_name = layer.name + "_b_onnx_node_output"
    dense_b_initializer_tensor = create_onnx_initializer_tensor(name=dense_b_initializer_name,
                                                                tensor_array=parameters['bias'])

    weights_node = onnx.helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=[w_output_node_name],
        value=dense_w_initializer_tensor,
    )

    matmul_node = onnx.helper.make_node(
        op_type="MatMul",
        inputs=[input_node_name, w_output_node_name],
        outputs=[layer.name + "matmul_output"]
    )

    bias_node = onnx.helper.make_node(
        op_type="Constant",
        inputs=[],
        outputs=[b_output_node_name],
        value=dense_b_initializer_tensor,
    )

    add_node = onnx.helper.make_node(
        op_type="Add",
        inputs=[layer.name + "matmul_output", b_output_node_name],
        outputs=[output_node_name]
    )

    return [weights_node, matmul_node, bias_node, add_node], \
           [dense_w_initializer_tensor, dense_b_initializer_tensor], output_node_name


def generate_onnx_flatten_node(layer: Flatten, input_node_name: str, output_node_name: str = None) -> (
        [onnx.NodeProto], [onnx.TensorProto], str):
    output_node_name = output_node_name if output_node_name is not None else layer.name + "_onnx_node_output"

    flatten_node = onnx.helper.make_node(
        name=layer.name + "_onnx_node",  # Name is optional.
        op_type="Flatten",
        inputs=[input_node_name],
        outputs=[output_node_name],
    )
    return [flatten_node], [], output_node_name


def generate_onnx_activation_node(
        activation: Activation,
        input_node_name: str,
        output_node_name: str = None
) -> ([onnx.NodeProto], [onnx.TensorProto], str):
    output_node_name = output_node_name if output_node_name is not None else activation.name + "_onnx_node_output"

    if isinstance(activation, Linear):
        linear_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Constant",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [linear_node], [], output_node_name
    elif isinstance(activation, Sigmoid):
        sigmoid_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Sigmoid",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [sigmoid_node], [], output_node_name
    elif isinstance(activation, Softmax):
        softmax_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Softmax",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [softmax_node], [], output_node_name

    elif isinstance(activation, Softplus):
        softplus_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Softplus",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [softplus_node], [], output_node_name
    elif isinstance(activation, Relu):
        relu_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Relu",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [relu_node], [], output_node_name
    elif isinstance(activation, Tanh):
        tanh_node = onnx.helper.make_node(
            name=activation.name + "_onnx_node",
            op_type="Tanh",
            inputs=[input_node_name],
            outputs=[output_node_name],
        )
        return [tanh_node], [], output_node_name
    else:
        raise NotImplementedError(f'no conversion for {type(activation)} ')


def create_onnx_initializer_tensor(
        name: str,
        tensor_array: backend.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor
