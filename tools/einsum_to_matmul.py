import argparse
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto

def replace_einsum_with_matmul(model_path, output_path):
    print(f"Loading model from {model_path}")
    model = onnx.load(model_path)
    
    new_nodes = []
    graph = model.graph
    
    einsum_found = False
    einsum_input1 = None
    einsum_input2 = None
    einsum_output = None
    einsum_idx = -1
    
    for i, node in enumerate(graph.node):
        if node.op_type == "Einsum":
            equation = None
            for attr in node.attribute:
                if attr.name == "equation":
                    equation = attr.s.decode('utf-8')
                    break
            
            if equation == "BD,BCD->BC":
                print(f"Found Einsum operator with equation {equation} in node {i}")
                einsum_found = True
                einsum_input1 = node.input[0]  # BD
                einsum_input2 = node.input[1]  # BCD
                einsum_output = node.output[0]  # BC
                einsum_idx = i
                break
    
    if not einsum_found:
        print("Einsum operator not found or has a different equation")
        return
    
    for i, node in enumerate(graph.node):
        if i != einsum_idx:
            new_nodes.append(node)
    
    unsqueeze_output = einsum_input1 + "_unsqueezed"
    transpose_output = einsum_input2 + "_transposed"
    matmul_output = einsum_output + "_matmul"
    squeeze_output = einsum_output
    
    
    # Unsqueeze 
    axes_tensor_name = f"{einsum_input1}_unsqueeze_axes"
    axes_data = np.array([1], dtype=np.int64)  # Unsqueeze on axis 1
    axes_tensor = helper.make_tensor(
        name=axes_tensor_name,
        data_type=TensorProto.INT64,
        dims=[1],
        vals=axes_data.flatten()
    )
    graph.initializer.append(axes_tensor)
    
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=[einsum_input1, axes_tensor_name],
        outputs=[unsqueeze_output],
        name=f"{einsum_input1}_unsqueeze"
    )
    
    # Transpose
    transpose_node = helper.make_node(
        "Transpose",
        inputs=[einsum_input2],
        outputs=[transpose_output],
        perm=[0, 2, 1],  # Swap C and D positions
        name=f"{einsum_input2}_transpose"
    )
    
    # BatchMatMul
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[unsqueeze_output, transpose_output],
        outputs=[matmul_output],
        name=f"{einsum_output}_matmul"
    )
    
    # Squeeze 
    squeeze_axes_tensor_name = f"{matmul_output}_squeeze_axes"
    squeeze_axes_data = np.array([1], dtype=np.int64)  # Squeeze on axis 1
    squeeze_axes_tensor = helper.make_tensor(
        name=squeeze_axes_tensor_name,
        data_type=TensorProto.INT64,
        dims=[1],
        vals=squeeze_axes_data.flatten()
    )
    graph.initializer.append(squeeze_axes_tensor)
    
    squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[matmul_output, squeeze_axes_tensor_name],
        outputs=[squeeze_output],
        name=f"{matmul_output}_squeeze"
    )
    
    new_nodes.append(unsqueeze_node)
    new_nodes.append(transpose_node)
    new_nodes.append(matmul_node)
    new_nodes.append(squeeze_node)
    
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    try:
        onnx.checker.check_model(model)
        print("Model is valid after changes")
    except Exception as e:
        print(f"Model validation error: {e}")
        return
    
    # Save the modified model
    print(f"Saving modified model to {output_path}")
    onnx.save(model, output_path)
    print("Done!")


if __name__ == "__main__":
    # input_model = "../examples/onnx/model.onnx"
    # output_model = "../examples/onnx/model_fixed_einsum.onnx"
    
    parser = argparse.ArgumentParser(description="Replace Einsum with MatMul in ONNX model")
    parser.add_argument("input_model", type=str, help="Path to the input ONNX model")
    parser.add_argument("output_model", type=str, help="Path to save the modified ONNX model")
    
    args = parser.parse_args()
    
    replace_einsum_with_matmul(args.input_model, args.output_model)