import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto

# Create a minimal ONNX model: a single input, single output identity op
input_tensor = onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
output_tensor = onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])

node_def = onnx.helper.make_node(
    "Identity",
    inputs=["input"],
    outputs=["output"],
)

graph_def = onnx.helper.make_graph(
    [node_def],
    "minimal-identity-graph",
    [input_tensor],
    [output_tensor],
)

model_def = onnx.helper.make_model(graph_def, producer_name="minimal-onnx-example")
onnx.save(model_def, "minimal_valid.onnx")
print("Minimal valid ONNX model written to minimal_valid.onnx")
