/*
Purpose: Read lazy MLX operation metadata without evaluation or mutation.
Input: Output arrays and known graph boundaries.
Output: A topological primitive list with array identities, shapes, and dtypes.
*/
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <mlx/array.h>
#include <mlx/primitives.h>
#include <mlx/utils.h>
#include <algorithm>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace nb = nanobind;
namespace mx = mlx::core;

nb::tuple array_metadata(const mx::array& array) {
    std::ostringstream dtype;
    dtype << array.dtype();
    const auto& dimensions = array.shape();
    std::vector<int> shape(dimensions.begin(), dimensions.end());
    return nb::make_tuple(std::to_string(array.id()), shape, dtype.str());
}

nb::list array_metadata_list(const std::vector<mx::array>& arrays) {
    nb::list result;
    for (const auto& array : arrays) {
        result.append(array_metadata(array));
    }
    return result;
}

/*
Purpose: Visit each output group once and stop at the supplied graph boundaries.
Input: Lazy output arrays and boundaries. Output: Primitive metadata.
Note: The function never calls an evaluation or data-access operation.
*/
nb::list read_graph(
    const std::vector<mx::array>& outputs,
    const std::vector<mx::array>& boundaries
) {
    std::unordered_set<std::uintptr_t> visited;
    for (const auto& boundary : boundaries) {
        visited.insert(boundary.id());
    }
    nb::list result;
    std::function<void(const mx::array&)> visit = [&](const mx::array& array) {
        if (!visited.insert(array.id()).second || !array.has_primitive()) {
            return;
        }
        auto group = array.outputs();
        for (const auto& sibling : group) {
            visited.insert(sibling.id());
        }
        for (const auto& input : array.inputs()) {
            visit(input);
        }
        auto& primitive = array.primitive();
        nb::dict node;
        node["type"] = "primitive";
        node["name"] = primitive.name();
        node["inputs"] = array_metadata_list(array.inputs());
        node["outputs"] = array_metadata_list(group);
        node["device_type"] = static_cast<int>(primitive.device().type);
        nb::list arguments;
        if (std::string(primitive.name()) == "QuantizedMatmul") {
            const auto& quantized = static_cast<const mx::QuantizedMatmul&>(primitive);
            const auto [group_size, bits, mode, transpose] = quantized.state();
            arguments.append(group_size);
            arguments.append(bits);
            arguments.append(static_cast<int>(mode));
            arguments.append(transpose);
        }
        node["arguments"] = arguments;
        result.append(node);
    };
    for (const auto& output : outputs) {
        visit(output);
    }
    return result;
}

NB_MODULE(_mlx_graph_reader, module) {
    module.def("read_graph", &read_graph);
    module.def("metadata", &array_metadata_list);
}
