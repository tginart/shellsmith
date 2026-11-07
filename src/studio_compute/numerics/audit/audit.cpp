/*
Purpose: Inspect lazy MLX graph dtypes without evaluation or tensor mutation.
Input: Graph outputs from the pinned MLX runtime.
Output: One description per primitive, plus its input and output tensor facts.
*/
#include <functional>
#include <sstream>
#include <unordered_set>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "mlx/array.h"
#include "mlx/utils.h"
#include "mlx/primitives.h"
namespace nb=nanobind;
namespace mx=mlx::core;

static nb::dict describe(const mx::array& value) {
    nb::dict result;
    std::ostringstream dtype;
    dtype<<value.dtype();
    result["dtype"]=dtype.str();
    result["shape"]=std::vector<int>(value.shape().begin(),value.shape().end());
    result["id"]=value.id();
    result["bytes"]=value.nbytes();
    return result;
}

static nb::list inspect(const std::vector<mx::array>& outputs) {
    nb::list result;
    std::unordered_set<std::uintptr_t> primitives;
    std::unordered_set<std::uintptr_t> arrays;
    std::function<void(const mx::array&)> visit;
    visit=[&](const mx::array& value) {
        if(!arrays.insert(value.id()).second)return;
        for(const auto& sibling:value.siblings())arrays.insert(sibling.id());
        for(const auto& input:value.inputs())visit(input);
        if(!value.has_primitive() || !primitives.insert(value.primitive_id()).second)return;
        nb::dict node;
        node["primitive"]=value.primitive().name();
        nb::list inputs,children;
        for(const auto& input:value.inputs())inputs.append(describe(input));
        for(const auto& output:value.outputs())children.append(describe(output));
        node["inputs"]=inputs;
        node["outputs"]=children;
        result.append(node);
    };
    for(const auto& output:outputs)visit(output);
    return result;
}

NB_MODULE(_shellsmith_graph_audit,module) {
    module.def("inspect",&inspect,nb::arg("outputs"));
}
