"""Execute a deliberately tiny, side-effect-free Python policy subset, without keys."""

from __future__ import annotations

import ast
import json
import resource
import sys

ALLOWED = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.AnnAssign,
    ast.If,
    ast.IfExp,
    ast.Subscript,
    ast.Tuple,
    ast.BinOp,
    ast.BitOr,
    ast.Return,
    ast.Dict,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Expr,
)


def evaluate(payload: dict) -> dict:
    code = payload["code"].strip()
    if code.startswith("```") and code.endswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
    if len(code) > 20_000:
        raise ValueError("Code exceeds the artifact size limit")
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one function definition")
    function = tree.body[0]
    if function.name != "evaluate_request" or function.decorator_list:
        raise ValueError("Expected undecorated evaluate_request")
    if [arg.arg for arg in function.args.args] != ["payload_bytes", "age_seconds"]:
        raise ValueError("Unexpected function parameters")
    if (
        function.args.defaults
        or function.args.kw_defaults
        or function.args.vararg
        or function.args.kwarg
    ):
        raise ValueError("Default and variadic parameters are not allowed")
    nodes = list(ast.walk(tree))
    if len(nodes) > 500:
        raise ValueError("Code exceeds the AST size limit")
    for node in nodes:
        if not isinstance(node, ALLOWED):
            raise ValueError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("Nested functions are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Private interpreter names are not allowed")
    scope = {"__builtins__": {}, "int": int, "bool": bool, "str": str, "dict": dict}
    exec(compile(tree, "<generated-policy>", "exec"), scope)  # noqa: S102
    results = []
    for case in payload["tests"]:
        try:
            actual = scope["evaluate_request"](*case["input"])
            expected = case["expected"]
            passed = (
                isinstance(actual, dict)
                and actual == expected
                and all(type(actual[k]) is type(expected[k]) for k in expected)
            )
            results.append({**case, "actual": actual, "passed": passed})
        except Exception as error:
            results.append({**case, "actual": None, "passed": False, "error": type(error).__name__})
    return {"valid_code": True, "tests": results}


def main() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
    except (OSError, ValueError):
        pass
    try:
        result = evaluate(json.load(sys.stdin))
    except Exception as error:
        result = {"valid_code": False, "error": f"{type(error).__name__}: {error}", "tests": []}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
