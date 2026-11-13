import ast
import copy

CMP_SWAPS = {
    ast.Lt: [ast.LtE, ast.Gt],
    ast.LtE: [ast.Lt],
    ast.Gt: [ast.GtE, ast.Lt],
    ast.GtE: [ast.Gt],
    ast.Eq: [ast.NotEq],
    ast.NotEq: [ast.Eq],
    ast.In: [ast.NotIn],
    ast.NotIn: [ast.In],
    ast.Is: [ast.IsNot],
    ast.IsNot: [ast.Is],
}
BIN_SWAPS = {
    ast.Add: [ast.Sub],
    ast.Sub: [ast.Add],
    ast.Mult: [ast.FloorDiv],
    ast.FloorDiv: [ast.Mult, ast.Mod],
    ast.Mod: [ast.FloorDiv],
    ast.Div: [ast.FloorDiv],
    ast.BitXor: [ast.BitAnd],
    ast.BitAnd: [ast.BitXor, ast.BitOr],
    ast.BitOr: [ast.BitAnd],
}
_FLIP = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}


def _is_int(node):
    return isinstance(node, ast.Constant) and type(node.value) is int


def _shift(node, delta):
    if _is_int(node):
        return ast.Constant(node.value + delta)
    op = ast.Add() if delta > 0 else ast.Sub()
    return ast.BinOp(left=copy.deepcopy(node), op=op, right=ast.Constant(abs(delta)))


def expr_variants(node):
    """single-edit rewrites of one expression node. used to inject bugs and to propose fixes."""
    out = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        for new_op in CMP_SWAPS.get(type(node.ops[0]), []):
            v = copy.deepcopy(node)
            v.ops = [new_op()]
            out.append(v)
    elif isinstance(node, ast.BinOp):
        for new_op in BIN_SWAPS.get(type(node.op), []):
            v = copy.deepcopy(node)
            v.op = new_op()
            out.append(v)
        # x + 1 -> x, the classic off-by-one
        if isinstance(node.op, (ast.Add, ast.Sub)) and _is_int(node.right) and node.right.value == 1:
            out.append(copy.deepcopy(node.left))
    elif isinstance(node, ast.BoolOp):
        v = copy.deepcopy(node)
        v.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        out.append(v)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        out.append(copy.deepcopy(node.operand))
    elif _is_int(node):
        out.append(ast.Constant(node.value + 1))
        out.append(ast.Constant(node.value - 1))
    elif isinstance(node, ast.Subscript) and not isinstance(node.slice, (ast.Slice, ast.Tuple)):
        for delta in (1, -1):
            v = copy.deepcopy(node)
            v.slice = _shift(node.slice, delta)
            out.append(v)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
        for delta in (1, -1):
            v = copy.deepcopy(node)
            v.args[-1] = _shift(node.args[-1], delta)
            out.append(v)
    return out


def node_variants(node):
    if isinstance(node, ast.AugAssign):
        out = []
        for new_op in BIN_SWAPS.get(type(node.op), []):
            v = copy.deepcopy(node)
            v.op = new_op()
            out.append(v)
        return out
    return expr_variants(node)


def stmt_variants(node, parent_body):
    """statement level bugs: drop a statement, or drop the guard around a block."""
    out = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        return out
    if len(parent_body) > 1 and isinstance(node, (ast.Assign, ast.AugAssign, ast.Expr, ast.If)):
        out.append(None)
    if isinstance(node, ast.If) and not node.orelse:
        out.append(copy.deepcopy(node.body))
    return out


class _Replace(ast.NodeTransformer):
    def __init__(self, target, replacement):
        self.target = target
        self.replacement = replacement

    def visit(self, node):
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)


def _bodies(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def bug_sites(tree):
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.expr):
            sites.extend((node, v) for v in expr_variants(node))
    for block in _bodies(tree):
        for stmt in block:
            sites.extend((stmt, v) for v in stmt_variants(stmt, block))
    return sites


def inject_bug(source, rng):
    """one random single-edit mutant of `source`, or None when there is nothing to mutate."""
    tree = ast.parse(source)
    sites = bug_sites(tree)
    if not sites:
        return None
    target, replacement = sites[rng.randrange(len(sites))]
    tree = _Replace(target, replacement).visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


class _Refactor(ast.NodeTransformer):
    # behaviour preserving rewrites, each applied with probability p
    def __init__(self, rng, p):
        self.rng = rng
        self.p = p
        self.applied = 0

    def _roll(self):
        return self.rng.random() < self.p

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and self._roll():
            self.applied += 1
            load = ast.Name(id=node.target.id, ctx=ast.Load())
            return ast.Assign(targets=[node.target], value=ast.BinOp(left=load, op=node.op, right=node.value))
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        value = node.value
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(value, ast.BinOp)
                and isinstance(value.left, ast.Name) and value.left.id == node.targets[0].id and self._roll()):
            self.applied += 1
            return ast.AugAssign(target=node.targets[0], op=value.op, value=value.right)
        return node

    def visit_IfExp(self, node):
        self.generic_visit(node)
        if self._roll():
            self.applied += 1
            return ast.IfExp(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=node.orelse, orelse=node.body)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and type(node.ops[0]) in _FLIP and self._roll():
            self.applied += 1
            return ast.Compare(left=node.comparators[0], ops=[_FLIP[type(node.ops[0])]()], comparators=[node.left])
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if node.orelse and not (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If)) and self._roll():
            self.applied += 1
            return ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test), body=node.orelse, orelse=node.body)
        return node


def refactor(source, rng, p=0.6):
    tree = ast.parse(source)
    tx = _Refactor(rng, p)
    tree = tx.visit(tree)
    if not tx.applied:
        return None
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"
