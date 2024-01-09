from typing import List
from lark import Tree, Token
from lark.visitors import Interpreter
from llvmlite import ir, binding
from collections import OrderedDict

from scope import Scope


int_t = ir.IntType(32)

i8_t = ir.IntType(8)
i16_t = ir.IntType(16)
i32_t = ir.IntType(32)
i64_t = ir.IntType(64)

bool_t = ir.IntType(1)
char_t = ir.IntType(8)
double = ir.DoubleType()
void_t = ir.VoidType()


class AstNode:
    ...


class FunctionDefinition(AstNode):
    name: Token
    params: Tree
    return_type: Tree
    body: Tree

    def __init__(self, values: Tree):
        self.name = values.children[0]
        self.params = values.children[1] if len(values.children) == 4 else None
        self.return_type = values.children[len(values.children) - 2]
        self.body = values.children[len(values.children) - 1]


class CodeGenerator(Interpreter):
    module: ir.Module
    symtable = dict()
    scope = Scope[ir.AllocaInstr]()
    id_map = dict({})
    struct_map = dict({})
    native_functions = dict({})
    vtable = dict({})

    def __init__(self):
        self.module = ir.Module(name=__file__)
        self.builder = None

    def program(self, values: Tree):
        for stmt in values.children:
            self.visit(stmt)

    def struct_declaration(self, struct_decl: Tree):
        struct_type = self.module.context.get_identified_type(struct_decl.children[0])

        s_fields = OrderedDict()
        self.struct_map[struct_decl.children[0]] = dict({})

        for i, decl in enumerate(struct_decl.children[1].children):
            field_name = decl.children[0].value
            field_type = self.visit(decl.children[1])
            s_fields[field_name] = field_type

            self.struct_map[struct_decl.children[0]][field_name] = i

        struct_type.set_body(*s_fields.values())

    def variable_declaration(self, var_decl: Tree):
        var_name = var_decl.children[0]
        declared_type = self.visit(var_decl.children[1])
        initializer: ir.Value = self.visit(var_decl.children[2])

        if declared_type != initializer.type:
            raise Exception(
                f"When initializing variable '{var_name}', declared type to be {str(declared_type)}, but initializer is of type {str(initializer.type)}"
            )

        alloca = self.builder.alloca(declared_type, 1, var_name)
        self.builder.store(initializer, alloca)
        self.scope.insert(var_name, alloca)

    def array_variable_declaration(self, arr_var_decl: Tree):
        name = arr_var_decl.children[0].value
        arr_size = int(arr_var_decl.children[1])
        arr_el_type: ir.Type = self.visit(arr_var_decl.children[2])
        arr_type = ir.ArrayType(arr_el_type, arr_size)

        alloca = self.builder.alloca(typ=arr_type, size=1, name=name)
        self.scope.insert(name, alloca)
        return alloca

    def unary_expression(self, unary_expr: Tree):
        match unary_expr.children[0].data:
            case "deref":
                return self.builder.load(self.visit(unary_expr.children[1]))
            case "ref":
                return self.visit(unary_expr.children[1]).operands[0]

    def index(self, index: Tree):
        indexee_ptr: ir.LoadInstr = self.visit(index.children[0])
        index = self.visit(index.children[1])
        ptr = self.builder.gep(indexee_ptr, [index])
        return self.builder.load(ptr, name="idx")

    def assignment_expression(self, assign_expr: Tree):
        load: ir.LoadInstr = self.visit(assign_expr.children[0])
        return self.builder.store(
            self.visit(assign_expr.children[2]),
            load.operands[0],
        )

    def identifier(self, id: Tree):
        name = id.children[0].value
        alloca = self.scope.get(name)

        if alloca is None:
            for f in self.module.functions:
                if f.name == name:
                    return f

            raise Exception(f"Referenced unknown variable '{name}'.")

        return self.builder.load(self.scope.get(name))

    def static_access(self, values: Tree):
        ty = self.visit(values.children[0])
        accessor = values.children[1].value
        return self.vtable[ty][accessor]

    def normalized_type_name(self, t: ir.Type):
        if isinstance(t, ir.IdentifiedStructType):
            return t.name

        raise Exception("Unhandled type: ", t)

    def method_definition(self, values: Tree):
        ty = values.children[0]
        method_name = values.children[1].value
        params = values.children[2] if len(values.children) == 5 else None
        return_type = values.children[len(values.children) - 2]
        body = values.children[len(values.children) - 1]

        # tp = self.visit(ty)

        llvm_method_name = f"{self.normalized_type_name(self.visit(ty))}_{method_name}"

        args = (
            CodeGenerator.flatten_parameter_type_list(params)
            if params is not None
            else []
        )

        arg_types = [self.visit(arg.children[1]) for arg in args]
        arg_names = [arg.children[0].value for arg in args]

        func_type = ir.FunctionType(return_type=self.visit(return_type), args=arg_types)
        func = ir.Function(self.module, func_type, llvm_method_name)

        ty_to_attach_to = self.visit(ty)
        if ty_to_attach_to not in self.vtable:
            self.vtable[ty_to_attach_to] = dict({})
        self.vtable[ty_to_attach_to][method_name] = func

        entry_block = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_block)

        self.scope = Scope(self.scope)

        for i, arg in enumerate(func.args):
            arg._set_name(arg_names[i])
            alloca = self.builder.alloca(arg.type, name=arg._get_name())
            self.builder.store(arg, alloca)
            self.scope.insert(arg.name, alloca)

        self.visit(body)

        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        self.scope = self.scope.parent
        return func

    def function_definition(self, values: Tree):
        node = FunctionDefinition(values)

        args = (
            CodeGenerator.flatten_parameter_type_list(node.params)
            if node.params is not None
            else []
        )
        arg_types = [self.visit(arg.children[1]) for arg in args]
        arg_names = [arg.children[0].value for arg in args]
        func_type = ir.FunctionType(
            return_type=self.visit(node.return_type), args=arg_types
        )
        func = ir.Function(self.module, func_type, node.name)

        entry_block = func.append_basic_block("entry")
        self.builder = ir.IRBuilder(entry_block)

        self.scope = Scope(self.scope)

        for i, arg in enumerate(func.args):
            arg._set_name(arg_names[i])
            alloca = self.builder.alloca(arg.type, name=arg._get_name())
            self.builder.store(arg, alloca)
            self.scope.insert(arg.name, alloca)

        self.visit(node.body)

        if not self.builder.block.is_terminated:
            self.builder.ret_void()

        self.scope = self.scope.parent
        return func

    def while_statement(self, while_stmt: Tree):
        w_cond = self.builder.append_basic_block(name=self.get_repetitive_id("w_cond"))
        w_body = self.builder.append_basic_block(name=self.get_repetitive_id("w_body"))
        w_else = self.builder.append_basic_block(name=self.get_repetitive_id("w_else"))

        self.builder.branch(w_cond)

        self.builder.position_at_start(w_cond)
        self.builder.cbranch(self.visit(while_stmt.children[0]), w_body, w_else)

        self.builder.position_at_start(w_body)
        self.visit(while_stmt.children[1])
        self.builder.branch(w_cond)

        self.builder.position_at_start(w_else)

    def parameter_type_list(self, values: Tree):
        params = CodeGenerator.flatten_parameter_type_list(values)
        param_t = tuple(self.visit(p.children[1]) for p in params)
        return param_t

    def get_size_of(self, val: ir.Type):
        return val.get_abi_size(binding.create_target_data(self.module.data_layout))

    def cast_expression(self, cast_expr: Tree):
        val = self.visit(cast_expr.children[1])
        type_to_cast_to: ir.Type = self.visit(cast_expr.children[0])

        if type_to_cast_to.is_pointer and not val.type.is_pointer:
            return self.builder.inttoptr(val, type_to_cast_to)

        if self.get_size_of(val.type) > self.get_size_of(type_to_cast_to):
            return self.builder.trunc(val, type_to_cast_to)

        if self.get_size_of(val.type) < self.get_size_of(type_to_cast_to):
            return self.builder.sext(val, type_to_cast_to)

        return self.builder.bitcast(val, self.visit(cast_expr.children[0]))

    def type(self, values: Tree):
        match values.children[0].data:
            case "int":
                return int_t
            case "i8":
                return i8_t
            case "i16":
                return i16_t
            case "i32":
                return i32_t
            case "i64":
                return i64_t

            case "bool":
                return bool_t

            case "char":
                return char_t

            case "void":
                return void_t

            case "user_type":
                typename = values.children[0].children[0].value
                return self.module.context.get_identified_type(typename)

        if values.children[0].data == "type":
            t: ir.Type = self.visit(values.children[0]).as_pointer()
            return t

        raise Exception(f"Unhandled type: {values}")

    def compound_statement(self, values: Tree):
        self.scope = Scope(self.scope)
        for statement in values.children:
            self.visit(statement)
        self.scope = self.scope.parent

    def sizeof_type_expr(self, sizeof_type_expr: Tree):
        size_of_type = self.get_size_of(self.visit(sizeof_type_expr.children[0]))
        return i32_t(size_of_type)

    def pointer_access(self, ptr_access_expr: Tree):
        access = Tree(
            "access",
            [
                Tree(
                    Token("RULE", "primary_expression"),
                    [
                        Tree(
                            Token("RULE", "unary_expression"),
                            [
                                Tree("deref", []),
                                # Tree("identifier", [Token("IDENTIFIER", "iv")]),
                                ptr_access_expr.children[0],
                            ],
                        )
                    ],
                ),
                Token("IDENTIFIER", ptr_access_expr.children[1].value),
            ],
        )

        return self.visit(access)

        # quit()

        # accessee = self.builder.load(self.visit(ptr_access_expr.children[0]))
        # accessor = ptr_access_expr.children[1].value

        # field_map = self.struct_map[accessee.type.name]
        # accessor_index = field_map[accessor]

        # ptr = self.builder.gep(accessee.operands[0], [i32_t(0), i32_t(accessor_index)])
        # return self.builder.load(ptr)

    def access(self, access_expr: Tree):
        accessee = self.visit(access_expr.children[0])
        accessor = access_expr.children[1].value

        field_map = self.struct_map[accessee.type.name]
        accessor_index = field_map[accessor]

        gepper = accessee

        if isinstance(gepper, ir.LoadInstr) or isinstance(gepper, ir.AllocaInstr):
            gepper = accessee.operands[0]

        if not gepper.type.is_pointer:
            gepper = self.builder.alloca(gepper.type)

        ptr = self.builder.gep(gepper, [i32_t(0), i32_t(accessor_index)])
        return self.builder.load(ptr)

    def function_call(self, call: Tree):
        args: List[Tree] = []
        curr = call.children[1] if len(call.children) == 2 else None
        if curr is not None:
            while len(curr.children) == 2:
                args.insert(0, curr.children[1])
                curr = curr.children[0]
            args.insert(0, curr.children[0])

        fn = None

        if call.children[0].data != "identifier":
            if call.children[0].data == "access":
                val = self.visit(call.children[0].children[0])
                method: ir.Function = self.vtable[val.type][
                    call.children[0].children[1].value
                ]
                return self.builder.call(
                    method, [val.operands[0]] + [self.visit(arg) for arg in args]
                )
            elif call.children[0].data == "pointer_access":
                val = self.builder.load(self.visit(call.children[0].children[0]))
                method: ir.Function = self.vtable[val.type][
                    call.children[0].children[1].value
                ]
                return self.builder.call(
                    method, [val.operands[0]] + [self.visit(arg) for arg in args]
                )

            fn = self.visit(call.children[0])
        else:
            if call.children[0].children[0].value in self.struct_map:
                typename = call.children[0].children[0].value
                t: ir.BaseStructType = self.module.context.get_identified_type(typename)
                field_map = self.struct_map[typename]

                field_values = []
                for arg in args:
                    field_name = arg.children[0].value
                    field_value = self.visit(arg.children[1])
                    field_values.insert(field_map[field_name], field_value)

                return ir.Constant(t, field_values)

            if call.children[0].children[0].value == "printf":
                if "printf" not in self.native_functions:
                    voidptr_ty = char_t.as_pointer()
                    printf_ty = ir.FunctionType(
                        ir.IntType(32), [voidptr_ty], var_arg=True
                    )
                    self.native_functions["printf"] = ir.Function(
                        self.module, printf_ty, name="printf"
                    )

                return self.builder.call(
                    self.native_functions["printf"],
                    [self.visit(arg) for arg in args],
                )

            elif call.children[0].children[0].value == "free":
                if "free" not in self.native_functions:
                    free_ty = ir.FunctionType(void_t, [i32_t.as_pointer()])
                    self.native_functions["free"] = ir.Function(
                        self.module, free_ty, name="free"
                    )

                return self.builder.call(
                    self.native_functions["free"],
                    [self.visit(arg) for arg in args],
                )

            elif call.children[0].children[0].value == "malloc":
                if "malloc" not in self.native_functions:
                    voidptr_ty = ir.IntType(32).as_pointer()
                    malloc_ty = ir.FunctionType(voidptr_ty, [ir.IntType(32)])
                    self.native_functions["malloc"] = ir.Function(
                        self.module, malloc_ty, name="malloc"
                    )

                return self.builder.call(
                    self.native_functions["malloc"],
                    [self.visit(args[0])],
                )

            else:
                fn = self.visit(call.children[0])

        args = [self.visit(arg) for arg in args]

        return self.builder.call(fn=fn, args=args)

    def mul_expr(self, values: Tree):
        return self.builder.mul(
            lhs=self.visit(values.children[0]),
            rhs=self.visit(values.children[1]),
            name="mul_tmp",
        )

    def div_expr(self, values: Tree):
        return self.builder.sdiv(
            lhs=self.visit(values.children[0]),
            rhs=self.visit(values.children[1]),
            name="div_tmp",
        )

    def le_expr(self, le_expr: Tree):
        return self.builder.icmp_signed(
            "<=",
            lhs=self.visit(le_expr.children[0]),
            rhs=self.visit(le_expr.children[1]),
            name="le_tmp",
        )

    def lt_expr(self, lt_expr: Tree):
        return self.builder.icmp_signed(
            "<",
            lhs=self.visit(lt_expr.children[0]),
            rhs=self.visit(lt_expr.children[1]),
            name="lt_tmp",
        )

    def ge_expr(self, ge_expr: Tree):
        return self.builder.icmp_signed(
            ">=",
            lhs=self.visit(ge_expr.children[0]),
            rhs=self.visit(ge_expr.children[1]),
            name="ge_tmp",
        )

    def gt_expr(self, gt_expr: Tree):
        return self.builder.icmp_signed(
            ">",
            lhs=self.visit(gt_expr.children[0]),
            rhs=self.visit(gt_expr.children[1]),
            name="gt_tmp",
        )

    def eq_expr(self, eq_expr: Tree):
        return self.builder.icmp_signed(
            "==",
            lhs=self.visit(eq_expr.children[0]),
            rhs=self.visit(eq_expr.children[1]),
            name="eq_tmp",
        )

    def ne_expr(self, ne_expr: Tree):
        return self.builder.icmp_signed(
            "!=",
            lhs=self.visit(ne_expr.children[0]),
            rhs=self.visit(ne_expr.children[1]),
            name="ne_tmp",
        )

    def if_else_stmt(self, if_else_stmt: Tree):
        with self.builder.if_else(self.visit(if_else_stmt.children[0])) as (
            then,
            otherwise,
        ):
            with then:
                self.visit(if_else_stmt.children[1])
            with otherwise:
                self.visit(if_else_stmt.children[2])

    def if_stmt(self, if_stmt: Tree):
        with self.builder.if_else(self.visit(if_stmt.children[0])) as (then, otherwise):
            with then:
                self.visit(if_stmt.children[1])
            with otherwise:
                ...

    def return_statement(self, ret_stmt: Tree):
        if len(ret_stmt.children) == 0:
            return self.builder.ret_void()

        return self.builder.ret(self.visit(ret_stmt.children[0]))

    def expression_statement(self, expr_stmt: Tree):
        self.visit(expr_stmt.children[0])

    def add_expr(self, add_expr: Tree):
        return self.builder.add(
            lhs=self.visit(add_expr.children[0]),
            rhs=self.visit(add_expr.children[1]),
            name="addtmp",
        )

    def sub_expr(self, sub_expr: Tree):
        return self.builder.sub(
            lhs=self.visit(sub_expr.children[0]),
            rhs=self.visit(sub_expr.children[1]),
            name="subtmp",
        )

    def get_repetitive_id(self, key: str) -> str:
        if key in self.id_map:
            self.id_map[key] += 1
        else:
            self.id_map[key] = 0
        return f"{key}{self.id_map[key]}"

    def constant_string(self, constant_str: Tree):
        s: str = constant_str.children[0].value[1:-1]
        if not s.endswith("\0"):
            s += "\0"

        s = bytearray(s.encode("latin-1").decode("unicode_escape").encode("utf-8"))

        s_value = ir.Constant(ir.ArrayType(char_t, len(s)), s)
        global_s = ir.GlobalVariable(
            self.module, s_value.type, name=self.get_repetitive_id("str_")
        )
        global_s.linkage = "internal"
        global_s.global_constant = True
        global_s.initializer = s_value

        return self.builder.bitcast(global_s, char_t.as_pointer())

    def primary_expression(self, primary_expr: Tree):
        assert len(primary_expr.children) == 1
        return self.visit(primary_expr.children[0])

    def constant_true(self, _: Tree):
        return ir.Constant(ir.IntType(1), 1)

    def constant_false(self, _: Tree):
        return ir.Constant(ir.IntType(1), 0)

    def constant(self, constant: Tree):
        tok: Token = constant.children[0]

        match tok.type:
            case "INT":
                return ir.Constant(int_t, int(tok.value))

        raise Exception(f"Unhandled constant kind: {tok.type}")

    # ... UTILS ...

    @staticmethod
    def flatten_parameter_type_list(values: Tree) -> List[Tree]:
        params: List[Tree] = []
        curr = values
        while len(curr.children) == 2:
            params.insert(0, curr.children[1])
            curr = curr.children[0]
        params.insert(0, curr.children[0])
        return params
