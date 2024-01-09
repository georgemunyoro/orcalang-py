from ctypes import CFUNCTYPE, c_int32
from codegen import CodeGenerator
import llvmlite.binding as llvm
import subprocess
from uuid import uuid4


def create_execution_engine() -> llvm.ExecutionEngine:
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    backing_mod = llvm.parse_assembly("")
    return llvm.create_mcjit_compiler(backing_mod, target_machine)


def compile_ir(engine: llvm.ExecutionEngine, llvm_ir: str):
    mod = llvm.parse_assembly(llvm_ir)
    mod.verify()

    engine.add_module(mod)
    engine.finalize_object()
    engine.run_static_constructors()
    return mod


def compile(cg: CodeGenerator):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_default_triple()
    machine = target.create_target_machine()

    mod = llvm.parse_assembly(str(cg.module))
    modfilepath = f"/tmp/{uuid4()}.o"

    with llvm.create_mcjit_compiler(mod, machine) as mcjit:

        def on_compiled(_, objbytes):
            open(modfilepath, "wb").write(objbytes)

        mcjit.set_object_cache(on_compiled, lambda m: None)
        mcjit.finalize_object()

    subprocess.run(["gcc", "-o", "a.out", modfilepath])
    subprocess.run(["rm", modfilepath])
