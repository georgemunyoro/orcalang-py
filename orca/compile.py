from ctypes import CFUNCTYPE, c_int32
from orca.codegen import CodeGenerator
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


def compile(cg: CodeGenerator, outfilepath: str):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_default_triple()
    machine = target.create_target_machine()

    mod = llvm.parse_assembly(str(cg.module))
    mod.verify()

    modfilepath = f"/tmp/{uuid4()}.o"

    pm = llvm.ModulePassManager()

    pm.add_global_optimizer_pass()
    pm.add_instruction_combining_pass()
    pm.add_aggressive_dead_code_elimination_pass()
    pm.add_arg_promotion_pass()
    pm.add_strip_symbols_pass()
    pm.add_constant_merge_pass()
    pm.add_global_dce_pass()
    pm.add_global_optimizer_pass()
    pm.add_partial_inlining_pass()
    pm.add_gvn_pass()
    pm.add_jump_threading_pass()
    pm.add_lcssa_pass()
    pm.add_licm_pass()
    pm.add_loop_deletion_pass()
    pm.add_loop_extractor_pass()
    pm.add_loop_unroll_pass()
    pm.add_loop_rotate_pass()
    pm.add_loop_simplification_pass()
    pm.add_loop_unroll_and_jam_pass()
    pm.add_loop_unswitch_pass()
    pm.add_memcpy_optimization_pass()
    pm.add_sroa_pass()
    pm.add_cfg_simplification_pass()
    pm.add_sink_pass()

    pm.run(mod)

    with llvm.create_mcjit_compiler(mod, machine) as mcjit:

        def on_compiled(_, objbytes):
            open(modfilepath, "wb").write(objbytes)

        mcjit.set_object_cache(on_compiled, lambda m: None)
        mcjit.finalize_object()

    subprocess.run(["gcc", "-O3", "-flto", "-o", outfilepath, modfilepath])
    subprocess.run(["rm", modfilepath])
