from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class Pipeline:
    _pipeline: list[str] = []

    def __init__(self, pipeline=None, wrapper=None):
        if pipeline is None:
            pipeline = []
        self._pipeline = pipeline
        self._wrapper = wrapper

    def WRAP(self, scope):
        assert (
            self._wrapper is None
        ), "you probably forgot to close a FUNC with a matching CNUF"
        self._wrapper = scope
        return self

    def UNWRAP(self, scope):
        assert self._wrapper == scope
        self._wrapper = None
        return self

    def FUNC(self):
        assert (
            self._wrapper is None
        ), "you probably forgot to close a FUNC with a matching CNUF"
        self._wrapper = "func.func"
        return self

    def CNUF(self):
        assert self._wrapper == "func.func"
        self._wrapper = None
        return self

    def SPIRV(self):
        assert (
            self._wrapper is None
        ), "you probably forgot to close a SPIRV with a matching VRIPS"
        self._wrapper = "spirv.module"
        return self

    def VRIPS(self):
        assert self._wrapper == "spirv.module"
        self._wrapper = None
        return self

    def GPU(self):
        assert (
            self._wrapper is None
        ), "you probably forgot to close a GPU with a matching VRIPS"
        self._wrapper = "gpu.module"
        return self

    def UPG(self):
        assert self._wrapper == "gpu.module"
        self._wrapper = None
        return self

    def materialize(self, module=True):
        pipeline_str = ",".join(self._pipeline)
        if module:
            pipeline_str = f"builtin.module({pipeline_str})"
        logger.debug(f"{pipeline_str}")
        return pipeline_str

    def __add__(self, other: Pipeline):
        return Pipeline(self._pipeline + other._pipeline)

    def __iadd__(self, other: Pipeline):
        self._pipeline += other._pipeline
        return self

    def _add_pass(self, pass_name, **kwargs):
        kwargs = {
            k.replace("_", "-"): int(v) if isinstance(v, bool) else v
            for k, v in kwargs.items()
            if v is not None
        }
        if kwargs:
            args_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            pass_str = f"{pass_name}{{ {args_str} }}"
        else:
            pass_str = f"{pass_name}"
        if self._wrapper is not None:
            pass_str = f"{self._wrapper}({pass_str})"
        self._pipeline.append(pass_str)

    def lower_to_llvm_(self):
        return any(["to-llvm" in p for p in self._pipeline])

    def bufferize(self):
        return (
            self.FUNC()
            .scf_bufferize()
            .empty_tensor_to_alloc_tensor()
            .linalg_bufferize()
            .CNUF()
            .func_bufferize()
            .arith_bufferize()
            .FUNC()
            .tensor_bufferize()
            .finalizing_bufferize()
            .buffer_deallocation()
            .CNUF()
        )

    def lower_to_llvm(self):
        return (
            self.cse()
            .FUNC()
            .lower_affine()
            .arith_expand()
            .convert_math_to_llvm()
            .CNUF()
            .convert_math_to_libm()
            .convert_linalg_to_llvm()
            .expand_strided_metadata()
            .finalize_memref_to_llvm()
            .convert_scf_to_cf()
            .convert_cf_to_llvm()
            .cse()
            .lower_affine()
            .FUNC()
            .convert_arith_to_llvm()
            .CNUF()
            .convert_func_to_llvm()
            .canonicalize()
            .convert_openmp_to_llvm()
            .cse()
            .reconcile_unrealized_casts()
        )

    def lower_to_openmp(self):
        return self.convert_scf_to_openmp().FUNC().lower_affine().CNUF()

    def sparse_compiler(
        self,
        parallelization_strategy=None,
        enable_runtime_library=None,
        enable_buffer_initialization=None,
        vl=None,
        s2s_strategy=None,
        reassociate_fp_reductions=None,
        enable_index_optimizations=None,
        enable_amx=None,
        enable_arm_neon=None,
        enable_arm_sve=None,
        enable_x86vector=None,
    ):
        self._add_pass(
            "sparse-compiler",
            parallelization_strategy=parallelization_strategy,
            enable_runtime_library=enable_runtime_library,
            enable_buffer_initialization=enable_buffer_initialization,
            vl=vl,
            s2s_strategy=s2s_strategy,
            reassociate_fp_reductions=reassociate_fp_reductions,
            enable_index_optimizations=enable_index_optimizations,
            enable_amx=enable_amx,
            enable_arm_neon=enable_arm_neon,
            enable_arm_sve=enable_arm_sve,
            enable_x86vector=enable_x86vector,
        )
        return self

    def lower_to_vulkan(self, index_bitwidth=None):
        return (
            self.gpu_kernel_outlining()
            .fold_memref_alias_ops()
            .convert_gpu_to_spirv()
            .SPIRV()
            .spirv_lower_abi_attrs()
            .spirv_update_vce()
            .VRIPS()
            .convert_gpu_launch_to_vulkan_launch()
            .finalize_memref_to_llvm()
            .FUNC()
            .llvm_request_c_wrappers()
            .CNUF()
            .convert_func_to_llvm(index_bitwidth=index_bitwidth)
            .reconcile_unrealized_casts()
            .launch_func_to_vulkan()
        )

    ############################
    # autogen starts
    ############################

    def add_outer_parallel_loop(self):
        self._add_pass("add-outer-parallel-loop")
        return self

    def affine_data_copy_generate(
        self,
        fast_mem_capacity=None,
        fast_mem_space=None,
        generate_dma=None,
        min_dma_transfer=None,
        skip_non_unit_stride_loops=None,
        slow_mem_space=None,
        tag_mem_space=None,
    ):
        if fast_mem_capacity is not None and isinstance(
            fast_mem_capacity, (list, tuple)
        ):
            fast_mem_capacity = ",".join(map(str, fast_mem_capacity))
        if fast_mem_space is not None and isinstance(fast_mem_space, (list, tuple)):
            fast_mem_space = ",".join(map(str, fast_mem_space))
        if min_dma_transfer is not None and isinstance(min_dma_transfer, (list, tuple)):
            min_dma_transfer = ",".join(map(str, min_dma_transfer))
        if slow_mem_space is not None and isinstance(slow_mem_space, (list, tuple)):
            slow_mem_space = ",".join(map(str, slow_mem_space))
        if tag_mem_space is not None and isinstance(tag_mem_space, (list, tuple)):
            tag_mem_space = ",".join(map(str, tag_mem_space))
        self._add_pass(
            "affine-data-copy-generate",
            fast_mem_capacity=fast_mem_capacity,
            fast_mem_space=fast_mem_space,
            generate_dma=generate_dma,
            min_dma_transfer=min_dma_transfer,
            skip_non_unit_stride_loops=skip_non_unit_stride_loops,
            slow_mem_space=slow_mem_space,
            tag_mem_space=tag_mem_space,
        )
        return self

    def affine_expand_index_ops(self):
        self._add_pass("affine-expand-index-ops")
        return self

    def affine_loop_coalescing(self):
        self._add_pass("affine-loop-coalescing")
        return self

    def affine_loop_fusion(
        self,
        fusion_compute_tolerance=None,
        fusion_fast_mem_space=None,
        fusion_local_buf_threshold=None,
        fusion_maximal=None,
        mode=None,
    ):
        if fusion_compute_tolerance is not None and isinstance(
            fusion_compute_tolerance, (list, tuple)
        ):
            fusion_compute_tolerance = ",".join(map(str, fusion_compute_tolerance))
        if fusion_fast_mem_space is not None and isinstance(
            fusion_fast_mem_space, (list, tuple)
        ):
            fusion_fast_mem_space = ",".join(map(str, fusion_fast_mem_space))
        if fusion_local_buf_threshold is not None and isinstance(
            fusion_local_buf_threshold, (list, tuple)
        ):
            fusion_local_buf_threshold = ",".join(map(str, fusion_local_buf_threshold))
        if mode is not None and isinstance(mode, (list, tuple)):
            mode = ",".join(map(str, mode))
        self._add_pass(
            "affine-loop-fusion",
            fusion_compute_tolerance=fusion_compute_tolerance,
            fusion_fast_mem_space=fusion_fast_mem_space,
            fusion_local_buf_threshold=fusion_local_buf_threshold,
            fusion_maximal=fusion_maximal,
            mode=mode,
        )
        return self

    def affine_loop_invariant_code_motion(self):
        self._add_pass("affine-loop-invariant-code-motion")
        return self

    def affine_loop_normalize(self, promote_single_iter=None):
        self._add_pass("affine-loop-normalize", promote_single_iter=promote_single_iter)
        return self

    def affine_loop_tile(
        self, cache_size=None, separate=None, tile_size=None, tile_sizes=None
    ):
        if cache_size is not None and isinstance(cache_size, (list, tuple)):
            cache_size = ",".join(map(str, cache_size))
        if tile_size is not None and isinstance(tile_size, (list, tuple)):
            tile_size = ",".join(map(str, tile_size))
        if tile_sizes is not None and isinstance(tile_sizes, (list, tuple)):
            tile_sizes = ",".join(map(str, tile_sizes))
        self._add_pass(
            "affine-loop-tile",
            cache_size=cache_size,
            separate=separate,
            tile_size=tile_size,
            tile_sizes=tile_sizes,
        )
        return self

    def affine_loop_unroll(
        self,
        cleanup_unroll=None,
        unroll_factor=None,
        unroll_full=None,
        unroll_full_threshold=None,
        unroll_num_reps=None,
        unroll_up_to_factor=None,
    ):
        if unroll_factor is not None and isinstance(unroll_factor, (list, tuple)):
            unroll_factor = ",".join(map(str, unroll_factor))
        if unroll_full_threshold is not None and isinstance(
            unroll_full_threshold, (list, tuple)
        ):
            unroll_full_threshold = ",".join(map(str, unroll_full_threshold))
        if unroll_num_reps is not None and isinstance(unroll_num_reps, (list, tuple)):
            unroll_num_reps = ",".join(map(str, unroll_num_reps))
        self._add_pass(
            "affine-loop-unroll",
            cleanup_unroll=cleanup_unroll,
            unroll_factor=unroll_factor,
            unroll_full=unroll_full,
            unroll_full_threshold=unroll_full_threshold,
            unroll_num_reps=unroll_num_reps,
            unroll_up_to_factor=unroll_up_to_factor,
        )
        return self

    def affine_loop_unroll_jam(self, unroll_jam_factor=None):
        if unroll_jam_factor is not None and isinstance(
            unroll_jam_factor, (list, tuple)
        ):
            unroll_jam_factor = ",".join(map(str, unroll_jam_factor))
        self._add_pass("affine-loop-unroll-jam", unroll_jam_factor=unroll_jam_factor)
        return self

    def affine_parallelize(self, max_nested=None, parallel_reductions=None):
        if max_nested is not None and isinstance(max_nested, (list, tuple)):
            max_nested = ",".join(map(str, max_nested))
        self._add_pass(
            "affine-parallelize",
            max_nested=max_nested,
            parallel_reductions=parallel_reductions,
        )
        return self

    def affine_pipeline_data_transfer(self):
        self._add_pass("affine-pipeline-data-transfer")
        return self

    def affine_scalrep(self):
        self._add_pass("affine-scalrep")
        return self

    def affine_simplify_structures(self):
        self._add_pass("affine-simplify-structures")
        return self

    def affine_super_vectorize(
        self,
        test_fastest_varying=None,
        vectorize_reductions=None,
        virtual_vector_size=None,
    ):
        if test_fastest_varying is not None and isinstance(
            test_fastest_varying, (list, tuple)
        ):
            test_fastest_varying = ",".join(map(str, test_fastest_varying))
        if virtual_vector_size is not None and isinstance(
            virtual_vector_size, (list, tuple)
        ):
            virtual_vector_size = ",".join(map(str, virtual_vector_size))
        self._add_pass(
            "affine-super-vectorize",
            test_fastest_varying=test_fastest_varying,
            vectorize_reductions=vectorize_reductions,
            virtual_vector_size=virtual_vector_size,
        )
        return self

    def arith_bufferize(self, alignment=None):
        if alignment is not None and isinstance(alignment, (list, tuple)):
            alignment = ",".join(map(str, alignment))
        self._add_pass("arith-bufferize", alignment=alignment)
        return self

    def arith_emulate_wide_int(self, widest_int_supported=None):
        if widest_int_supported is not None and isinstance(
            widest_int_supported, (list, tuple)
        ):
            widest_int_supported = ",".join(map(str, widest_int_supported))
        self._add_pass(
            "arith-emulate-wide-int", widest_int_supported=widest_int_supported
        )
        return self

    def arith_expand(self):
        self._add_pass("arith-expand")
        return self

    def arith_unsigned_when_equivalent(self):
        self._add_pass("arith-unsigned-when-equivalent")
        return self

    def arm_neon_2d_to_intr(self):
        self._add_pass("arm-neon-2d-to-intr")
        return self

    def async_func_to_async_runtime(self):
        self._add_pass("async-func-to-async-runtime")
        return self

    def async_parallel_for(
        self, async_dispatch=None, min_task_size=None, num_workers=None
    ):
        if min_task_size is not None and isinstance(min_task_size, (list, tuple)):
            min_task_size = ",".join(map(str, min_task_size))
        if num_workers is not None and isinstance(num_workers, (list, tuple)):
            num_workers = ",".join(map(str, num_workers))
        self._add_pass(
            "async-parallel-for",
            async_dispatch=async_dispatch,
            min_task_size=min_task_size,
            num_workers=num_workers,
        )
        return self

    def async_runtime_policy_based_ref_counting(self):
        self._add_pass("async-runtime-policy-based-ref-counting")
        return self

    def async_runtime_ref_counting(self):
        self._add_pass("async-runtime-ref-counting")
        return self

    def async_runtime_ref_counting_opt(self):
        self._add_pass("async-runtime-ref-counting-opt")
        return self

    def async_to_async_runtime(self):
        self._add_pass("async-to-async-runtime")
        return self

    def buffer_deallocation(self):
        self._add_pass("buffer-deallocation")
        return self

    def buffer_hoisting(self):
        self._add_pass("buffer-hoisting")
        return self

    def buffer_host_register(self):
        self._add_pass("buffer-host-register")
        return self

    def buffer_loop_hoisting(self):
        self._add_pass("buffer-loop-hoisting")
        return self

    def buffer_results_to_out_params(self):
        self._add_pass("buffer-results-to-out-params")
        return self

    def bufferization_bufferize(self):
        self._add_pass("bufferization-bufferize")
        return self

    def canonicalize(
        self,
        disable_patterns=None,
        enable_patterns=None,
        max_iterations=None,
        max_num_rewrites=None,
        region_simplify=None,
        test_convergence=None,
        top_down=None,
    ):
        if disable_patterns is not None and isinstance(disable_patterns, (list, tuple)):
            disable_patterns = ",".join(map(str, disable_patterns))
        if enable_patterns is not None and isinstance(enable_patterns, (list, tuple)):
            enable_patterns = ",".join(map(str, enable_patterns))
        if max_iterations is not None and isinstance(max_iterations, (list, tuple)):
            max_iterations = ",".join(map(str, max_iterations))
        if max_num_rewrites is not None and isinstance(max_num_rewrites, (list, tuple)):
            max_num_rewrites = ",".join(map(str, max_num_rewrites))
        self._add_pass(
            "canonicalize",
            disable_patterns=disable_patterns,
            enable_patterns=enable_patterns,
            max_iterations=max_iterations,
            max_num_rewrites=max_num_rewrites,
            region_simplify=region_simplify,
            test_convergence=test_convergence,
            top_down=top_down,
        )
        return self

    def control_flow_sink(self):
        self._add_pass("control-flow-sink")
        return self

    def convert_affine_for_to_gpu(self, gpu_block_dims=None, gpu_thread_dims=None):
        if gpu_block_dims is not None and isinstance(gpu_block_dims, (list, tuple)):
            gpu_block_dims = ",".join(map(str, gpu_block_dims))
        if gpu_thread_dims is not None and isinstance(gpu_thread_dims, (list, tuple)):
            gpu_thread_dims = ",".join(map(str, gpu_thread_dims))
        self._add_pass(
            "convert-affine-for-to-gpu",
            gpu_block_dims=gpu_block_dims,
            gpu_thread_dims=gpu_thread_dims,
        )
        return self

    def convert_amdgpu_to_rocdl(self, chipset=None):
        if chipset is not None and isinstance(chipset, (list, tuple)):
            chipset = ",".join(map(str, chipset))
        self._add_pass("convert-amdgpu-to-rocdl", chipset=chipset)
        return self

    def convert_arith_to_llvm(self, index_bitwidth=None):
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass("convert-arith-to-llvm", index_bitwidth=index_bitwidth)
        return self

    def convert_arith_to_spirv(
        self, emulate_lt_32_bit_scalar_types=None, enable_fast_math=None
    ):
        self._add_pass(
            "convert-arith-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
            enable_fast_math=enable_fast_math,
        )
        return self

    def convert_async_to_llvm(self, use_opaque_pointers=None):
        self._add_pass("convert-async-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_bufferization_to_memref(self):
        self._add_pass("convert-bufferization-to-memref")
        return self

    def convert_cf_to_llvm(self, index_bitwidth=None, use_opaque_pointers=None):
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass(
            "convert-cf-to-llvm",
            index_bitwidth=index_bitwidth,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_cf_to_spirv(self, emulate_lt_32_bit_scalar_types=None):
        self._add_pass(
            "convert-cf-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_complex_to_libm(self):
        self._add_pass("convert-complex-to-libm")
        return self

    def convert_complex_to_llvm(self):
        self._add_pass("convert-complex-to-llvm")
        return self

    def convert_complex_to_standard(self):
        self._add_pass("convert-complex-to-standard")
        return self

    def convert_elementwise_to_linalg(self):
        self._add_pass("convert-elementwise-to-linalg")
        return self

    def convert_func_to_llvm(
        self,
        data_layout=None,
        index_bitwidth=None,
        use_bare_ptr_memref_call_conv=None,
        use_opaque_pointers=None,
    ):
        if data_layout is not None and isinstance(data_layout, (list, tuple)):
            data_layout = ",".join(map(str, data_layout))
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass(
            "convert-func-to-llvm",
            data_layout=data_layout,
            index_bitwidth=index_bitwidth,
            use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_func_to_spirv(self, emulate_lt_32_bit_scalar_types=None):
        self._add_pass(
            "convert-func-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_gpu_launch_to_vulkan_launch(self):
        self._add_pass("convert-gpu-launch-to-vulkan-launch")
        return self

    def convert_gpu_to_nvvm(
        self, has_redux=None, index_bitwidth=None, use_opaque_pointers=None
    ):
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass(
            "convert-gpu-to-nvvm",
            has_redux=has_redux,
            index_bitwidth=index_bitwidth,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_gpu_to_rocdl(
        self,
        chipset=None,
        index_bitwidth=None,
        runtime=None,
        use_bare_ptr_memref_call_conv=None,
        use_opaque_pointers=None,
    ):
        if chipset is not None and isinstance(chipset, (list, tuple)):
            chipset = ",".join(map(str, chipset))
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        if runtime is not None and isinstance(runtime, (list, tuple)):
            runtime = ",".join(map(str, runtime))
        self._add_pass(
            "convert-gpu-to-rocdl",
            chipset=chipset,
            index_bitwidth=index_bitwidth,
            runtime=runtime,
            use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_gpu_to_spirv(self, use_64bit_index=None):
        self._add_pass("convert-gpu-to-spirv", use_64bit_index=use_64bit_index)
        return self

    def convert_gpux_to_spirv(
        self,
        client_api=None,
        emulate_lt_32_bit_scalar_types=None,
        map_memory_space=None,
    ):
        if client_api is not None and isinstance(client_api, (list, tuple)):
            client_api = ",".join(map(str, client_api))
        self._add_pass(
            "convert-gpux-to-spirv",
            client_api=client_api,
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
            map_memory_space=map_memory_space,
        )
        return self

    def convert_index_to_llvm(self, index_bitwidth=None):
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass("convert-index-to-llvm", index_bitwidth=index_bitwidth)
        return self

    def convert_linalg_to_affine_loops(self):
        self._add_pass("convert-linalg-to-affine-loops")
        return self

    def convert_linalg_to_llvm(self, use_opaque_pointers=None):
        self._add_pass(
            "convert-linalg-to-llvm", use_opaque_pointers=use_opaque_pointers
        )
        return self

    def convert_linalg_to_loops(self):
        self._add_pass("convert-linalg-to-loops")
        return self

    def convert_linalg_to_parallel_loops(self):
        self._add_pass("convert-linalg-to-parallel-loops")
        return self

    def convert_linalg_to_std(self):
        self._add_pass("convert-linalg-to-std")
        return self

    def convert_math_to_funcs(self, min_width_of_fpowi_exponent=None):
        if min_width_of_fpowi_exponent is not None and isinstance(
            min_width_of_fpowi_exponent, (list, tuple)
        ):
            min_width_of_fpowi_exponent = ",".join(
                map(str, min_width_of_fpowi_exponent)
            )
        self._add_pass(
            "convert-math-to-funcs",
            min_width_of_fpowi_exponent=min_width_of_fpowi_exponent,
        )
        return self

    def convert_math_to_libm(self):
        self._add_pass("convert-math-to-libm")
        return self

    def convert_math_to_llvm(self, approximate_log1p=None):
        self._add_pass("convert-math-to-llvm", approximate_log1p=approximate_log1p)
        return self

    def convert_math_to_spirv(self):
        self._add_pass("convert-math-to-spirv")
        return self

    def convert_memref_to_spirv(self, bool_num_bits=None):
        if bool_num_bits is not None and isinstance(bool_num_bits, (list, tuple)):
            bool_num_bits = ",".join(map(str, bool_num_bits))
        self._add_pass("convert-memref-to-spirv", bool_num_bits=bool_num_bits)
        return self

    def convert_nvgpu_to_nvvm(self, use_opaque_pointers=None):
        self._add_pass("convert-nvgpu-to-nvvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_openacc_to_llvm(self, use_opaque_pointers=None):
        self._add_pass(
            "convert-openacc-to-llvm", use_opaque_pointers=use_opaque_pointers
        )
        return self

    def convert_openacc_to_scf(self):
        self._add_pass("convert-openacc-to-scf")
        return self

    def convert_openmp_to_llvm(self):
        self._add_pass("convert-openmp-to-llvm")
        return self

    def convert_parallel_loops_to_gpu(self):
        self._add_pass("convert-parallel-loops-to-gpu")
        return self

    def convert_pdl_to_pdl_interp(self):
        self._add_pass("convert-pdl-to-pdl-interp")
        return self

    def convert_scf_to_cf(self):
        self._add_pass("convert-scf-to-cf")
        return self

    def convert_scf_to_openmp(self, use_opaque_pointers=None):
        self._add_pass("convert-scf-to-openmp", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_scf_to_spirv(self):
        self._add_pass("convert-scf-to-spirv")
        return self

    def convert_shape_constraints(self):
        self._add_pass("convert-shape-constraints")
        return self

    def convert_shape_to_std(self):
        self._add_pass("convert-shape-to-std")
        return self

    def convert_spirv_to_llvm(self, use_opaque_pointers=None):
        self._add_pass("convert-spirv-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def convert_tensor_to_linalg(self):
        self._add_pass("convert-tensor-to-linalg")
        return self

    def convert_tensor_to_spirv(self, emulate_lt_32_bit_scalar_types=None):
        self._add_pass(
            "convert-tensor-to-spirv",
            emulate_lt_32_bit_scalar_types=emulate_lt_32_bit_scalar_types,
        )
        return self

    def convert_to_nvvm(self):
        self._add_pass("convert-to-nvvm")
        return self

    def convert_vector_to_gpu(self, use_nvgpu=None):
        self._add_pass("convert-vector-to-gpu", use_nvgpu=use_nvgpu)
        return self

    def convert_vector_to_llvm(
        self,
        enable_amx=None,
        enable_arm_neon=None,
        enable_arm_sve=None,
        enable_x86vector=None,
        force_32bit_vector_indices=None,
        reassociate_fp_reductions=None,
        use_opaque_pointers=None,
    ):
        self._add_pass(
            "convert-vector-to-llvm",
            enable_amx=enable_amx,
            enable_arm_neon=enable_arm_neon,
            enable_arm_sve=enable_arm_sve,
            enable_x86vector=enable_x86vector,
            force_32bit_vector_indices=force_32bit_vector_indices,
            reassociate_fp_reductions=reassociate_fp_reductions,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def convert_vector_to_scf(
        self, full_unroll=None, lower_tensors=None, target_rank=None
    ):
        if target_rank is not None and isinstance(target_rank, (list, tuple)):
            target_rank = ",".join(map(str, target_rank))
        self._add_pass(
            "convert-vector-to-scf",
            full_unroll=full_unroll,
            lower_tensors=lower_tensors,
            target_rank=target_rank,
        )
        return self

    def convert_vector_to_spirv(self):
        self._add_pass("convert-vector-to-spirv")
        return self

    def cse(self):
        self._add_pass("cse")
        return self

    def decorate_spirv_composite_type_layout(self):
        self._add_pass("decorate-spirv-composite-type-layout")
        return self

    def drop_equivalent_buffer_results(self):
        self._add_pass("drop-equivalent-buffer-results")
        return self

    def duplicate_function_elimination(self):
        self._add_pass("duplicate-function-elimination")
        return self

    def eliminate_empty_tensors(self):
        self._add_pass("eliminate-empty-tensors")
        return self

    def empty_tensor_to_alloc_tensor(self):
        self._add_pass("empty-tensor-to-alloc-tensor")
        return self

    def ensure_debug_info_scope_on_llvm_func(self):
        self._add_pass("ensure-debug-info-scope-on-llvm-func")
        return self

    def expand_strided_metadata(self):
        self._add_pass("expand-strided-metadata")
        return self

    def finalize_memref_to_llvm(
        self,
        index_bitwidth=None,
        use_aligned_alloc=None,
        use_generic_functions=None,
        use_opaque_pointers=None,
    ):
        if index_bitwidth is not None and isinstance(index_bitwidth, (list, tuple)):
            index_bitwidth = ",".join(map(str, index_bitwidth))
        self._add_pass(
            "finalize-memref-to-llvm",
            index_bitwidth=index_bitwidth,
            use_aligned_alloc=use_aligned_alloc,
            use_generic_functions=use_generic_functions,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def finalizing_bufferize(self):
        self._add_pass("finalizing-bufferize")
        return self

    def fold_memref_alias_ops(self):
        self._add_pass("fold-memref-alias-ops")
        return self

    def func_bufferize(self):
        self._add_pass("func-bufferize")
        return self

    def generate_runtime_verification(self):
        self._add_pass("generate-runtime-verification")
        return self

    def gpu_async_region(self):
        self._add_pass("gpu-async-region")
        return self

    def gpu_kernel_outlining(self, data_layout_str=None):
        if data_layout_str is not None and isinstance(data_layout_str, (list, tuple)):
            data_layout_str = ",".join(map(str, data_layout_str))
        self._add_pass("gpu-kernel-outlining", data_layout_str=data_layout_str)
        return self

    def gpu_launch_sink_index_computations(self):
        self._add_pass("gpu-launch-sink-index-computations")
        return self

    def gpu_map_parallel_loops(self):
        self._add_pass("gpu-map-parallel-loops")
        return self

    def gpu_to_cubin(
        self, chip=None, features=None, gpu_binary_annotation=None, triple=None
    ):
        if chip is not None and isinstance(chip, (list, tuple)):
            chip = ",".join(map(str, chip))
        if features is not None and isinstance(features, (list, tuple)):
            features = ",".join(map(str, features))
        if gpu_binary_annotation is not None and isinstance(
            gpu_binary_annotation, (list, tuple)
        ):
            gpu_binary_annotation = ",".join(map(str, gpu_binary_annotation))
        if triple is not None and isinstance(triple, (list, tuple)):
            triple = ",".join(map(str, triple))
        self._add_pass(
            "gpu-to-cubin",
            chip=chip,
            features=features,
            gpu_binary_annotation=gpu_binary_annotation,
            triple=triple,
        )
        return self

    def gpu_to_llvm(
        self,
        gpu_binary_annotation=None,
        use_bare_pointers_for_kernels=None,
        use_opaque_pointers=None,
    ):
        if gpu_binary_annotation is not None and isinstance(
            gpu_binary_annotation, (list, tuple)
        ):
            gpu_binary_annotation = ",".join(map(str, gpu_binary_annotation))
        self._add_pass(
            "gpu-to-llvm",
            gpu_binary_annotation=gpu_binary_annotation,
            use_bare_pointers_for_kernels=use_bare_pointers_for_kernels,
            use_opaque_pointers=use_opaque_pointers,
        )
        return self

    def inline(self, default_pipeline=None, max_iterations=None, op_pipelines=None):
        if default_pipeline is not None and isinstance(default_pipeline, (list, tuple)):
            default_pipeline = ",".join(map(str, default_pipeline))
        if max_iterations is not None and isinstance(max_iterations, (list, tuple)):
            max_iterations = ",".join(map(str, max_iterations))
        if op_pipelines is not None and isinstance(op_pipelines, (list, tuple)):
            op_pipelines = ",".join(map(str, op_pipelines))
        self._add_pass(
            "inline",
            default_pipeline=default_pipeline,
            max_iterations=max_iterations,
            op_pipelines=op_pipelines,
        )
        return self

    def insert_gpu_allocs(self, client_api=None):
        if client_api is not None and isinstance(client_api, (list, tuple)):
            client_api = ",".join(map(str, client_api))
        self._add_pass("insert-gpu-allocs", client_api=client_api)
        return self

    def int_range_optimizations(self):
        self._add_pass("int-range-optimizations")
        return self

    def launch_func_to_vulkan(self, use_opaque_pointers=None):
        self._add_pass("launch-func-to-vulkan", use_opaque_pointers=use_opaque_pointers)
        return self

    def linalg_bufferize(self):
        self._add_pass("linalg-bufferize")
        return self

    def linalg_detensorize(self, aggressive_mode=None):
        self._add_pass("linalg-detensorize", aggressive_mode=aggressive_mode)
        return self

    def linalg_fake_quantize(self, bits=None):
        if bits is not None and isinstance(bits, (list, tuple)):
            bits = ",".join(map(str, bits))
        self._add_pass("linalg-fake-quantize", bits=bits)
        return self

    def linalg_fold_unit_extent_dims(
        self, fold_one_trip_loops_only=None, use_rank_reducing_slices=None
    ):
        self._add_pass(
            "linalg-fold-unit-extent-dims",
            fold_one_trip_loops_only=fold_one_trip_loops_only,
            use_rank_reducing_slices=use_rank_reducing_slices,
        )
        return self

    def linalg_fuse_elementwise_ops(self):
        self._add_pass("linalg-fuse-elementwise-ops")
        return self

    def linalg_generalize_named_ops(self):
        self._add_pass("linalg-generalize-named-ops")
        return self

    def linalg_inline_scalar_operands(self):
        self._add_pass("linalg-inline-scalar-operands")
        return self

    def linalg_named_op_conversion(self):
        self._add_pass("linalg-named-op-conversion")
        return self

    def linalg_transform_patterns(
        self,
        bubble_up_extract_slice_op_pattern=None,
        conv2d_im2col=None,
        decompose_convolutions=None,
        erase_unnecessary_inputs=None,
        erase_unused_operands_and_results=None,
        generalize_pad_tensor=None,
        generalize_tensor_pack=None,
        generalize_tensor_unpack=None,
        linalg_to_vector_patterns=None,
        loop_type=None,
        patterns=None,
        peeled_loops=None,
        skip_partial=None,
        swap_extract_slice_with_fill_pattern=None,
        swap_subtensor_padtensor=None,
        tile_sizes=None,
        transform_pad_tensor=None,
        vector_transfer_forwarding_patterns=None,
    ):
        if loop_type is not None and isinstance(loop_type, (list, tuple)):
            loop_type = ",".join(map(str, loop_type))
        if peeled_loops is not None and isinstance(peeled_loops, (list, tuple)):
            peeled_loops = ",".join(map(str, peeled_loops))
        if tile_sizes is not None and isinstance(tile_sizes, (list, tuple)):
            tile_sizes = ",".join(map(str, tile_sizes))
        self._add_pass(
            "linalg-transform-patterns",
            bubble_up_extract_slice_op_pattern=bubble_up_extract_slice_op_pattern,
            conv2d_im2col=conv2d_im2col,
            decompose_convolutions=decompose_convolutions,
            erase_unnecessary_inputs=erase_unnecessary_inputs,
            erase_unused_operands_and_results=erase_unused_operands_and_results,
            generalize_pad_tensor=generalize_pad_tensor,
            generalize_tensor_pack=generalize_tensor_pack,
            generalize_tensor_unpack=generalize_tensor_unpack,
            linalg_to_vector_patterns=linalg_to_vector_patterns,
            loop_type=loop_type,
            patterns=patterns,
            peeled_loops=peeled_loops,
            skip_partial=skip_partial,
            swap_extract_slice_with_fill_pattern=swap_extract_slice_with_fill_pattern,
            swap_subtensor_padtensor=swap_subtensor_padtensor,
            tile_sizes=tile_sizes,
            transform_pad_tensor=transform_pad_tensor,
            vector_transfer_forwarding_patterns=vector_transfer_forwarding_patterns,
        )
        return self

    def llvm_legalize_for_export(self):
        self._add_pass("llvm-legalize-for-export")
        return self

    def llvm_optimize_for_nvvm_target(self):
        self._add_pass("llvm-optimize-for-nvvm-target")
        return self

    def llvm_request_c_wrappers(self):
        self._add_pass("llvm-request-c-wrappers")
        return self

    def llvmgpu_vector_lowering(self):
        self._add_pass("llvmgpu-vector-lowering")
        return self

    def loop_invariant_code_motion(self):
        self._add_pass("loop-invariant-code-motion")
        return self

    def lower_affine(self):
        self._add_pass("lower-affine")
        return self

    def lower_host_to_llvm(self, use_opaque_pointers=None):
        self._add_pass("lower-host-to-llvm", use_opaque_pointers=use_opaque_pointers)
        return self

    def lower_vector_mask(self):
        self._add_pass("lower-vector-mask")
        return self

    def map_memref_spirv_storage_class(self, client_api=None):
        if client_api is not None and isinstance(client_api, (list, tuple)):
            client_api = ",".join(map(str, client_api))
        self._add_pass("map-memref-spirv-storage-class", client_api=client_api)
        return self

    def memref_emulate_wide_int(self, widest_int_supported=None):
        if widest_int_supported is not None and isinstance(
            widest_int_supported, (list, tuple)
        ):
            widest_int_supported = ",".join(map(str, widest_int_supported))
        self._add_pass(
            "memref-emulate-wide-int", widest_int_supported=widest_int_supported
        )
        return self

    def memref_expand(self):
        self._add_pass("memref-expand")
        return self

    def nelli_map_memref_spirv_storage_class(self, client_api=None):
        if client_api is not None and isinstance(client_api, (list, tuple)):
            client_api = ",".join(map(str, client_api))
        self._add_pass("nelli-map-memref-spirv-storage-class", client_api=client_api)
        return self

    def normalize_memrefs(self):
        self._add_pass("normalize-memrefs")
        return self

    def nvgpu_optimize_shared_memory(self):
        self._add_pass("nvgpu-optimize-shared-memory")
        return self

    def one_shot_bufferize(
        self,
        allow_return_allocs=None,
        allow_unknown_ops=None,
        analysis_fuzzer_seed=None,
        analysis_heuristic=None,
        bufferize_function_boundaries=None,
        copy_before_write=None,
        create_deallocs=None,
        dialect_filter=None,
        function_boundary_type_conversion=None,
        must_infer_memory_space=None,
        no_analysis_func_filter=None,
        print_conflicts=None,
        test_analysis_only=None,
        unknown_type_conversion=None,
    ):
        if analysis_fuzzer_seed is not None and isinstance(
            analysis_fuzzer_seed, (list, tuple)
        ):
            analysis_fuzzer_seed = ",".join(map(str, analysis_fuzzer_seed))
        if analysis_heuristic is not None and isinstance(
            analysis_heuristic, (list, tuple)
        ):
            analysis_heuristic = ",".join(map(str, analysis_heuristic))
        if dialect_filter is not None and isinstance(dialect_filter, (list, tuple)):
            dialect_filter = ",".join(map(str, dialect_filter))
        if function_boundary_type_conversion is not None and isinstance(
            function_boundary_type_conversion, (list, tuple)
        ):
            function_boundary_type_conversion = ",".join(
                map(str, function_boundary_type_conversion)
            )
        if no_analysis_func_filter is not None and isinstance(
            no_analysis_func_filter, (list, tuple)
        ):
            no_analysis_func_filter = ",".join(map(str, no_analysis_func_filter))
        if unknown_type_conversion is not None and isinstance(
            unknown_type_conversion, (list, tuple)
        ):
            unknown_type_conversion = ",".join(map(str, unknown_type_conversion))
        self._add_pass(
            "one-shot-bufferize",
            allow_return_allocs=allow_return_allocs,
            allow_unknown_ops=allow_unknown_ops,
            analysis_fuzzer_seed=analysis_fuzzer_seed,
            analysis_heuristic=analysis_heuristic,
            bufferize_function_boundaries=bufferize_function_boundaries,
            copy_before_write=copy_before_write,
            create_deallocs=create_deallocs,
            dialect_filter=dialect_filter,
            function_boundary_type_conversion=function_boundary_type_conversion,
            must_infer_memory_space=must_infer_memory_space,
            no_analysis_func_filter=no_analysis_func_filter,
            print_conflicts=print_conflicts,
            test_analysis_only=test_analysis_only,
            unknown_type_conversion=unknown_type_conversion,
        )
        return self

    def outline_shape_computation(self):
        self._add_pass("outline-shape-computation")
        return self

    def pdl_bytecode_pass(self):
        self._add_pass("pdl-bytecode-pass")
        return self

    def post_sparsification_rewrite(
        self, enable_convert=None, enable_foreach=None, enable_runtime_library=None
    ):
        self._add_pass(
            "post-sparsification-rewrite",
            enable_convert=enable_convert,
            enable_foreach=enable_foreach,
            enable_runtime_library=enable_runtime_library,
        )
        return self

    def pre_sparsification_rewrite(self):
        self._add_pass("pre-sparsification-rewrite")
        return self

    def print_ir(self, label=None):
        if label is not None and isinstance(label, (list, tuple)):
            label = ",".join(map(str, label))
        self._add_pass("print-ir", label=label)
        return self

    def print_op_stats(self, json=None):
        self._add_pass("print-op-stats", json=json)
        return self

    def promote_buffers_to_stack(
        self, max_alloc_size_in_bytes=None, max_rank_of_allocated_memref=None
    ):
        if max_alloc_size_in_bytes is not None and isinstance(
            max_alloc_size_in_bytes, (list, tuple)
        ):
            max_alloc_size_in_bytes = ",".join(map(str, max_alloc_size_in_bytes))
        if max_rank_of_allocated_memref is not None and isinstance(
            max_rank_of_allocated_memref, (list, tuple)
        ):
            max_rank_of_allocated_memref = ",".join(
                map(str, max_rank_of_allocated_memref)
            )
        self._add_pass(
            "promote-buffers-to-stack",
            max_alloc_size_in_bytes=max_alloc_size_in_bytes,
            max_rank_of_allocated_memref=max_rank_of_allocated_memref,
        )
        return self

    def reconcile_unrealized_casts(self):
        self._add_pass("reconcile-unrealized-casts")
        return self

    def refbackend_generalize_tensor_pad(self):
        self._add_pass("refbackend-generalize-tensor-pad")
        return self

    def refbackend_munge_calling_conventions(self):
        self._add_pass("refbackend-munge-calling-conventions")
        return self

    def refbackend_munge_memref_copy(self):
        self._add_pass("refbackend-munge-memref-copy")
        return self

    def remove_shape_constraints(self):
        self._add_pass("remove-shape-constraints")
        return self

    def resolve_ranked_shaped_type_result_dims(self):
        self._add_pass("resolve-ranked-shaped-type-result-dims")
        return self

    def resolve_shaped_type_result_dims(self):
        self._add_pass("resolve-shaped-type-result-dims")
        return self

    def sccp(self):
        self._add_pass("sccp")
        return self

    def scf_bufferize(self):
        self._add_pass("scf-bufferize")
        return self

    def scf_for_loop_canonicalization(self):
        self._add_pass("scf-for-loop-canonicalization")
        return self

    def scf_for_loop_peeling(self, skip_partial=None):
        self._add_pass("scf-for-loop-peeling", skip_partial=skip_partial)
        return self

    def scf_for_loop_range_folding(self):
        self._add_pass("scf-for-loop-range-folding")
        return self

    def scf_for_loop_specialization(self):
        self._add_pass("scf-for-loop-specialization")
        return self

    def scf_for_to_while(self):
        self._add_pass("scf-for-to-while")
        return self

    def scf_parallel_loop_collapsing(
        self,
        collapsed_indices_0=None,
        collapsed_indices_1=None,
        collapsed_indices_2=None,
    ):
        if collapsed_indices_0 is not None and isinstance(
            collapsed_indices_0, (list, tuple)
        ):
            collapsed_indices_0 = ",".join(map(str, collapsed_indices_0))
        if collapsed_indices_1 is not None and isinstance(
            collapsed_indices_1, (list, tuple)
        ):
            collapsed_indices_1 = ",".join(map(str, collapsed_indices_1))
        if collapsed_indices_2 is not None and isinstance(
            collapsed_indices_2, (list, tuple)
        ):
            collapsed_indices_2 = ",".join(map(str, collapsed_indices_2))
        self._add_pass(
            "scf-parallel-loop-collapsing",
            collapsed_indices_0=collapsed_indices_0,
            collapsed_indices_1=collapsed_indices_1,
            collapsed_indices_2=collapsed_indices_2,
        )
        return self

    def scf_parallel_loop_fusion(self):
        self._add_pass("scf-parallel-loop-fusion")
        return self

    def scf_parallel_loop_specialization(self):
        self._add_pass("scf-parallel-loop-specialization")
        return self

    def scf_parallel_loop_tiling(
        self, no_min_max_bounds=None, parallel_loop_tile_sizes=None
    ):
        if parallel_loop_tile_sizes is not None and isinstance(
            parallel_loop_tile_sizes, (list, tuple)
        ):
            parallel_loop_tile_sizes = ",".join(map(str, parallel_loop_tile_sizes))
        self._add_pass(
            "scf-parallel-loop-tiling",
            no_min_max_bounds=no_min_max_bounds,
            parallel_loop_tile_sizes=parallel_loop_tile_sizes,
        )
        return self

    def serialize_spirv(self):
        self._add_pass("serialize-spirv")
        return self

    def set_spirv_abi_attrs(self, workgroup_size=None):
        if workgroup_size is not None and isinstance(workgroup_size, (list, tuple)):
            workgroup_size = ",".join(map(str, workgroup_size))
        self._add_pass("set-spirv-abi-attrs", workgroup_size=workgroup_size)
        return self

    def set_spirv_capabilities(self, client_api=None):
        if client_api is not None and isinstance(client_api, (list, tuple)):
            client_api = ",".join(map(str, client_api))
        self._add_pass("set-spirv-capabilities", client_api=client_api)
        return self

    def shape_bufferize(self):
        self._add_pass("shape-bufferize")
        return self

    def shape_to_shape_lowering(self):
        self._add_pass("shape-to-shape-lowering")
        return self

    def snapshot_op_locations(self, filename=None, tag=None):
        if filename is not None and isinstance(filename, (list, tuple)):
            filename = ",".join(map(str, filename))
        if tag is not None and isinstance(tag, (list, tuple)):
            tag = ",".join(map(str, tag))
        self._add_pass("snapshot-op-locations", filename=filename, tag=tag)
        return self

    def sparse_buffer_rewrite(self, enable_buffer_initialization=None):
        self._add_pass(
            "sparse-buffer-rewrite",
            enable_buffer_initialization=enable_buffer_initialization,
        )
        return self

    def sparse_storage_specifier_to_llvm(self):
        self._add_pass("sparse-storage-specifier-to-llvm")
        return self

    def sparse_tensor_codegen(self, enable_buffer_initialization=None):
        self._add_pass(
            "sparse-tensor-codegen",
            enable_buffer_initialization=enable_buffer_initialization,
        )
        return self

    def sparse_tensor_conversion(self, s2s_strategy=None):
        if s2s_strategy is not None and isinstance(s2s_strategy, (list, tuple)):
            s2s_strategy = ",".join(map(str, s2s_strategy))
        self._add_pass("sparse-tensor-conversion", s2s_strategy=s2s_strategy)
        return self

    def sparse_vectorization(
        self, enable_simd_index32=None, enable_vla_vectorization=None, vl=None
    ):
        if vl is not None and isinstance(vl, (list, tuple)):
            vl = ",".join(map(str, vl))
        self._add_pass(
            "sparse-vectorization",
            enable_simd_index32=enable_simd_index32,
            enable_vla_vectorization=enable_vla_vectorization,
            vl=vl,
        )
        return self

    def sparsification(
        self, enable_index_reduction=None, parallelization_strategy=None
    ):
        if parallelization_strategy is not None and isinstance(
            parallelization_strategy, (list, tuple)
        ):
            parallelization_strategy = ",".join(map(str, parallelization_strategy))
        self._add_pass(
            "sparsification",
            enable_index_reduction=enable_index_reduction,
            parallelization_strategy=parallelization_strategy,
        )
        return self

    def spirv_canonicalize_gl(self):
        self._add_pass("spirv-canonicalize-gl")
        return self

    def spirv_lower_abi_attrs(self):
        self._add_pass("spirv-lower-abi-attrs")
        return self

    def spirv_rewrite_inserts(self):
        self._add_pass("spirv-rewrite-inserts")
        return self

    def spirv_unify_aliased_resource(self):
        self._add_pass("spirv-unify-aliased-resource")
        return self

    def spirv_update_vce(self):
        self._add_pass("spirv-update-vce")
        return self

    def spirv_webgpu_prepare(self):
        self._add_pass("spirv-webgpu-prepare")
        return self

    def strip_debuginfo(self):
        self._add_pass("strip-debuginfo")
        return self

    def symbol_dce(self):
        self._add_pass("symbol-dce")
        return self

    def symbol_privatize(self, exclude=None):
        if exclude is not None and isinstance(exclude, (list, tuple)):
            exclude = ",".join(map(str, exclude))
        self._add_pass("symbol-privatize", exclude=exclude)
        return self

    def tensor_bufferize(self):
        self._add_pass("tensor-bufferize")
        return self

    def tiling_interface(
        self, filter_name=None, interchange=None, strategy=None, tile_sizes=None
    ):
        if filter_name is not None and isinstance(filter_name, (list, tuple)):
            filter_name = ",".join(map(str, filter_name))
        if interchange is not None and isinstance(interchange, (list, tuple)):
            interchange = ",".join(map(str, interchange))
        if strategy is not None and isinstance(strategy, (list, tuple)):
            strategy = ",".join(map(str, strategy))
        if tile_sizes is not None and isinstance(tile_sizes, (list, tuple)):
            tile_sizes = ",".join(map(str, tile_sizes))
        self._add_pass(
            "tiling-interface",
            filter_name=filter_name,
            interchange=interchange,
            strategy=strategy,
            tile_sizes=tile_sizes,
        )
        return self

    def topological_sort(self):
        self._add_pass("topological-sort")
        return self

    def tosa_infer_shapes(self):
        self._add_pass("tosa-infer-shapes")
        return self

    def tosa_layerwise_constant_fold(self):
        self._add_pass("tosa-layerwise-constant-fold")
        return self

    def tosa_make_broadcastable(self):
        self._add_pass("tosa-make-broadcastable")
        return self

    def tosa_optional_decompositions(self):
        self._add_pass("tosa-optional-decompositions")
        return self

    def tosa_to_arith(self, include_apply_rescale=None, use_32_bit=None):
        self._add_pass(
            "tosa-to-arith",
            include_apply_rescale=include_apply_rescale,
            use_32_bit=use_32_bit,
        )
        return self

    def tosa_to_linalg(self):
        self._add_pass("tosa-to-linalg")
        return self

    def tosa_to_linalg_named(self):
        self._add_pass("tosa-to-linalg-named")
        return self

    def tosa_to_scf(self):
        self._add_pass("tosa-to-scf")
        return self

    def tosa_to_tensor(self):
        self._add_pass("tosa-to-tensor")
        return self

    def tosa_validate(self, profile=None):
        if profile is not None and isinstance(profile, (list, tuple)):
            profile = ",".join(map(str, profile))
        self._add_pass("tosa-validate", profile=profile)
        return self

    def transform_dialect_check_uses(self):
        self._add_pass("transform-dialect-check-uses")
        return self

    def transform_dialect_erase_schedule(self):
        self._add_pass("transform-dialect-erase-schedule")
        return self

    def transform_dialect_interpreter(
        self,
        bind_first_extra_to_ops=None,
        bind_first_extra_to_params=None,
        bind_first_extra_to_results_of_ops=None,
        bind_second_extra_to_ops=None,
        bind_second_extra_to_params=None,
        bind_second_extra_to_results_of_ops=None,
        debug_payload_root_tag=None,
        debug_transform_root_tag=None,
        enable_expensive_checks=None,
        transform_file_name=None,
    ):
        if bind_first_extra_to_ops is not None and isinstance(
            bind_first_extra_to_ops, (list, tuple)
        ):
            bind_first_extra_to_ops = ",".join(map(str, bind_first_extra_to_ops))
        if bind_first_extra_to_params is not None and isinstance(
            bind_first_extra_to_params, (list, tuple)
        ):
            bind_first_extra_to_params = ",".join(map(str, bind_first_extra_to_params))
        if bind_first_extra_to_results_of_ops is not None and isinstance(
            bind_first_extra_to_results_of_ops, (list, tuple)
        ):
            bind_first_extra_to_results_of_ops = ",".join(
                map(str, bind_first_extra_to_results_of_ops)
            )
        if bind_second_extra_to_ops is not None and isinstance(
            bind_second_extra_to_ops, (list, tuple)
        ):
            bind_second_extra_to_ops = ",".join(map(str, bind_second_extra_to_ops))
        if bind_second_extra_to_params is not None and isinstance(
            bind_second_extra_to_params, (list, tuple)
        ):
            bind_second_extra_to_params = ",".join(
                map(str, bind_second_extra_to_params)
            )
        if bind_second_extra_to_results_of_ops is not None and isinstance(
            bind_second_extra_to_results_of_ops, (list, tuple)
        ):
            bind_second_extra_to_results_of_ops = ",".join(
                map(str, bind_second_extra_to_results_of_ops)
            )
        if debug_payload_root_tag is not None and isinstance(
            debug_payload_root_tag, (list, tuple)
        ):
            debug_payload_root_tag = ",".join(map(str, debug_payload_root_tag))
        if debug_transform_root_tag is not None and isinstance(
            debug_transform_root_tag, (list, tuple)
        ):
            debug_transform_root_tag = ",".join(map(str, debug_transform_root_tag))
        if transform_file_name is not None and isinstance(
            transform_file_name, (list, tuple)
        ):
            transform_file_name = ",".join(map(str, transform_file_name))
        self._add_pass(
            "transform-dialect-interpreter",
            bind_first_extra_to_ops=bind_first_extra_to_ops,
            bind_first_extra_to_params=bind_first_extra_to_params,
            bind_first_extra_to_results_of_ops=bind_first_extra_to_results_of_ops,
            bind_second_extra_to_ops=bind_second_extra_to_ops,
            bind_second_extra_to_params=bind_second_extra_to_params,
            bind_second_extra_to_results_of_ops=bind_second_extra_to_results_of_ops,
            debug_payload_root_tag=debug_payload_root_tag,
            debug_transform_root_tag=debug_transform_root_tag,
            enable_expensive_checks=enable_expensive_checks,
            transform_file_name=transform_file_name,
        )
        return self

    def vector_bufferize(self):
        self._add_pass("vector-bufferize")
        return self

    def view_op_graph(
        self,
        max_label_len=None,
        print_attrs=None,
        print_control_flow_edges=None,
        print_data_flow_edges=None,
        print_result_types=None,
    ):
        if max_label_len is not None and isinstance(max_label_len, (list, tuple)):
            max_label_len = ",".join(map(str, max_label_len))
        self._add_pass(
            "view-op-graph",
            max_label_len=max_label_len,
            print_attrs=print_attrs,
            print_control_flow_edges=print_control_flow_edges,
            print_data_flow_edges=print_data_flow_edges,
            print_result_types=print_result_types,
        )
        return self
