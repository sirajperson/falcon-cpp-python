import sys
import os
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib
from typing import List, Union


# Load the library
def _load_shared_library(lib_base_name: str):
    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    # Searching for the library in the current directory under the name "libFalcon" (default name
    # for falconcpp) and "falcon" (default name for this repo)
    _lib_paths: List[pathlib.Path] = []
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
        ]
    elif sys.platform == "darwin":
        _lib_paths += [
            _base_path / f"lib{lib_base_name}.so",
            _base_path / f"lib{lib_base_name}.dylib",
        ]
    elif sys.platform == "win32":
        _lib_paths += [
            _base_path / f"{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    if "FALCON_CPP_LIB" in os.environ:
        lib_base_name = os.environ["FALCON_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "falcon"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

# falcon.h bindings

GGML_USE_CUBLAS = hasattr(_lib, "ggml_init_cublas")
GGML_CUDA_MAX_DEVICES = ctypes.c_int(16)
FALCON_MAX_DEVICES = GGML_CUDA_MAX_DEVICES if GGML_USE_CUBLAS else ctypes.c_int(1)

# #define FALCON_FILE_MAGIC_GGJT        0x67676a74u // 'ggjt'
FALCON_FILE_MAGIC_GGJT = ctypes.c_uint(0x67676A74)
# #define FALCON_FILE_MAGIC_GGLA        0x67676c61u // 'ggla'
FALCON_FILE_MAGIC_GGLA = ctypes.c_uint(0x67676C61)
# #define FALCON_FILE_MAGIC_GGMF        0x67676d66u // 'ggmf'
FALCON_FILE_MAGIC_GGMF = ctypes.c_uint(0x67676D66)
# #define FLACON_FILE_MAGIC_GGML        0x67676d6cu // 'ggml'
FALCON_FILE_MAGIC_GGML = ctypes.c_uint(0x67676D6C)
# #define FALCON_FILE_MAGIC_GGSN        0x6767736eu // 'ggsn'
FALCON_FILE_MAGIC_GGSN = ctypes.c_uint(0x6767736E)

# #define FALCON_FILE_VERSION           3
FALCON_FILE_VERSION = c_int(3)
FALCON_FILE_MAGIC = FALCON_FILE_MAGIC_GGJT
FALCON_FILE_MAGIC_UNVERSIONED = FALCON_FILE_MAGIC_GGML
FALCON_SESSION_MAGIC = FALCON_FILE_MAGIC_GGSN
FALCON_SESSION_VERSION = c_int(1)

# struct falcon_model;
falcon_model_p = c_void_p

# struct falcon_context;
falcon_context_p = c_void_p


# typedef int falcon_token;
falcon_token = c_int
falcon_token_p = POINTER(falcon_token)


# typedef struct falcon_token_data {
#     falcon_token id; // token id
#     float logit;    // log-odds of the token
#     float p;        // probability of the token
# } falcon_token_data;
class falcon_token_data(Structure):
    _fields_ = [
        ("id", falcon_token),
        ("logit", c_float),
        ("p", c_float),
    ]


falcon_token_data_p = POINTER(falcon_token_data)

# typedef struct falcon_token_data_array {
#     falcon_token_data * data;
#     size_t size;
#     bool sorted;
# } falcon_token_data_array;
class falcon_token_data_array(Structure):
    _fields_ = [
        ("data", falcon_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


falcon_token_data_array_p = POINTER(falcon_token_data_array)

# typedef void (*falcon_progress_callback)(float progress, void *ctx);
falcon_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)


# struct falcon_context_params {
#     int seed;                              // RNG seed, -1 for random
#     int n_ctx;                             // text context
#     int n_batch;                           // prompt processing batch size
#     int n_gpu_layers;                      // number of layers to store in VRAM
#     int main_gpu;                          // the GPU that is used for scratch and small tensors
#     float tensor_split[FALCON_MAX_DEVICES]; // how to split layers across multiple GPUs
#     // called with a progress value between 0 and 1, pass NULL to disable
#     falcon_progress_callback progress_callback;
#     // context pointer passed to the progress callback
#     void * progress_callback_user_data;


#     // Keep the booleans together to avoid misalignment during copy-by-value.
#     bool low_vram;   // if true, reduce VRAM usage at the cost of performance
#     bool f16_kv;     // use fp16 for KV cache
#     bool logits_all; // the falcon_eval() call computes all logits, not just the last one
#     bool vocab_only; // only load the vocabulary, no weights
#     bool use_mmap;   // use mmap if possible
#     bool use_mlock;  // force system to keep model in RAM
#     bool embedding;  // embedding mode only
# };
class ggllm_context_params(Structure):
    _fields_ = [
        ("seed", c_int),
        ("n_ctx", c_int),
        ("n_batch", c_int),
        ("n_gpu_layers", c_int),
        ("main_gpu", c_int),
        ("tensor_split", c_float * FALCON_MAX_DEVICES.value),
        ("progress_callback", falcon_progress_callback),
        ("progress_callback_user_data", c_void_p),
        ("low_vram", c_bool),
        ("f16_kv", c_bool),
        ("logits_all", c_bool),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("embedding", c_bool),
    ]


falcon_context_params_p = POINTER(ggllm_context_params)

# enum falcon_ftype {
#     FALCON_FTYPE_ALL_F32              = 0,
#     FALCON_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     // FALCON_FTYPE_MOSTLY_Q4_2       = 5, // support has been removed
#     // FALCON_FTYPE_MOSTLY_Q4_3       = 6, // support has been removed
#     FALCON_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q2_K          = 10,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q3_K_S        = 11,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q3_K_M        = 12,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q3_K_L        = 13,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q4_K_S        = 14,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q4_K_M        = 15,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q5_K_S        = 16,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q5_K_M        = 17,// except 1d tensors
#     FALCON_FTYPE_MOSTLY_Q6_K          = 18,// except 1d tensors
# };
FALCON_FTYPE_ALL_F32 = c_int(0)
FALCON_FTYPE_MOSTLY_F16 = c_int(1)
FALCON_FTYPE_MOSTLY_Q4_0 = c_int(2)
FALCON_FTYPE_MOSTLY_Q4_1 = c_int(3)
FALCON_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(4)
FALCON_FTYPE_MOSTLY_Q8_0 = c_int(7)
FALCON_FTYPE_MOSTLY_Q5_0 = c_int(8)
FALCON_FTYPE_MOSTLY_Q5_1 = c_int(9)
FALCON_FTYPE_MOSTLY_Q2_K = c_int(10)
FALCON_FTYPE_MOSTLY_Q3_K_S = c_int(11)
FALCON_FTYPE_MOSTLY_Q3_K_M = c_int(12)
FALCON_FTYPE_MOSTLY_Q3_K_L = c_int(13)
FALCON_FTYPE_MOSTLY_Q4_K_S = c_int(14)
FALCON_FTYPE_MOSTLY_Q4_K_M = c_int(15)
FALCON_FTYPE_MOSTLY_Q5_K_S = c_int(16)
FALCON_FTYPE_MOSTLY_Q5_K_M = c_int(17)
FALCON_FTYPE_MOSTLY_Q6_K = c_int(18)


# // model quantization parameters
# typedef struct falcon_model_quantize_params {
#     int nthread;                 // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
#     enum falcon_ftype   ftype;    // quantize to this falcon_ftype
#     bool allow_requantize;       // allow quantizing non-f32/f16 tensors
#     bool quantize_output_tensor; // quantize output.weight
# } falcon_model_quantize_params;
class falcon_model_quantize_params(Structure):
    _fields_ = [
        ("nthread", c_int),
        ("ftype", c_int),
        ("allow_requantize", c_bool),
        ("quantize_output_tensor", c_bool),
    ]


# FALCON_API struct falcon_context_params falcon_context_default_params();
def falcon_context_default_params() -> ggllm_context_params:
    return _lib.ggllm_context_default_params()


_lib.ggllm_context_default_params.argtypes = []
_lib.ggllm_context_default_params.restype = ggllm_context_params


# FALCON_API struct falcon_model_quantize_params falcon_model_quantize_default_params();
def falcon_model_quantize_default_params() -> falcon_model_quantize_params:
    return _lib.ggllm_model_quantize_default_params()


_lib.ggllm_model_quantize_default_params.argtypes = []
_lib.ggllm_model_quantize_default_params.restype = falcon_model_quantize_params


# FALCON_API bool falcon_mmap_supported();
def falcon_mmap_supported() -> bool:
    return _lib.ggllm_mmap_supported()


_lib.ggllm_mmap_supported.argtypes = []
_lib.ggllm_mmap_supported.restype = c_bool


# FALCON_API bool falcon_mlock_supported();
def falcon_mlock_supported() -> bool:
    return _lib.ggllm_mlock_supported()


_lib.ggllm_mlock_supported.argtypes = []
_lib.ggllm_mlock_supported.restype = c_bool


# // TODO: not great API - very likely to change
# // Initialize the falcon + ggml backend
# // If numa is true, use NUMA optimizations
# // Call once at the start of the program
# FLACON_API void falcon_init_backend(bool numa);
def falcon_init_backend(numa: c_bool):
    return _lib.ggllm_init_backend(numa)


_lib.ggllm_init_backend.argtypes = [c_bool]
_lib.ggllm_init_backend.restype = None


# FALCON_API struct falcon_model * falcon_load_model_from_file(
#                             const char * path_model,
#         struct falcon_context_params   params);
def falcon_load_model_from_file(
    path_model: bytes, params: ggllm_context_params
) -> falcon_model_p:
    return _lib.ggllm_load_model_from_file(path_model, params)


_lib.ggllm_load_model_from_file.argtypes = [c_char_p, ggllm_context_params]
_lib.ggllm_load_model_from_file.restype = falcon_model_p


# FALCON_API void falcon_free_model(struct falcon_model * model);
def falcon_free_model(model: falcon_model_p):
    return _lib.ggllm_free_model(model)


_lib.ggllm_free_model.argtypes = [falcon_model_p]
_lib.ggllm_free_model.restype = None


# FALCON_API struct falcon_context * falcon_new_context_with_model(
#                     struct falcon_model * model,
#         struct falcon_context_params   params);
def falcon_new_context_with_model(
    model: falcon_model_p, params: ggllm_context_params
) -> falcon_context_p:
    return _lib.ggllm_new_context_with_model(model, params)


_lib.ggllm_new_context_with_model.argtypes = [falcon_model_p, ggllm_context_params]
_lib.ggllm_new_context_with_model.restype = falcon_context_p


# FALCON_API int64_t ggllm_time_us();
def ggllm_time_us() -> int:
    return _lib.ggllm_time_us()


_lib.ggllm_time_us.argtypes = []
_lib.ggllm_time_us.restype = ctypes.c_int64


# // Various functions for loading a ggml falcon model.
# // Allocate (almost) all memory needed for the model.
# // Return NULL on failure
# FALCON_API struct falcon_context * falcon_init_from_file(
#                             const char * path_model,
#         struct falcon_context_params   params);
def ggllm_init_from_file(
    path_model: bytes, params: ggllm_context_params
) -> falcon_context_p:
    return _lib.ggllm_init_from_file(path_model, params)


_lib.ggllm_init_from_file.argtypes = [c_char_p, ggllm_context_params]
_lib.ggllm_init_from_file.restype = falcon_context_p


# Frees all allocated memory
# FALCON_API void falcon_free(struct falcon_context * ctx);
def falcon_free(ctx: falcon_context_p):
    return _lib.ggllm_free(ctx)


_lib.ggllm_free.argtypes = [falcon_context_p]
_lib.ggllm_free.restype = None


# // Returns 0 on success
# FALCON_API int ggllm_model_quantize(
#         const char * fname_inp,
#         const char * fname_out,
#         const falcon_model_quantize_params * params);
def ggllm_model_quantize(
    fname_inp: bytes,
    fname_out: bytes,
    params,  # type: POINTER(falcon_model_quantize_params) # type: ignore
) -> int:
    return _lib.ggllm_model_quantize(fname_inp, fname_out, params)


_lib.ggllm_model_quantize.argtypes = [
    c_char_p,
    c_char_p,
    POINTER(falcon_model_quantize_params),
]
_lib.ggllm_model_quantize.restype = c_int


# Apply a LoRA adapter to a loaded model
# path_base_model is the path to a higher quality model to use as a base for
# the layers modified by the adapter. Can be NULL to use the current loaded model.
# The model needs to be reloaded before applying a new adapter, otherwise the adapter
# will be applied on top of the previous one
# Returns 0 on success
# FALCON_API int falcon_apply_lora_from_file(
#         struct falcon_context * ctx,
#                   const char * path_lora,
#                   const char * path_base_model,
#                          int   n_threads);
def ggllm_apply_lora_from_file(
    ctx: falcon_context_p,
    path_lora: c_char_p,
    path_base_model: c_char_p,
    n_threads: c_int,
) -> int:
    return _lib.ggllm_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


_lib.ggllm_apply_lora_from_file.argtypes = [falcon_context_p, c_char_p, c_char_p, c_int]
_lib.ggllm_apply_lora_from_file.restype = c_int


# FALCON_API int ggllm_model_apply_lora_from_file(
#         const struct ggllm_model * model,
#                     const char * path_lora,
#                     const char * path_base_model,
#                             int   n_threads);
def falcon_model_apply_lora_from_file(
    model: falcon_model_p,
    path_lora: Union[c_char_p, bytes],
    path_base_model: Union[c_char_p, bytes],
    n_threads: c_int,
) -> int:
    return _lib.ggllm_model_apply_lora_from_file(
        model, path_lora, path_base_model, n_threads
    )


_lib.ggllm_model_apply_lora_from_file.argtypes = [
    falcon_model_p,
    c_char_p,
    c_char_p,
    c_int,
]
_lib.ggllm_model_apply_lora_from_file.restype = c_int


# Returns the number of tokens in the KV cache
# FALCON_API int falcon_get_kv_cache_token_count(const struct falcon_context * ctx);
def ggllm_get_kv_cache_token_count(ctx: falcon_context_p) -> int:
    return _lib.ggllm_get_kv_cache_token_count(ctx)


_lib.ggllm_get_kv_cache_token_count.argtypes = [falcon_context_p]
_lib.ggllm_get_kv_cache_token_count.restype = c_int


# Sets the current rng seed.
# FALCON_API void falcon_set_rng_seed(struct falcon_context * ctx, int seed);
def falcon_set_rng_seed(ctx: falcon_context_p, seed: c_int):
    return _lib.ggllm_set_rng_seed(ctx, seed)


_lib.ggllm_set_rng_seed.argtypes = [falcon_context_p, c_int]
_lib.ggllm_set_rng_seed.restype = None


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
# FALCON_API size_t falcon_get_state_size(const struct falcon_context * ctx);
def falcon_get_state_size(ctx: falcon_context_p) -> int:
    return _lib.ggllm_get_state_size(ctx)


_lib.ggllm_get_state_size.argtypes = [falcon_context_p]
_lib.ggllm_get_state_size.restype = c_size_t


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
# FALCON_API size_t falcon_copy_state_data(struct falcon_context * ctx, uint8_t * dst);
def falcon_copy_state_data(
    ctx: falcon_context_p, dst  # type: Array[c_uint8]
) -> int:
    return _lib.ggllm_copy_state_data(ctx, dst)


_lib.ggllm_copy_state_data.argtypes = [falcon_context_p, c_uint8_p]
_lib.ggllm_copy_state_data.restype = c_size_t


# Set the state reading from the specified address
# Returns the number of bytes read
# FALCON_API size_t falcon_set_state_data(struct falcon_context * ctx, uint8_t * src);
def falcon_set_state_data(
    ctx: falcon_context_p, src  # type: Array[c_uint8]
) -> int:
    return _lib.ggllm_set_state_data(ctx, src)


_lib.ggllm_set_state_data.argtypes = [falcon_context_p, c_uint8_p]
_lib.ggllm_set_state_data.restype = c_size_t


# Save/load session file
# GGLLM_API bool falcon_load_session_file(struct falcon_context * ctx, const char * path_session, falcon_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
def ggllm_load_session_file(
    ctx: falcon_context_p,
    path_session: bytes,
    tokens_out,  # type: Array[falcon_token]
    n_token_capacity: c_size_t,
    n_token_count_out,  # type: _Pointer[c_size_t]
) -> int:
    return _lib.ggllm_load_session_file(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out
    )


_lib.ggllm_load_session_file.argtypes = [
    falcon_context_p,
    c_char_p,
    falcon_token_p,
    c_size_t,
    c_size_t_p,
]
_lib.ggllm_load_session_file.restype = c_size_t


# FALCON_API bool falcon_save_session_file(struct falcon_context * ctx, const char * path_session, const falcon_token * tokens, size_t n_token_count);
def ggllm_save_session_file(
    ctx: falcon_context_p,
    path_session: bytes,
    tokens,  # type: Array[falcon_token]
    n_token_count: c_size_t,
) -> int:
    return _lib.ggllm_save_session_file(ctx, path_session, tokens, n_token_count)


_lib.ggllm_save_session_file.argtypes = [
    falcon_context_p,
    c_char_p,
    falcon_token_p,
    c_size_t,
]
_lib.ggllm_save_session_file.restype = c_size_t


# Run the falcon inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
# GGLLM_API int falcon_eval(
#         struct falcon_context * ctx,
#            const falcon_token * tokens,
#                          int   n_tokens,
#                          int   n_past,
#                          int   n_threads);
def falcon_eval(
    ctx: falcon_context_p,
    tokens,  # type: Array[falcon_token]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> int:
    return _lib.ggllm_eval(ctx, tokens, n_tokens, n_past, n_threads)


_lib.ggllm_eval.argtypes = [falcon_context_p, falcon_token_p, c_int, c_int, c_int]
_lib.ggllm_eval.restype = c_int


# // Same as falcon_eval, but use float matrix input directly.
# FALCON_API int falcon_eval_embd(
#         struct falcon_context * ctx,
#                     const float * embd,
#                             int   n_tokens,
#                             int   n_past,
#                             int   n_threads);
def ggllm_eval_embd(
    ctx: falcon_context_p,
    embd,  # type: Array[c_float]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> int:
    return _lib.ggllm_eval_embd(ctx, embd, n_tokens, n_past, n_threads)


_lib.ggllm_eval_embd.argtypes = [falcon_context_p, c_float_p, c_int, c_int, c_int]
_lib.ggllm_eval_embd.restype = c_int


# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
# FALCON_API int ggllm_tokenize(
#         struct falcon_context * ctx,
#                   const char * text,
#                  falcon_token * tokens,
#                          int   n_max_tokens,
#                         bool   add_bos);
def falcon_tokenize(
    ctx: falcon_context_p,
    text: bytes,
    tokens,  # type: Array[falcon_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> int:
    return _lib.ggllm_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


_lib.ggllm_tokenize.argtypes = [falcon_context_p, c_char_p, falcon_token_p, c_int, c_bool]
_lib.ggllm_tokenize.restype = c_int


# GGLLM_API int ggllm_n_vocab(const struct falcon_context * ctx);
def falcon_n_vocab(ctx: falcon_context_p) -> int:
    return _lib.ggllm_n_vocab(ctx)


_lib.ggllm_n_vocab.argtypes = [falcon_context_p]
_lib.ggllm_n_vocab.restype = c_int


# FALCON_API int falcon_n_ctx  (const struct falcon_context * ctx);
def falcon_n_ctx(ctx: falcon_context_p) -> int:
    return _lib.ggllm_n_ctx(ctx)


_lib.ggllm_n_ctx.argtypes = [falcon_context_p]
_lib.ggllm_n_ctx.restype = c_int


# FALCON_API int falcon_n_embd (const struct falcon_context * ctx);
def falcon_n_embd(ctx: falcon_context_p) -> int:
    return _lib.ggllm_n_embd(ctx)


_lib.ggllm_n_embd.argtypes = [falcon_context_p]
_lib.ggllm_n_embd.restype = c_int


# // Get the vocabulary as output parameters.
# // Returns number of results.
# FALCON_API int falcon_get_vocab(
#         const struct falcon_context * ctx,
#                         const char * * strings,
#                                 float * scores,
#                                 int   capacity);
def falcon_get_vocab(
    ctx: falcon_context_p,
    strings,  # type: Array[c_char_p] # type: ignore
    scores,  # type: Array[c_float] # type: ignore
    capacity: c_int,
) -> int:
    return _lib.ggllm_get_vocab(ctx, strings, scores, capacity)


_lib.ggllm_get_vocab.argtypes = [falcon_context_p, c_char_p, c_float, c_int]
_lib.ggllm_get_vocab.restype = c_int


# Token logits obtained from the last call to falcon_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
# FALCON_API float * falcon_get_logits(struct falcon_context * ctx);
def falcon_get_logits(
    ctx: falcon_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.ggllm_get_logits(ctx)


_lib.ggllm_get_logits.argtypes = [falcon_context_p]
_lib.ggllm_get_logits.restype = c_float_p


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
# FALCON_API float * falcon_get_embeddings(struct falcon_context * ctx);
def falcon_get_embeddings(
    ctx: falcon_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.ggllm_get_embeddings(ctx)


_lib.ggllm_get_embeddings.argtypes = [falcon_context_p]
_lib.ggllm_get_embeddings.restype = c_float_p


# Token Id -> String. Uses the vocabulary in the provided context
# FLACON_API const char * falcon_token_to_str(const struct falcon_context * ctx, falcon_token token);
def falcon_token_to_str(ctx: falcon_context_p, token: falcon_token) -> bytes:
    return _lib.ggllm_token_to_str(ctx, token)


_lib.ggllm_token_to_str.argtypes = [falcon_context_p, falcon_token]
_lib.ggllm_token_to_str.restype = c_char_p

# Special tokens


# FALCON_API falcon_token falcon_token_bos(); // beginning-of-sentence
def falcon_token_bos() -> int:
    return _lib.ggllm_token_bos()


_lib.ggllm_token_bos.argtypes = []
_lib.ggllm_token_bos.restype = falcon_token


# FALCON_API falcon_token falcon_token_eos(); // end-of-sentence
def falcon_token_eos() -> int:
    return _lib.ggllm_token_eos()


_lib.ggllm_token_eos.argtypes = []
_lib.ggllm_token_eos.restype = falcon_token


# FALCON_API falcon_token falcon_token_nl(); // next-line
def falcon_token_nl() -> int:
    return _lib.ggllm_token_nl()


_lib.ggllm_token_nl.argtypes = []
_lib.ggllm_token_nl.restype = falcon_token


# Sampling functions


# @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
# FALCON_API void falcon_sample_repetition_penalty(struct falcon_context * ctx, falcon_token_data_array * candidates, const falcon_token * last_tokens, size_t last_tokens_size, float penalty);
def falcon_sample_repetition_penalty(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    last_tokens_data,  # type: Array[falcon_token]
    last_tokens_size: c_int,
    penalty: c_float,
):
    return _lib.ggllm_sample_repetition_penalty(
        ctx, candidates, last_tokens_data, last_tokens_size, penalty
    )


_lib.ggllm_sample_repetition_penalty.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    falcon_token_p,
    c_int,
    c_float,
]
_lib.ggllm_sample_repetition_penalty.restype = None


# @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
# FALCON_API void falcon_sample_frequency_and_presence_penalties(struct falcon_context * ctx, falcon_token_data_array * candidates, const falcon_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);
def falcon_sample_frequency_and_presence_penalties(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    last_tokens_data,  # type: Array[falcon_token]
    last_tokens_size: c_int,
    alpha_frequency: c_float,
    alpha_presence: c_float,
):
    return _lib.ggllm_sample_frequency_and_presence_penalties(
        ctx,
        candidates,
        last_tokens_data,
        last_tokens_size,
        alpha_frequency,
        alpha_presence,
    )


_lib.ggllm_sample_frequency_and_presence_penalties.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    falcon_token_p,
    c_int,
    c_float,
    c_float,
]
_lib.ggllm_sample_frequency_and_presence_penalties.restype = None


# @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
# FALCON_API void falcon_sample_softmax(struct falcon_context * ctx, falcon_token_data_array * candidates);
def falcon_sample_softmax(
    ctx: falcon_context_p, candidates  # type: _Pointer[falcon_token_data]
):
    return _lib.ggllm_sample_softmax(ctx, candidates)


_lib.ggllm_sample_softmax.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
]
_lib.ggllm_sample_softmax.restype = None


# @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# FALCON_API void falcon_sample_top_k(struct falcon_context * ctx, falcon_token_data_array * candidates, int k, size_t min_keep);
def falcon_sample_top_k(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    k: c_int,
    min_keep: c_size_t,
):
    return _lib.ggllm_sample_top_k(ctx, candidates, k, min_keep)


_lib.ggllm_sample_top_k.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_int,
    c_size_t,
]
_lib.ggllm_sample_top_k.restype = None


# @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# FALCON_API void falcon_sample_top_p(struct falcon_context * ctx, falcon_token_data_array * candidates, float p, size_t min_keep);
def falcon_sample_top_p(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.ggllm_sample_top_p(ctx, candidates, p, min_keep)


_lib.ggllm_sample_top_p.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.ggllm_sample_top_p.restype = None


# @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
# FALCON_API void falcon_sample_tail_free(struct falcon_context * ctx, falcon_token_data_array * candidates, float z, size_t min_keep);
def falcon_sample_tail_free(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    z: c_float,
    min_keep: c_size_t,
):
    return _lib.ggllm_sample_tail_free(ctx, candidates, z, min_keep)


_lib.ggllm_sample_tail_free.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.ggllm_sample_tail_free.restype = None


# @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
# FALCON_API void falcon_sample_typical(struct falcon_context * ctx, falcon_token_data_array * candidates, float p, size_t min_keep);
def falcon_sample_typical(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.ggllm_sample_typical(ctx, candidates, p, min_keep)


_lib.ggllm_sample_typical.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.ggllm_sample_typical.restype = None


# FALCON_API void falcon_sample_temperature(struct falcon_context * ctx, falcon_token_data_array * candidates, float temp);
def falcon_sample_temperature(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    temp: c_float,
):
    return _lib.ggllm_sample_temperature(ctx, candidates, temp)


_lib.ggllm_sample_temperature.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
]
_lib.ggllm_sample_temperature.restype = None


# @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `falcon_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# FALCON_API falcon_token falcon_sample_token_mirostat(struct falcon_context * ctx, falcon_token_data_array * candidates, float tau, float eta, int m, float * mu);
def falcon_sample_token_mirostat(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    tau: c_float,
    eta: c_float,
    m: c_int,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.ggllm_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)


_lib.ggllm_sample_token_mirostat.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
    c_float,
    c_int,
    c_float_p,
]
_lib.ggllm_sample_token_mirostat.restype = falcon_token


# @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `falcon_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# FALCON_API falcon_token falcon_sample_token_mirostat_v2(struct falcon_context * ctx, falcon_token_data_array * candidates, float tau, float eta, float * mu);
def falcon_sample_token_mirostat_v2(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
    tau: c_float,
    eta: c_float,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.ggllm_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)


_lib.ggllm_sample_token_mirostat_v2.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
    c_float,
    c_float,
    c_float_p,
]
_lib.ggllm_sample_token_mirostat_v2.restype = falcon_token


# @details Selects the token with the highest probability.
# FALCON_API falcon_token falcon_sample_token_greedy(struct falcon_context * ctx, falcon_token_data_array * candidates);
def falcon_sample_token_greedy(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
) -> int:
    return _lib.ggllm_sample_token_greedy(ctx, candidates)


_lib.ggllm_sample_token_greedy.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
]
_lib.ggllm_sample_token_greedy.restype = falcon_token


# @details Randomly selects a token from the candidates based on their probabilities.
# FALCON_API falcon_token falcon_sample_token(struct falcon_context * ctx, falcon_token_data_array * candidates);
def falcon_sample_token(
    ctx: falcon_context_p,
    candidates,  # type: _Pointer[falcon_token_data_array]
) -> int:
    return _lib.ggllm_sample_token(ctx, candidates)


_lib.ggllm_sample_token.argtypes = [
    falcon_context_p,
    falcon_token_data_array_p,
]
_lib.ggllm_sample_token.restype = falcon_token


# Performance information


# FALCON_API void falcon_print_timings(struct falcon_context * ctx);
def falcon_print_timings(ctx: falcon_context_p):
    _lib.ggllm_print_timings(ctx)


_lib.ggllm_print_timings.argtypes = [falcon_context_p]
_lib.ggllm_print_timings.restype = None


# FALCON_API void falcon_reset_timings(struct falcon_context * ctx);
def falcon_reset_timings(ctx: falcon_context_p):
    _lib.ggllm_reset_timings(ctx)


_lib.ggllm_reset_timings.argtypes = [falcon_context_p]
_lib.ggllm_reset_timings.restype = None


# Print system information
# FALCON_API const char * falcon_print_system_info(void);
def falcon_print_system_info() -> bytes:
    return _lib.ggllm_print_system_info()


_lib.ggllm_print_system_info.argtypes = []
_lib.ggllm_print_system_info.restype = c_char_p

###################################################################################################


_falcon_initialized = False

if not _falcon_initialized:
    falcon_init_backend(c_bool(False))
    _falcon_initialized = True