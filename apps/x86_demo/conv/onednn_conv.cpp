// largely adapted from
// https://oneapi-src.github.io/oneDNN/v2/convolution_example_cpp.html licensed
// by Intel Corporation, 2020 under Apache 2.0

#include "onednn_conv.hpp"

#include <cassert>
#include <iostream>

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) {
    throw std::runtime_error("handle is nullptr.");
  }

  assert(eng.get_kind() == dnnl::engine::kind::cpu);

  auto src = static_cast<uint8_t *>(mem.get_data_handle());
  if (!src) {
    throw std::runtime_error("get_data_handle returned nullptr.");
  }

  for (size_t i = 0; i < size; ++i) {
    ((uint8_t *)handle)[i] = src[i];
  }
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
  dnnl::engine eng = mem.get_engine();
  size_t size = mem.get_desc().get_size();

  if (!handle) {
    throw std::runtime_error("handle is nullptr.");
  }

  assert(eng.get_kind() == dnnl::engine::kind::cpu);

  auto dst = static_cast<uint8_t *>(mem.get_data_handle());
  if (!dst) {
    throw std::runtime_error("get_data_handle returned nullptr.");
  }

  for (size_t i = 0; i < size; ++i) {
    dst[i] = ((uint8_t *)handle)[i];
  }
}

void print_vec(const std::vector<long> &vec) {
  for (auto i : vec) {
    std::cout << i << ", ";
  }
  std::cout << "\n";
}

OneDNN_Conv::OneDNN_Conv(conv_instance &ci) : ci(ci) {
  // OneDNN expects dimension arguments in the following order, REGARDLESS
  // of the actual memory layout.

  memory::dims src_dims = {ci.N, ci.IC, ci.IH, ci.IW};
  memory::dims weights_dims = {ci.OC, ci.IC, ci.KH, ci.KW};
  memory::dims dst_dims = {ci.N, ci.OC, ci.OH, ci.OW};
  memory::dims bias_dims = {ci.OC};

  memory::dims strides_dims = {ci.SH, ci.SW};
  memory::dims padding_dims_l = {ci.PH_L, ci.PW_L};
  memory::dims padding_dims_r = {ci.PH_R, ci.PW_R};

  // Create memory objects for tensor data (src, weights, dst). In this
  // example, NHWC layout is assumed for src and dst, and IHWO for
  // weights.
  auto user_src_mem = memory({src_dims, dt::f32, tag::nhwc}, engine);
  auto user_weights_mem = memory({weights_dims, dt::f32, tag::ihwo}, engine);
  user_dst_mem = memory({dst_dims, dt::f32, tag::nhwc}, engine);

  // Create memory descriptors with format_tag::any for the primitive.
  // This enables the convolution primitive to choose memory layouts for
  // an optimized primitive implementation, and these layouts may differ
  // from the ones provided by the user.
  auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
  auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
  auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

  // Create memory descriptor and memory object for input bias.
  auto user_bias_md = memory::desc(bias_dims, dt::f32, tag::a);
  auto user_bias_mem = memory(user_bias_md, engine);

  // Write data to memory object's handle.
  write_to_dnnl_memory(ci.src_data.data(), user_src_mem);
  write_to_dnnl_memory(ci.weights_data.data(), user_weights_mem);
  write_to_dnnl_memory(ci.bias_data.data(), user_bias_mem);

  // Create operation descriptor.
  auto conv_desc = convolution_forward::desc(
      prop_kind::forward_training, algorithm::convolution_direct, conv_src_md,
      conv_weights_md, user_bias_md, conv_dst_md, strides_dims, padding_dims_l,
      padding_dims_r);

  // Create primitive post-ops (ReLU).
  const float scale = 1.f;
  const float alpha = 0.f;
  const float beta = 0.f;
  post_ops conv_ops;
  conv_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
  primitive_attr conv_attr;
  conv_attr.set_post_ops(conv_ops);

  // Create primitive descriptor.
  conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

  // For now, assume that the src, weights, and dst memory layouts
  // generated by the primitive and the ones provided by the user are
  // identical.
  auto conv_src_mem = user_src_mem;
  auto conv_weights_mem = user_weights_mem;
  conv_dst_mem = user_dst_mem;

  // Reorder the data in case the src and weights memory layouts generated
  // by the primitive and the ones provided by the user are different. In
  // this case, we create additional memory objects with internal buffers
  // that will contain the reordered data. The data in dst will be
  // reordered after the convolution computation has finalized.
  if (conv_pd.src_desc() != user_src_mem.get_desc()) {
    conv_src_mem = memory(conv_pd.src_desc(), engine);
    reorder(user_src_mem, conv_src_mem)
        .execute(engine_stream, user_src_mem, conv_src_mem);
  }

  if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
    conv_weights_mem = memory(conv_pd.weights_desc(), engine);
    reorder(user_weights_mem, conv_weights_mem)
        .execute(engine_stream, user_weights_mem, conv_weights_mem);
  }

  if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    conv_dst_mem = memory(conv_pd.dst_desc(), engine);
  }

  // Create the primitive.
  conv_prim = convolution_forward(conv_pd);

  // Primitive arguments.
  conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
  conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
  conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
  conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
}

void OneDNN_Conv::run() {
  // Primitive execution: convolution with ReLU.
  conv_prim.execute(engine_stream, conv_args);

  // Wait for the computation to finalize.
  engine_stream.wait();

  // Reorder the data in case the dst memory descriptor generated by the
  // primitive and the one provided by the user are different.
  if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
    reorder(conv_dst_mem, user_dst_mem)
        .execute(engine_stream, conv_dst_mem, user_dst_mem);
  } else {
    user_dst_mem = conv_dst_mem;
  }

  // Wait for the computation to finalize.
  engine_stream.wait();

  // Read data from memory object's handle.
  read_from_dnnl_memory(ci.dst_data.data(), user_dst_mem);
}
