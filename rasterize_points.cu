/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>
#include <cstdio>
#include <functional>
#include <tuple>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
  auto lambda = [&t](size_t N) {
    t.resize_({(long long)N});
    return reinterpret_cast<char*>(t.contiguous().data_ptr());
  };
  return lambda;
}

std::tuple<int,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
RasterizeGaussiansCUDA(const torch::Tensor& background,
                       const torch::Tensor& means3D,
                       const torch::Tensor& colors,
                       const torch::Tensor& opacity,
                       const torch::Tensor& scales,
                       const torch::Tensor& rotations,
                       const float scale_modifier,
                       const torch::Tensor& cov3D_precomp,
                       const torch::Tensor& viewmatrix,
                       const torch::Tensor& projmatrix,
                       const torch::Tensor& projmatrix_raw,
                       const float tan_fovx,
                       const float tan_fovy,
                       const int image_height,
                       const int image_width,
                       const torch::Tensor& sh,
                       const int degree,
                       const torch::Tensor& campos,
                       const bool prefiltered,
                       const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }

    rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P,
        degree,
        M,
        background.contiguous().data_ptr<float>(),
        W,
        H,
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_color.contiguous().data_ptr<float>(),
        out_depth.contiguous().data_ptr<float>(),
        out_opaticy.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        n_touched.contiguous().data_ptr<int>(),
        debug);
  }
  return std::make_tuple(rendered,
                         out_color,
                         radii,
                         geomBuffer,
                         binningBuffer,
                         imgBuffer,
                         out_depth,
                         out_opaticy,
                         n_touched);
}

std::tuple<int,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
RasterizeSemanticGaussiansCUDA(const torch::Tensor& background_color,
                               const torch::Tensor& background_semantics,
                               const torch::Tensor& means3D,
                               const torch::Tensor& colors,
                               const torch::Tensor& semantics,
                               const torch::Tensor& opacity,
                               const torch::Tensor& scales,
                               const torch::Tensor& rotations,
                               const float scale_modifier,
                               const torch::Tensor& cov3D_precomp,
                               const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix,
                               const torch::Tensor& projmatrix_raw,
                               const float tan_fovx,
                               const float tan_fovy,
                               const int image_height,
                               const int image_width,
                               const torch::Tensor& sh,
                               const torch::Tensor& semantic_sh,
                               const int degree,
                               const torch::Tensor& campos,
                               const bool prefiltered,
                               const bool debug) {
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);  // the number of 3D Gaussian ellipsoids
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_colors = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  // ?
  torch::Tensor out_semantics =
      torch::full({NUM_SEMANTIC_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor n_touched =
      torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_opaticy = torch::full({1, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor semantic_geometry_buffer =
      torch::empty({0}, options.device(device));
  torch::Tensor binning_buffer = torch::empty({0}, options.device(device));
  torch::Tensor img_buffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> semantic_geometry_func =
      resizeFunctional(semantic_geometry_buffer);
  std::function<char*(size_t)> binning_func = resizeFunctional(binning_buffer);
  std::function<char*(size_t)> img_func = resizeFunctional(img_buffer);

  int num_rendered = 0;
  if (P != 0) {
    int M = 0;
    if (sh.size(0) != 0) {
      M = sh.size(1);
    }
    int semantic_M = 0;
    if (semantic_sh.size(0) != 0) {
      semantic_M = semantic_sh.size(1);
    }

    num_rendered = CudaRasterizer::SemanticRasterizer::forward(
        semantic_geometry_func,
        binning_func,
        img_func,
        P,
        degree,
        M,
        semantic_M,
        background_color.contiguous().data_ptr<float>(),
        background_semantics.contiguous().data_ptr<float>(),
        W,
        H,
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        semantic_sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        semantics.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        scale_modifier,
        rotations.contiguous().data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        out_colors.contiguous().data_ptr<float>(),
        out_semantics.contiguous().data_ptr<float>(),
        out_depth.contiguous().data_ptr<float>(),
        out_opaticy.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        n_touched.contiguous().data_ptr<int>(),
        debug);
  }
  return std::make_tuple(num_rendered,
                         out_colors,
                         out_semantics,
                         radii,
                         semantic_geometry_buffer,
                         binning_buffer,
                         img_buffer,
                         out_depth,
                         out_opaticy,
                         n_touched);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
RasterizeGaussiansBackwardCUDA(const torch::Tensor& background,
                               const torch::Tensor& means3D,
                               const torch::Tensor& radii,
                               const torch::Tensor& colors,
                               const torch::Tensor& scales,
                               const torch::Tensor& rotations,
                               const float scale_modifier,
                               const torch::Tensor& cov3D_precomp,
                               const torch::Tensor& viewmatrix,
                               const torch::Tensor& projmatrix,
                               const torch::Tensor& projmatrix_raw,
                               const float tan_fovx,
                               const float tan_fovy,
                               const torch::Tensor& dL_dout_color,
                               const torch::Tensor& dL_dout_depths,
                               const torch::Tensor& sh,
                               const int degree,
                               const torch::Tensor& campos,
                               const torch::Tensor& geomBuffer,
                               const int R,
                               const torch::Tensor& binningBuffer,
                               const torch::Tensor& imageBuffer,
                               const bool debug) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dtau = torch::zeros({P, 6}, means3D.options());

  if (P != 0) {
    CudaRasterizer::Rasterizer::backward(
        P,
        degree,
        M,
        R,
        background.contiguous().data_ptr<float>(),
        W,
        H,
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        scales.data_ptr<float>(),
        scale_modifier,
        rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        projmatrix_raw.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dout_depths.contiguous().data_ptr<float>(),
        dL_dmeans2D.contiguous().data_ptr<float>(),
        dL_dconic.contiguous().data_ptr<float>(),
        dL_dopacity.contiguous().data_ptr<float>(),
        dL_dcolors.contiguous().data_ptr<float>(),
        dL_ddepths.contiguous().data_ptr<float>(),
        dL_dmeans3D.contiguous().data_ptr<float>(),
        dL_dcov3D.contiguous().data_ptr<float>(),
        dL_dsh.contiguous().data_ptr<float>(),
        dL_dscales.contiguous().data_ptr<float>(),
        dL_drotations.contiguous().data_ptr<float>(),
        dL_dtau.contiguous().data_ptr<float>(),
        debug);
  }

  return std::make_tuple(dL_dmeans2D,
                         dL_dcolors,
                         dL_dopacity,
                         dL_dmeans3D,
                         dL_dcov3D,
                         dL_dsh,
                         dL_dscales,
                         dL_drotations,
                         dL_dtau);
}

std::tuple<torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor>
RasterizeSemanticGaussiansBackwardCUDA(
    const torch::Tensor& background_color,
    const torch::Tensor& background_semantics,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& semantics,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const torch::Tensor& projmatrix_raw,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_semantics,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& sh,
    const torch::Tensor& semantic_sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& semantic_geometry_buffer,
    const int R,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& image_buffer,
    const bool debug) {
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if (sh.size(0) != 0) {
    M = sh.size(1);
  }
  int semantic_M = 0;
  if (semantic_sh.size(0) != 0) {
    semantic_M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  // ?
  torch::Tensor dL_dsemantics =
      torch::zeros({P, NUM_SEMANTIC_CHANNELS}, means3D.options());
  torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dsemantic_sh = torch::zeros({P, M, NUM_SEMANTIC_CHANNELS}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dtau = torch::zeros({P, 6}, means3D.options());

  if (P != 0) {
    CudaRasterizer::SemanticRasterizer::backward(
        P,
        degree,
        M,
        semantic_M,
        R,
        background_color.contiguous().data_ptr<float>(),
        background_semantics.contiguous().data_ptr<float>(),
        W,
        H,
        means3D.contiguous().data_ptr<float>(),
        sh.contiguous().data_ptr<float>(),
        semantic_sh.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        semantics.contiguous().data_ptr<float>(),
        scales.data_ptr<float>(),
        scale_modifier,
        rotations.data_ptr<float>(),
        cov3D_precomp.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        projmatrix_raw.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char*>(
            semantic_geometry_buffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binning_buffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(image_buffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dout_semantics.contiguous().data_ptr<float>(),
        dL_dout_depth.contiguous().data_ptr<float>(),
        dL_dmeans2D.contiguous().data_ptr<float>(),
        dL_dconic.contiguous().data_ptr<float>(),
        dL_dopacity.contiguous().data_ptr<float>(),
        dL_dcolors.contiguous().data_ptr<float>(),
        dL_dsemantics.contiguous().data_ptr<float>(),
        dL_ddepths.contiguous().data_ptr<float>(),
        dL_dmeans3D.contiguous().data_ptr<float>(),
        dL_dcov3D.contiguous().data_ptr<float>(),
        dL_dsh.contiguous().data_ptr<float>(),
        dL_dsemantic_sh.contiguous().data_ptr<float>(),
        dL_dscales.contiguous().data_ptr<float>(),
        dL_drotations.contiguous().data_ptr<float>(),
        dL_dtau.contiguous().data_ptr<float>(),
        debug);
  }

  return std::make_tuple(dL_dmeans2D,
                         dL_dcolors,
                         dL_dsemantics,
                         dL_dopacity,
                         dL_dmeans3D,
                         dL_dcov3D,
                         dL_dsh,
                         dL_dsemantic_sh,
                         dL_dscales,
                         dL_drotations,
                         dL_dtau);
}
torch::Tensor markVisible(torch::Tensor& means3D,
                          torch::Tensor& viewmatrix,
                          torch::Tensor& projmatrix) {
  const int P = means3D.size(0);

  torch::Tensor present =
      torch::full({P}, false, means3D.options().dtype(at::kBool));

  if (P != 0) {
    CudaRasterizer::Rasterizer::markVisible(
        P,
        means3D.contiguous().data_ptr<float>(),
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        present.contiguous().data_ptr<bool>());
  }

  return present;
}
