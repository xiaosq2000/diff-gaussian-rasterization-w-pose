#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from utils.semantic_decoder import SemanticDecoder
from . import _C


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    background_color: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    projmatrix_raw: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings: GaussianRasterizationSettings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )


def rasterize_semantic_gaussians(
    means3D,
    means2D,
    sh,
    semantic_sh,
    colors_precomp,
    semantics_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    theta,
    rho,
    raster_settings: GaussianRasterizationSettings,
):
    return _RasterizeSemanticGaussians.apply(
        means3D,
        means2D,
        sh,
        semantic_sh,
        colors_precomp,
        semantics_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings: GaussianRasterizationSettings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.background_color,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                    depth,
                    opacity,
                    n_touched,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
                depth,
                opacity,
                n_touched,
            ) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii, depth, opacity, n_touched

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_radii,
        grad_out_depth,
        grad_out_opacity,
        grad_n_touched,
    ):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.background_color,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_depth,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                grad_tau,
            ) = _C.rasterize_gaussians_backward(*args)

        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
        )

        return grads


class _RasterizeSemanticGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        semantic_sh,
        colors_precomp,
        semantics_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        theta,
        rho,
        raster_settings: GaussianRasterizationSettings,
    ):
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.background_color,
            raster_settings.background_color,  # TODO
            means3D,
            colors_precomp,
            semantics_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            semantic_sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    colors,
                    semantics,
                    radii,
                    semantic_geometry_buffer,
                    binning_buffer,
                    image_buffer,
                    depth,
                    opacity,
                    n_touched,
                ) = _C.rasterize_semantic_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                colors,
                semantics,
                radii,
                semantic_geometry_buffer,
                binning_buffer,
                image_buffer,
                depth,
                opacity,
                n_touched,
            ) = _C.rasterize_semantic_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            semantics_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            semantic_sh,
            semantic_geometry_buffer,
            binning_buffer,
            image_buffer,
        )
        return colors, semantics, radii, depth, opacity, n_touched

    @staticmethod
    def backward(
        ctx,
        grad_out_color,
        grad_out_semantics,
        grad_out_radii,
        grad_out_depth,
        grad_out_opacity,
        grad_n_touched,
    ):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            semantics_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            semantic_sh,
            semantic_geometry_buffer,
            binning_buffer,
            image_buffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            raster_settings.background_color,
            raster_settings.background_color,  # TODO
            means3D,
            radii,
            colors_precomp,
            semantics_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.projmatrix_raw,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_out_color,
            grad_out_semantics,
            grad_out_depth,
            sh,
            semantic_sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            semantic_geometry_buffer,
            num_rendered,
            binning_buffer,
            image_buffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_semantics_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_semantic_sh,
                    grad_scales,
                    grad_rotations,
                    grad_tau,
                ) = _C.rasterize_semantic_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_semantics_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_semantic_sh,
                grad_scales,
                grad_rotations,
                grad_tau,
            ) = _C.rasterize_semantic_gaussians_backward(*args)

        grad_tau = torch.sum(grad_tau.view(-1, 6), dim=0)
        grad_rho = grad_tau[:3].view(1, -1)
        grad_theta = grad_tau[3:].view(1, -1)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_semantic_sh,
            grad_colors_precomp,
            grad_semantics_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_theta,
            grad_rho,
            None,
        )

        return grads


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            theta,
            rho,
            raster_settings,
        )


class SemanticGaussianRasterizer(nn.Module):
    def __init__(
        self,
        raster_settings: GaussianRasterizationSettings,
        semantic_decoder: SemanticDecoder,
    ):
        super().__init__()
        self.raster_settings = raster_settings
        self.semantic_decoder = semantic_decoder

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, raster_settings.viewmatrix, raster_settings.projmatrix
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        semantic_shs=None,
        colors_precomp=None,
        semantics_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        theta=None,
        rho=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )
        if (semantic_shs is None and semantics_precomp is None) or (
            semantic_shs is not None and semantics_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either semantic SHs or precomputed semantics!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if semantic_shs is None:
            semantic_shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if semantics_precomp is None:
            semantics_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if theta is None:
            theta = torch.Tensor([])
        if rho is None:
            rho = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        colors, semantics, radii, depth, opacity, n_touched = (
            rasterize_semantic_gaussians(
                means3D,
                means2D,
                shs,
                semantic_shs,
                colors_precomp,
                semantics_precomp,
                opacities,
                scales,
                rotations,
                cov3D_precomp,
                theta,
                rho,
                raster_settings,
            )
        )
        # Decode
        decoded_semantics = self.semantic_decoder(semantics)

        return colors, semantics, decoded_semantics, radii, depth, opacity, n_touched
