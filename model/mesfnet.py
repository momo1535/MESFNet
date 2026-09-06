import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models import convnext_base, convnext_large


def normalize_convnext_variant(variant):
    variant = (variant or "base").lower().replace("-", "_")
    aliases = {
        "b": "base",
        "convnext_b": "base",
        "convnext_base": "base",
        "l": "large",
        "convnext_l": "large",
        "convnext_large": "large",
    }
    variant = aliases.get(variant, variant)
    if variant not in {"base", "large"}:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}. Use 'base' or 'large'.")
    return variant


def build_convnext_backbone(variant: str, pretrained: bool):
    """Support ConvNeXt-Base/Large and both old/new torchvision weight APIs."""
    variant = normalize_convnext_variant(variant)
    if variant == "large":
        builder = convnext_large
        channels = [192, 384, 768, 1536]
        weights_name = "ConvNeXt_Large_Weights"
    else:
        builder = convnext_base
        channels = [128, 256, 512, 1024]
        weights_name = "ConvNeXt_Base_Weights"

    try:
        import torchvision.models as tv_models

        weights_cls = getattr(tv_models, weights_name)
        weights = weights_cls.DEFAULT if pretrained else None
        return builder(weights=weights), channels
    except (AttributeError, TypeError):
        return builder(pretrained=pretrained), channels


class GatedDetailEnhancement(nn.Module):
    """Gated residual shallow detail enhancement.

    It extracts local detail with depthwise
    convolutions, then controls the residual with channel and spatial gates.
    """

    def __init__(self, channels: int, reduction: int = 8, freeze_pool_input: bool = False):
        super().__init__()
        self.freeze_pool_input = bool(freeze_pool_input)
        hidden = max(16, channels // reduction)

        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        detail = self.detail(x)
        channel_source = x.detach() if self.freeze_pool_input and self.training else x
        gate = self.channel_gate(channel_source) * self.spatial_gate(detail)
        return x + self.res_scale * gate * detail


class PyramidContextAggregation(nn.Module):
    def __init__(self, features, out_features=256, sizes=(1, 3, 5, 7), freeze_pool_input=False):
        super().__init__()
        self.freeze_pool_input = bool(freeze_pool_input)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(features, out_features, kernel_size=1, bias=False),
                    nn.GroupNorm(32, out_features),
                    nn.ReLU(inplace=True),
                )
                for size in sizes
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        height, width = x.size(2), x.size(3)
        pooled_source = x.detach() if self.freeze_pool_input and self.training else x
        priors = [
            F.interpolate(stage(pooled_source), size=(height, width), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        priors.append(x)
        return self.bottleneck(torch.cat(priors, dim=1))


class SelectiveKernelAttention(nn.Module):
    def __init__(self, channel=256, kernels=(1, 3, 5, 7), reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            ("conv", nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                            ("norm", nn.BatchNorm2d(channel)),
                            ("relu", nn.ReLU(inplace=True)),
                        ]
                    )
                )
                for k in kernels
            ]
        )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        fused = sum(conv_outs)
        pooled = fused.mean(-1).mean(-1)
        hidden = self.fc(pooled)
        weights = torch.stack([fc(hidden) for fc in self.fcs], dim=0)
        weights = self.softmax(weights).view(len(self.convs), batch_size, channels, 1, 1)
        return (weights * torch.stack(conv_outs, dim=0)).sum(dim=0)


class SelectiveKernelResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sk = SelectiveKernelAttention(channel=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.sk(x)
        out = self.conv(out)
        out = self.norm(out)
        return self.relu(out + x)


class LearnedFeatureAlignment(nn.Module):
    def __init__(self, high_ch, low_ch, fixed_coordinates=False):
        super().__init__()
        self.fixed_coordinates = bool(fixed_coordinates)
        self.flow_make = nn.Conv2d(high_ch + low_ch, 2, kernel_size=3, padding=1)

    def forward(self, x_high, x_low):
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:], mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([x_high_up, x_low], dim=1))
        batch_size, _, height, width = x_high_up.size()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=x_high.device, dtype=x_high_up.dtype),
            torch.arange(width, device=x_high.device, dtype=x_high_up.dtype),
            indexing="ij",
        )
        grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).expand(batch_size, -1, -1, -1)
        v_grid = grid + flow.permute(0, 2, 3, 1)
        if self.fixed_coordinates:
            # Pixel-center convention required by align_corners=False.
            v_grid[:, :, :, 0] = (2.0 * v_grid[:, :, :, 0] + 1.0) / width - 1.0
            v_grid[:, :, :, 1] = (2.0 * v_grid[:, :, :, 1] + 1.0) / height - 1.0
        else:
            # Historical convention retained for old checkpoint compatibility.
            v_grid[:, :, :, 0] = (v_grid[:, :, :, 0] / (width - 1) - 0.5) * 2
            v_grid[:, :, :, 1] = (v_grid[:, :, :, 1] / (height - 1) - 0.5) * 2
        return F.grid_sample(x_high_up, v_grid, mode="bilinear", padding_mode="border", align_corners=False)


class BoundaryResidualRefinement(nn.Module):
    """Predict a half-resolution mask residual from texture, shallow detail, and edges."""

    def __init__(self, shallow_channels, hidden_channels=32):
        super().__init__()
        shallow_out = 8
        self.shallow_projection = nn.Sequential(
            nn.Conv2d(shallow_channels, shallow_out, kernel_size=1, bias=False),
            nn.GroupNorm(4, shallow_out),
            nn.GELU(),
        )
        input_channels = 3 + 1 + 1 + shallow_out
        self.residual = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels // 2),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, image, shallow_feature, coarse_logits, edge_logits):
        half_size = (max(1, image.shape[2] // 2), max(1, image.shape[3] // 2))
        image_half = F.interpolate(image, size=half_size, mode="bilinear", align_corners=False)
        shallow_half = F.interpolate(
            self.shallow_projection(shallow_feature),
            size=half_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_half = F.interpolate(coarse_logits, size=half_size, mode="bilinear", align_corners=False)
        edge_half = F.interpolate(edge_logits, size=half_size, mode="bilinear", align_corners=False)
        residual = self.residual(torch.cat([image_half, shallow_half, coarse_half, edge_half], dim=1))
        return coarse_half + residual


class ZeroInitBoundaryResidualRefinement(nn.Module):
    """Half-resolution residual refinement with a zero-initialized output."""

    def __init__(self, shallow_channels, edge_channels, hidden_channels=32):
        super().__init__()
        projected_channels = 8
        self.shallow_projection = nn.Sequential(
            nn.Conv2d(shallow_channels, projected_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, projected_channels),
            nn.GELU(),
        )
        self.edge_projection = nn.Sequential(
            nn.Conv2d(edge_channels, projected_channels, kernel_size=1, bias=False),
            nn.GroupNorm(4, projected_channels),
            nn.GELU(),
        )
        input_channels = 3 + 1 + 1 + projected_channels * 2
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, hidden_channels // 2),
            nn.GELU(),
        )
        self.output = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image, shallow_feature, edge_feature, coarse_logits, edge_logits):
        half_size = (max(1, image.shape[2] // 2), max(1, image.shape[3] // 2))
        image_half = F.interpolate(image, size=half_size, mode="bilinear", align_corners=False)
        shallow_half = F.interpolate(
            self.shallow_projection(shallow_feature),
            size=half_size,
            mode="bilinear",
            align_corners=False,
        )
        edge_feature_half = F.interpolate(
            self.edge_projection(edge_feature),
            size=half_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_half = F.interpolate(coarse_logits, size=half_size, mode="bilinear", align_corners=False)
        edge_probability_half = torch.sigmoid(
            F.interpolate(edge_logits, size=half_size, mode="bilinear", align_corners=False)
        )
        refinement_input = torch.cat(
            [image_half, shallow_half, edge_feature_half, coarse_half, edge_probability_half],
            dim=1,
        )
        return self.output(self.features(refinement_input))


def batch_norm_to_group_norm(batch_norm, max_groups=32):
    """Create a shape-compatible GroupNorm from a BatchNorm2d layer."""
    if not isinstance(batch_norm, nn.BatchNorm2d):
        raise TypeError(f"Expected BatchNorm2d, got {type(batch_norm).__name__}")
    groups = min(max_groups, batch_norm.num_features)
    while batch_norm.num_features % groups != 0:
        groups -= 1
    replacement = nn.GroupNorm(
        groups,
        batch_norm.num_features,
        eps=batch_norm.eps,
        affine=batch_norm.affine,
    )
    if batch_norm.affine:
        with torch.no_grad():
            replacement.weight.copy_(batch_norm.weight)
            replacement.bias.copy_(batch_norm.bias)
    return replacement


def replace_batch_norm_with_group_norm(module, max_groups=32):
    """Replace BatchNorm2d recursively while preserving affine initialization."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, batch_norm_to_group_norm(child, max_groups=max_groups))
        else:
            replace_batch_norm_with_group_norm(child, max_groups=max_groups)


class HierarchicalSemanticFusion(nn.Module):
    def __init__(self, in_channels_list, out_ch=256):
        super().__init__()
        self.cvs = nn.ModuleList([nn.Conv2d(channels, out_ch, kernel_size=1) for channels in in_channels_list])
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        projected = [cv(feature) for cv, feature in zip(self.cvs, inputs)]
        p4 = projected[3]
        p3 = projected[2] + F.interpolate(p4, size=projected[2].shape[2:], mode="bilinear", align_corners=False)
        p2 = projected[1] + F.interpolate(p3, size=projected[1].shape[2:], mode="bilinear", align_corners=False)
        p1 = projected[0] + F.interpolate(p2, size=projected[0].shape[2:], mode="bilinear", align_corners=False)
        return self.final_conv(p1)


class ConvNormActivation(nn.Sequential):
    """Fixed-kernel fallback used by decoder ablations."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ShallowSemanticProjection(nn.Module):
    """Shape-compatible fallback that only uses the shallowest feature map."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = ConvNormActivation(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs):
        return self.project(inputs[0])


class BilinearFeatureAlignment(nn.Module):
    """Non-learned replacement for flow-based feature alignment."""

    def forward(self, x_high, x_low):
        return F.interpolate(x_high, size=x_low.shape[2:], mode="bilinear", align_corners=False)


class GatedBoundaryFusion(nn.Module):
    """Fuse shallow detail into the edge path without flow warping."""

    def __init__(self, high_ch, low_ch, out_ch=64):
        super().__init__()
        self.out_channels = out_ch
        self.high_proj = ConvNormActivation(high_ch, out_ch, kernel_size=1)
        self.low_proj = ConvNormActivation(low_ch, out_ch, kernel_size=1)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_high, x_low):
        high = F.interpolate(x_high, size=x_low.shape[2:], mode="bilinear", align_corners=False)
        high = self.high_proj(high)
        low = self.low_proj(x_low)
        gate_input = torch.cat(
            [high.mean(dim=1, keepdim=True), low.mean(dim=1, keepdim=True)],
            dim=1,
        )
        gate = self.spatial_gate(gate_input)
        return self.refine(high + gate * low)


class MESFNet(nn.Module):
    """Multi-scale Edge-supervised Semantic Fusion Network."""

    def __init__(
        self,
        num_classes=1,
        pretrained=True,
        backbone_variant=None,
        ablations=None,
        freeze_nondeterministic_paths=False,
    ):
        super().__init__()
        self.ablations = frozenset(ablations or ())
        self.freeze_nondeterministic_paths = bool(freeze_nondeterministic_paths)
        self.backbone_variant = normalize_convnext_variant(
            backbone_variant or os.environ.get("CONVNEXT_VARIANT", "base")
        )
        backbone, c_list = build_convnext_backbone(self.backbone_variant, pretrained=pretrained)
        self.stem = backbone.features[0]
        self.encoder_stage1 = backbone.features[1]
        self.encoder_stage2 = nn.Sequential(backbone.features[2], backbone.features[3])
        self.encoder_stage3 = nn.Sequential(backbone.features[4], backbone.features[5])
        self.encoder_stage4 = nn.Sequential(backbone.features[6], backbone.features[7])

        dec_ch = 256

        detail_factory = (
            (lambda channels: nn.Identity())
            if "no_scp" in self.ablations
            else (lambda channels: GatedDetailEnhancement(channels, freeze_pool_input=self.freeze_nondeterministic_paths))
        )
        self.detail_enhance1 = detail_factory(c_list[0])
        self.detail_enhance2 = detail_factory(c_list[1])
        self.detail_enhance3 = detail_factory(c_list[2])

        if "no_spam" in self.ablations:
            self.context_aggregation = ConvNormActivation(c_list[3], dec_ch, kernel_size=1)
        else:
            self.context_aggregation = PyramidContextAggregation(
                c_list[3],
                dec_ch,
                freeze_pool_input=self.freeze_nondeterministic_paths,
            )
        if "no_semantic_fusion" in self.ablations:
            self.semantic_fusion = ShallowSemanticProjection(c_list[0], dec_ch)
        else:
            self.semantic_fusion = HierarchicalSemanticFusion(c_list, dec_ch)

        self.lateral_projection3 = nn.Conv2d(c_list[2], dec_ch, kernel_size=1)
        decoder_factory = (lambda channels: ConvNormActivation(channels, channels)) if "no_sk" in self.ablations else SelectiveKernelResidualBlock
        self.decoder_stage4 = decoder_factory(dec_ch)
        self.decoder_stage3 = decoder_factory(dec_ch)

        edge_channels = dec_ch
        if "light_edge_fusion" in self.ablations:
            self.edge_alignment = GatedBoundaryFusion(dec_ch, c_list[0])
            edge_channels = self.edge_alignment.out_channels
        elif "no_flow_align" in self.ablations:
            self.edge_alignment = BilinearFeatureAlignment()
        else:
            self.edge_alignment = LearnedFeatureAlignment(
                dec_ch,
                c_list[0],
                fixed_coordinates="fixed_flow_align" in self.ablations,
            )
        if self.freeze_nondeterministic_paths:
            for parameter in self.edge_alignment.parameters():
                parameter.requires_grad_(False)

        self.segmentation_fusion = nn.Sequential(
            nn.Conv2d(dec_ch * 3, dec_ch, kernel_size=1),
            nn.BatchNorm2d(dec_ch),
            nn.ReLU(inplace=True),
        )
        self.segmentation_head = nn.Conv2d(dec_ch, num_classes, kernel_size=1)
        self.boundary_head = (
            None if "no_edge_branch" in self.ablations else nn.Conv2d(edge_channels, 1, kernel_size=1)
        )
        if "edge_guided_refine" in self.ablations and "edge_guided_refine_v2" in self.ablations:
            raise ValueError("edge_guided_refine and edge_guided_refine_v2 are mutually exclusive")
        if "edge_guided_refine_v2" in self.ablations:
            if self.boundary_head is None:
                raise ValueError("edge_guided_refine_v2 requires the edge branch")
            self.edge_refinement = ZeroInitBoundaryResidualRefinement(c_list[0], edge_channels)
            self.edge_refinement_version = 2
        elif "edge_guided_refine" in self.ablations:
            if self.boundary_head is None:
                raise ValueError("edge_guided_refine requires the edge branch")
            self.edge_refinement = BoundaryResidualRefinement(c_list[0])
            self.edge_refinement_version = 1
        else:
            self.edge_refinement = None
            self.edge_refinement_version = 0
        self.morphology_head = nn.Sequential(
            nn.Linear(dec_ch, dec_ch),
            nn.ReLU(inplace=True),
            nn.Linear(dec_ch, 4),
        )
        targeted_decoder_gn = bool(
            {"groupnorm_decoder", "groupnorm_decoder_semantic"} & self.ablations
        )
        if targeted_decoder_gn:
            self.decoder_stage4.norm = batch_norm_to_group_norm(self.decoder_stage4.norm)
            self.decoder_stage3.norm = batch_norm_to_group_norm(self.decoder_stage3.norm)
            self.segmentation_fusion[1] = batch_norm_to_group_norm(self.segmentation_fusion[1])
        if "groupnorm_decoder_semantic" in self.ablations:
            self.semantic_fusion.final_conv[1] = batch_norm_to_group_norm(
                self.semantic_fusion.final_conv[1]
            )
        if "groupnorm_all" in self.ablations:
            replace_batch_norm_with_group_norm(self)

    def forward(self, x):
        height, width = x.shape[2:]

        f1 = self.detail_enhance1(self.encoder_stage1(self.stem(x)))
        f2 = self.detail_enhance2(self.encoder_stage2(f1))
        f3 = self.detail_enhance3(self.encoder_stage3(f2))
        f4 = self.encoder_stage4(f3)

        f4_s = self.context_aggregation(f4)
        p4 = self.decoder_stage4(f4_s)
        p3 = self.decoder_stage3(self.lateral_projection3(f3) + F.interpolate(p4, size=f3.shape[2:], mode="bilinear", align_corners=False))

        global_semantic = self.semantic_fusion([f1, f2, f3, f4])
        p3_u = F.interpolate(p3, size=f1.shape[2:], mode="bilinear", align_corners=False)
        p4_u = F.interpolate(p4, size=f1.shape[2:], mode="bilinear", align_corners=False)
        mask_feat = self.segmentation_fusion(torch.cat([global_semantic, p3_u, p4_u], dim=1))

        if self.freeze_nondeterministic_paths and self.training:
            with torch.no_grad():
                edge_feat = self.edge_alignment(p4.detach(), f1.detach())
        else:
            edge_feat = self.edge_alignment(p4, f1)

        coarse_mask = self.segmentation_head(mask_feat)
        if self.boundary_head is None:
            edge_native = coarse_mask.new_zeros(coarse_mask.shape)
        else:
            edge_native = self.boundary_head(edge_feat)
        if self.edge_refinement is None:
            mask_native = coarse_mask
        elif self.edge_refinement_version == 2:
            residual_half = self.edge_refinement(x, f1, edge_feat, coarse_mask, edge_native)
            mask = F.interpolate(coarse_mask, size=(height, width), mode="bilinear", align_corners=False)
            mask = mask + F.interpolate(residual_half, size=(height, width), mode="bilinear", align_corners=False)
            mask_native = None
        else:
            mask_native = self.edge_refinement(x, f1, coarse_mask, edge_native)
        if mask_native is not None:
            mask = F.interpolate(mask_native, size=(height, width), mode="bilinear", align_corners=False)
        edge = F.interpolate(edge_native, size=(height, width), mode="bilinear", align_corners=False)
        morph_source = p4.detach() if self.freeze_nondeterministic_paths and self.training else p4
        morph = self.morphology_head(F.adaptive_avg_pool2d(morph_source, 1).view(p4.size(0), -1))

        return {"mask": mask, "edge": edge, "morph": morph}




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MESFNet(num_classes=1, pretrained=False).to(device).eval()
    sample = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        outputs = model(sample)
    print({key: tuple(value.shape) for key, value in outputs.items()})
