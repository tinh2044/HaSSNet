from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        difference = prediction - target
        loss = torch.sqrt(difference * difference + self.epsilon * self.epsilon)
        return loss.mean()


class _VGGFeatureExtractor(nn.Module):
    LAYER_TO_INDEX: Dict[str, int] = {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu3_4": 17,
        "relu4_1": 20,
        "relu4_2": 22,
        "relu4_3": 24,
        "relu4_4": 26,
        "relu5_1": 29,
        "relu5_2": 31,
        "relu5_3": 33,
        "relu5_4": 35,
    }

    def __init__(self, layers: Sequence[str]) -> None:
        super().__init__()
        try:
            from torchvision import models
        except Exception as exc:
            raise ImportError(
                "PerceptualLoss requires torchvision to be installed."
            ) from exc

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="torchvision.models._utils"
                )
                vgg = models.vgg19(
                    weights=getattr(models, "VGG19_Weights", object).IMAGENET1K_V1
                )  # type: ignore[attr-defined]
        except Exception:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, module="torchvision.models._utils"
                )
                vgg = models.vgg19(pretrained=True)

        self.features = vgg.features
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.features.eval()

        for layer in layers:
            if layer not in self.LAYER_TO_INDEX:
                raise ValueError(f"Unsupported VGG layer '{layer}'.")
        self.layers = list(layers)
        self.max_index = max(self.LAYER_TO_INDEX[layer] for layer in self.layers)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.size(1) != 3:
            raise ValueError("Images must have 1 or 3 channels for VGG features.")

        x = images.to(dtype=torch.float32)
        x = (x - self.imagenet_mean) / self.imagenet_std

        outputs: Dict[str, torch.Tensor] = {}
        out = x
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx in (self.LAYER_TO_INDEX[layer_name] for layer_name in self.layers):
                for name, index in self.LAYER_TO_INDEX.items():
                    if idx == index and name in self.layers:
                        outputs[name] = out
            if idx >= self.max_index:
                break
        return outputs


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layers: Sequence[str] = ("relu2_2", "relu3_3"),
        layer_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.extractor = _VGGFeatureExtractor(layers)
        if layer_weights is None:
            self.layer_weights = [1.0 / len(layers)] * len(layers)
        else:
            if len(layer_weights) != len(layers):
                raise ValueError("layer_weights must match layers length.")
            total = float(sum(layer_weights))
            if total <= 0:
                raise ValueError("layer_weights must sum to > 0.")
            self.layer_weights = [float(w) / total for w in layer_weights]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            target_feats = self.extractor.extract(target)
        pred_feats = self.extractor.extract(prediction)

        loss = prediction.new_tensor(0.0)
        for layer_name, weight in zip(self.extractor.layers, self.layer_weights):
            a = pred_feats[layer_name]
            b = target_feats[layer_name]
            loss = loss + weight * F.mse_loss(a, b, reduction="mean")
        return loss


class GradientLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(epsilon)

    @staticmethod
    def _gradients(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad_x = images[:, :, 1:, :] - images[:, :, :-1, :]
        grad_y = images[:, :, :, 1:] - images[:, :, :, :-1]
        return grad_x, grad_y

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gx_pred, gy_pred = self._gradients(prediction)
        gx_tgt, gy_tgt = self._gradients(target)
        loss_x = self.charbonnier(gx_pred, gx_tgt)
        loss_y = self.charbonnier(gy_pred, gy_tgt)
        return loss_x + loss_y


class FrequencyLoss(nn.Module):
    def __init__(self, delta: float = 1e-6) -> None:
        super().__init__()
        self.delta = float(delta)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred32 = prediction.to(dtype=torch.float32)
        tgt32 = target.to(dtype=torch.float32)

        f_pred = torch.fft.rfft2(pred32, dim=(-2, -1))
        f_tgt = torch.fft.rfft2(tgt32, dim=(-2, -1))
        m_pred = torch.log(torch.abs(f_pred) + self.delta)
        m_tgt = torch.log(torch.abs(f_tgt) + self.delta)
        return torch.abs(m_pred - m_tgt).mean()


class HighFrequencyBiasLoss(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel_size = int(kernel_size)
        self.charbonnier = CharbonnierLoss()

    @staticmethod
    def _make_depthwise_avg_kernel(num_channels: int, kernel_size: int, device, dtype):
        kernel = torch.ones(
            (num_channels, 1, kernel_size, kernel_size), device=device, dtype=dtype
        )
        kernel = kernel / (kernel_size * kernel_size)
        return kernel

    def forward(
        self, es_prediction: torch.Tensor, target_image: torch.Tensor
    ) -> torch.Tensor:
        if target_image.dim() != 4:
            raise ValueError("target_image must be a 4D tensor (B,C,H,W).")
        batch, channels, height, width = target_image.shape
        padding = self.kernel_size // 2

        kernel = self._make_depthwise_avg_kernel(
            num_channels=channels,
            kernel_size=self.kernel_size,
            device=target_image.device,
            dtype=target_image.dtype,
        )

        blurred = F.conv2d(target_image, kernel, padding=padding, groups=channels)
        high_pass_gt = target_image - blurred
        return self.charbonnier(es_prediction, high_pass_gt)


class MainLoss(nn.Module):
    def __init__(self, loss_weights: Dict[str, float]) -> None:
        super().__init__()
        self.weights: Dict[str, float] = {
            "rec": float(loss_weights.get("rec", 1.0)),
            "prec": float(loss_weights.get("prec", 1.0)),
            "grad": float(loss_weights.get("grad", 1.0)),
            "freq": float(loss_weights.get("freq", 1.0)),
            "e": float(loss_weights.get("e", 1.0)),
        }

        self.charbonnier_loss = CharbonnierLoss() if self.weights["rec"] > 0 else None

        self.perceptual_loss: Optional[PerceptualLoss]
        if self.weights["prec"] > 0:
            try:
                self.perceptual_loss = PerceptualLoss()
            except Exception:
                self.perceptual_loss = None
        else:
            self.perceptual_loss = None

        self.grad_loss = GradientLoss() if self.weights["grad"] > 0 else None
        self.freq_loss = FrequencyLoss() if self.weights["freq"] > 0 else None
        self.e_loss = HighFrequencyBiasLoss() if self.weights["e"] > 0 else None

    @staticmethod
    def _extract_outputs(outputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(outputs, torch.Tensor):
            return outputs, None
        if isinstance(outputs, dict):
            image_pred = None
            for key in ("pred", "image", "out", "output"):
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    image_pred = outputs[key]
                    break
            if image_pred is None:
                raise ValueError("Cannot find image prediction in outputs dict.")

            e_s_pred = None
            for key in ("e_s", "es", "highfreq"):
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    e_s_pred = outputs[key]
                    break
            return image_pred, e_s_pred

        raise TypeError("outputs must be a Tensor or a dict with Tensor values.")

    def forward(
        self, outputs, target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        image_pred, e_s_pred = self._extract_outputs(outputs)
        if target is None:
            target = image_pred

        losses: Dict[str, torch.Tensor] = {}
        total_loss = image_pred.new_tensor(0.0)

        if self.charbonnier_loss is not None:
            losses["rec"] = self.charbonnier_loss(image_pred, target)
            total_loss = total_loss + self.weights["rec"] * losses["rec"]

        if self.weights["prec"] > 0:
            if self.perceptual_loss is None:
                losses["prec"] = image_pred.new_tensor(0.0)
            else:
                losses["prec"] = self.perceptual_loss(image_pred, target)
            total_loss = total_loss + self.weights["prec"] * losses["prec"]

        if self.grad_loss is not None:
            losses["grad"] = self.grad_loss(image_pred, target)
            total_loss = total_loss + self.weights["grad"] * losses["grad"]

        if self.freq_loss is not None:
            losses["freq"] = self.freq_loss(image_pred, target)
            total_loss = total_loss + self.weights["freq"] * losses["freq"]

        if self.e_loss is not None:
            if e_s_pred is not None:
                losses["e"] = self.e_loss(e_s_pred, target)
            else:
                losses["e"] = image_pred.new_tensor(0.0)
            total_loss = total_loss + self.weights["e"] * losses["e"]

        losses["total"] = total_loss
        return losses
