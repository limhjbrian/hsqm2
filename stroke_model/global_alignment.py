"""Inference-only uniform scale/translation; predictions return to input coordinates."""
import torch
from torch.nn import functional as F


def _bounds(image):
    points = (image[0] > .5).nonzero()
    if not len(points):
        raise ValueError('Global alignment requires nonempty input and reference ink')
    lo, hi = points.amin(0), points.amax(0)
    # Pixel-edge boxes and pixel-center coordinates, in x,y order.
    center = ((lo + hi).float() / 2).flip(0)
    extent = (hi - lo + 1).float().flip(0)
    return center, extent


def alignment_matrices(image, reference):
    """Return grid_sample matrices aligned->input and input->aligned (B,2,3).

    Fit the complete foreground box inside the reference box, preserving aspect
    ratio. No rotation, shear, stroke repair, or label information is used.
    """
    if image.shape != reference.shape or image.ndim != 4 or image.shape[1] != 1:
        raise ValueError('Expected matching Bx1xHxW image/reference')
    h, w = image.shape[-2:]
    dimensions = image.new_tensor([w, h], dtype=torch.float32)
    forward, inverse = [], []
    for source, target in zip(image, reference):
        source_center, source_extent = _bounds(source)
        target_center, target_extent = _bounds(target)
        scale = (target_extent / source_extent).amin()
        source_center = (source_center + .5) * 2 / dimensions - 1
        target_center = (target_center + .5) * 2 / dimensions - 1
        matrix = torch.zeros((2, 3), device=image.device, dtype=torch.float32)
        matrix[:, :2] = torch.eye(2, device=image.device) / scale
        matrix[:, 2] = source_center - target_center / scale
        reverse = torch.zeros_like(matrix)
        reverse[:, :2] = torch.eye(2, device=image.device) * scale
        reverse[:, 2] = target_center - source_center * scale
        forward.append(matrix)
        inverse.append(reverse)
    return torch.stack(forward), torch.stack(inverse)


def _sample(values, matrix, padding_mode='zeros'):
    grid = F.affine_grid(matrix, values.shape, align_corners=False)
    return F.grid_sample(values.float(), grid, align_corners=False,
                         padding_mode=padding_mode)


def predict_with_global_alignment(model, image, reference, strokes, enabled=False):
    """Keep checkpoints unchanged; restore probabilities, then use original ink.

    Output coordinates are the original model-resolution input canvas, not the
    normalized canvas. Model-local flow/hints are explicitly marked as aligned.
    """
    if not enabled:
        return model(image, reference, strokes)
    theta, inverse = alignment_matrices(image, reference)
    aligned = _sample(image, theta)
    output = model(aligned, reference, strokes)
    probabilities = _sample(output['probabilities'], inverse, padding_mode='border')
    probabilities = probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)
    result = dict(probabilities=probabilities, masks=image * probabilities,
                  global_alignment_theta=theta, global_alignment_inverse=inverse,
                  global_aligned_image=aligned,
                  global_aligned_probabilities=output['probabilities'])
    if 'active' in output:
        result['active'] = output['active']
    for key in ('flow', 'hints'):
        if key in output:
            result['global_aligned_' + key] = output[key]
    return result


def preview_global_alignment(image, reference, aligned, path, show=True):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, content, title in zip(axes, (image, aligned, reference),
                                   ('Original input', 'Aligned model input', 'Reference')):
        ax.imshow(content[0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    if show:
        plt.show()
    plt.close(figure)
