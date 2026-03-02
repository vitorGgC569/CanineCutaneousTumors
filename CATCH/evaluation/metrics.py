from fastai.vision import *

def _tensor_iou(outputs_max: torch.Tensor, labels_squeezed: torch.Tensor, label_idx: int = None):
    # Computação nativa em GPU para evitar gargalo de transferência para CPU
    if label_idx is not None:
        pred_mask = (outputs_max == label_idx)
        target_mask = (labels_squeezed == label_idx)
    else:
        pred_mask = outputs_max.bool() # Simplificação, idealmente calcular por classe
        target_mask = labels_squeezed.bool()
        
    intersection = (pred_mask & target_mask).float().sum()
    union = (pred_mask | target_mask).float().sum()
    
    if union == 0:
        return tensor(float('nan')) # ou 1.0 dependendo da convenção, fastai lida bem com nan no mean
    return intersection / union

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    # Para IoU geral, a implementação original tirava a média de todas as classes.
    # Vamos calcular por classe e tirar a média.
    classes = torch.unique(labels_squeezed)
    ious = []
    for c in classes:
        val = _tensor_iou(outputs_max, labels_squeezed, c.item())
        if not torch.isnan(val):
            ious.append(val)
    if not ious:
        return tensor(0.)
    return torch.stack(ious).mean()

def background_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=0)

def dermis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=1)

def epidermis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=2)

def subcutis_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=3)

def infl_nec_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=4)

def tumor_iou(outputs: torch.Tensor, labels: torch.Tensor):
    outputs_max = outputs.argmax(dim=1)
    labels_squeezed = labels.squeeze(1)
    return _tensor_iou(outputs_max, labels_squeezed, label_idx=5)
