"""Fine-tune RETFound-DINOv2 (ViT-L/14, retinal-image pretraining) for sex
classification on mBRSET fundus photos, against the ConvNeXtV2-Large baseline.

Canonical split
---------------
A single patient-level 80/20 train/test split, stratified by sex so the
label ratio is preserved in both partitions. All 4 images of a patient land in
exactly one partition (no identity leakage). A validation subset (12.5% of the
train patients, i.e. 10% of all patients) is carved out of the *train* side for
hyperparameter selection and early stopping -- the test side is never touched
until final evaluation. The split is created once (seed=42), saved to
``splits/patient_split.csv``, and every model (linear, non-linear, ConvNeXt,
RETFound) loads that same file.

Usage
-----
  # create the split only
  python retfound_dinov2_sex.py --make-split-only

  # LR sweep (short runs scored on val) + full fine-tune of RETFound-DINOv2
  python retfound_dinov2_sex.py --backbone retfound_dinov2 --sweep

  # retrain the ConvNeXtV2-Large baseline on the identical split
  python retfound_dinov2_sex.py --backbone convnextv2_large

  # held-out test evaluation of a saved checkpoint
  python retfound_dinov2_sex.py --backbone retfound_dinov2 --eval-only

From a notebook:
  from retfound_dinov2_sex import (load_patient_split, attach_split,
                                   build_model, load_finetuned, eval_transform)
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '/oriondata/AIMLab/renee.zac/Datasets/mbrset/mbrset-a-mobile-brazilian-retinal-dataset-1.0/'
IMAGES_PATH = os.path.join(DATA_PATH, 'images/')
LABELS_CSV_PATH = os.path.join(DATA_PATH, 'labels_mbrset.csv')
SPLIT_CSV = os.path.join(REPO_DIR, 'splits', 'patient_split.csv')
MODELS_DIR = os.path.join(REPO_DIR, 'Models')

RETFOUND_REPO = 'YukunZhou/RETFound_dinov2_shanghai'
RETFOUND_FILE = 'RETFound_dinov2_shanghai.pth'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42
LABEL = 'sex'          # 0 = female, 1 = male
IMAGE_COL = 'file'

BACKBONES = {
    'retfound_dinov2': dict(img_size=224, batch_size=48, backbone_lr=1e-5,
                            head_lr=1e-3, layer_decay=0.75,
                            weights=os.path.join(MODELS_DIR, 'retfound_dinov2_2class_sex_best.pth')),
    'convnextv2_large': dict(img_size=384, batch_size=16, backbone_lr=1e-5,
                             head_lr=1e-4, layer_decay=None,
                             weights=os.path.join(MODELS_DIR, 'convnextv2_large_2class_sex_patientsplit_best.pth')),
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Canonical patient-level stratified split
# --------------------------------------------------------------------------- #
def make_patient_split(labels_df, test_size=0.2, val_size=0.125, seed=SEED):
    """80/20 patient-level split stratified by sex; val = 12.5% of the train
    patients (10% overall), also stratified. Returns a DataFrame
    (patient, sex, split) with split in {train, val, test}."""
    patients = labels_df.drop_duplicates('patient')[['patient', LABEL]].reset_index(drop=True)
    train_full, test = train_test_split(
        patients, test_size=test_size, stratify=patients[LABEL], random_state=seed)
    train, val = train_test_split(
        train_full, test_size=val_size, stratify=train_full[LABEL], random_state=seed)
    out = patients.copy()
    out['split'] = 'train'
    out.loc[out.patient.isin(val.patient), 'split'] = 'val'
    out.loc[out.patient.isin(test.patient), 'split'] = 'test'
    return out


def load_patient_split(labels_df=None, split_csv=SPLIT_CSV):
    """Load the canonical split, creating and saving it on first use."""
    if os.path.exists(split_csv):
        return pd.read_csv(split_csv)
    if labels_df is None:
        labels_df = pd.read_csv(LABELS_CSV_PATH)
    split_df = make_patient_split(labels_df)
    os.makedirs(os.path.dirname(split_csv), exist_ok=True)
    split_df.to_csv(split_csv, index=False)
    print(f'Created canonical patient split -> {split_csv}')
    return split_df


def attach_split(labels_df, split_df=None):
    """Add a 'split' column (train/val/test) to the image-level labels_df."""
    if split_df is None:
        split_df = load_patient_split(labels_df)
    return labels_df.merge(split_df[['patient', 'split']], on='patient', how='left')


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class FundusSexDataset(Dataset):
    """(image tensor, int label) pairs from an mBRSET labels dataframe."""

    def __init__(self, df, images_dir, transform):
        self.files = df[IMAGE_COL].values
        self.labels = df[LABEL].astype(int).values
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.images_dir, self.files[idx])).convert('RGB')
        return self.transform(img), int(self.labels[idx])


def train_transform(backbone):
    size = BACKBONES[backbone]['img_size']
    if backbone == 'retfound_dinov2':
        aug = [transforms.RandomResizedCrop(size, scale=(0.7, 1.0),
                                            interpolation=transforms.InterpolationMode.BICUBIC)]
    else:
        # keep the squash-resize the mBRSET ConvNeXt recipe uses
        aug = [transforms.Resize((size, size))]
    return transforms.Compose(aug + [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(backbone):
    size = BACKBONES[backbone]['img_size']
    if backbone == 'retfound_dinov2':
        # RETFound eval recipe: resize shorter side to size/0.875, center crop
        return transforms.Compose([
            transforms.Resize(int(round(size / 0.875)),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _resolve_retfound_ckpt():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(RETFOUND_REPO, RETFOUND_FILE)
    except Exception as exc:
        raise RuntimeError(
            f'Could not resolve the RETFound-DINOv2 checkpoint {RETFOUND_REPO}/'
            f'{RETFOUND_FILE}: {exc}') from exc


def _interpolate_pos_embed(model, state):
    """Resize the checkpoint pos_embed (518px -> model grid) if needed."""
    if 'pos_embed' not in state:
        return
    ckpt_pe, model_pe = state['pos_embed'], model.pos_embed
    if ckpt_pe.shape == model_pe.shape:
        return
    n_prefix = model_pe.shape[1] - model.patch_embed.num_patches
    dim = ckpt_pe.shape[-1]
    prefix, grid = ckpt_pe[:, :n_prefix], ckpt_pe[:, n_prefix:]
    src = int(grid.shape[1] ** 0.5)
    dst = int(model.patch_embed.num_patches ** 0.5)
    grid = grid.reshape(1, src, src, dim).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(dst, dst),
                                           mode='bicubic', align_corners=False)
    state['pos_embed'] = torch.cat([prefix, grid.permute(0, 2, 3, 1).reshape(1, dst * dst, dim)], dim=1)


def build_retfound_dinov2(num_classes=2, img_size=224, pretrained=True):
    """timm ViT-L/14 (DINOv2 flavour) with the RETFound retinal weights."""
    import timm
    model = timm.create_model('vit_large_patch14_dinov2', pretrained=False,
                              num_classes=num_classes, img_size=img_size)
    if pretrained:
        ckpt = torch.load(_resolve_retfound_ckpt(), map_location='cpu', weights_only=False)
        state = ckpt['teacher']
        state = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
        state.pop('mask_token', None)
        _interpolate_pos_embed(model, state)
        msg = model.load_state_dict(state, strict=False)
        assert not msg.unexpected_keys, f'unexpected keys: {msg.unexpected_keys[:5]}'
        assert set(msg.missing_keys) <= {'head.weight', 'head.bias'}, msg.missing_keys[:5]
        nn.init.trunc_normal_(model.head.weight, std=2e-5)
        nn.init.zeros_(model.head.bias)
    return model


def build_convnextv2(num_classes=2):
    """HF ConvNeXtV2-Large (ImageNet-22k, 384px) with a fresh 2-class head.
    Identical to the notebook's model, so state dicts are interchangeable."""
    from transformers import ConvNextV2ForImageClassification
    model = ConvNextV2ForImageClassification.from_pretrained('facebook/convnextv2-large-22k-384')
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_model(backbone, num_classes=2, pretrained=True):
    if backbone == 'retfound_dinov2':
        return build_retfound_dinov2(num_classes, BACKBONES[backbone]['img_size'], pretrained)
    if backbone == 'convnextv2_large':
        return build_convnextv2(num_classes)
    raise ValueError(backbone)


def load_finetuned(backbone, weights=None, device='cpu'):
    """Rebuild the architecture and load our fine-tuned weights (for notebooks)."""
    weights = weights or BACKBONES[backbone]['weights']
    model = build_model(backbone, pretrained=False)
    state = torch.load(weights, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def logits_of(model, x):
    out = model(x)
    return out.logits if hasattr(out, 'logits') else out


# --------------------------------------------------------------------------- #
# Optimizer with layer-wise LR decay (RETFound fine-tuning recipe)
# --------------------------------------------------------------------------- #
def param_groups(model, backbone, backbone_lr, head_lr, layer_decay, weight_decay=0.05):
    if backbone == 'retfound_dinov2' and layer_decay:
        n_layers = len(model.blocks)

        def layer_id(name):
            if name.startswith(('cls_token', 'pos_embed', 'patch_embed')):
                return 0
            if name.startswith('blocks.'):
                return int(name.split('.')[1]) + 1
            return n_layers + 1  # norm, head

        groups = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lid = layer_id(name)
            lr = head_lr if name.startswith('head') else backbone_lr * layer_decay ** (n_layers + 1 - lid)
            decay = 0.0 if p.ndim == 1 or name.endswith('.gamma') else weight_decay
            key = (lid, decay, name.startswith('head'))
            groups.setdefault(key, {'params': [], 'lr': lr, 'weight_decay': decay})['params'].append(p)
        return list(groups.values())

    head_names = ('classifier', 'head')
    head, body_decay, body_nodecay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(head_names):
            head.append(p)
        elif p.ndim == 1:
            body_nodecay.append(p)
        else:
            body_decay.append(p)
    return [
        {'params': head, 'lr': head_lr, 'weight_decay': weight_decay},
        {'params': body_decay, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': body_nodecay, 'lr': backbone_lr, 'weight_decay': 0.0},
    ]


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            out = logits_of(model, x.to(device, non_blocking=True))
        probs.append(torch.softmax(out.float(), dim=1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def metrics_from(probs, ys):
    preds = probs.argmax(1)
    return dict(
        accuracy=float(accuracy_score(ys, preds)),
        f1_macro=float(f1_score(ys, preds, average='macro')),
        f1_weighted=float(f1_score(ys, preds, average='weighted')),
        auc=float(roc_auc_score(ys, probs[:, 1])),
        confusion_matrix=confusion_matrix(ys, preds).tolist(),
    )


def train_model(backbone, df_train, df_val, epochs, backbone_lr, head_lr,
                batch_size=None, patience=4, device=None, num_workers=8,
                log_prefix=''):
    cfg = BACKBONES[backbone]
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = batch_size or cfg['batch_size']
    seed_everything()

    train_ds = FundusSexDataset(df_train, IMAGES_PATH, train_transform(backbone))
    val_ds = FundusSexDataset(df_val, IMAGES_PATH, eval_transform(backbone))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = build_model(backbone).to(device)
    counts = np.bincount(df_train[LABEL].astype(int).values, minlength=2)
    class_weights = torch.tensor(len(df_train) / (2.0 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        param_groups(model, backbone, backbone_lr, head_lr, cfg['layer_decay']))
    steps_per_epoch = len(train_dl)
    warmup = steps_per_epoch  # 1 epoch linear warmup
    total = epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best = {'f1_macro': -1, 'epoch': 0, 'state': None}
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_dl, desc=f'{log_prefix}{backbone} epoch {epoch}/{epochs}', leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = criterion(logits_of(model, x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        probs, ys = predict(model, val_dl, device)
        m = metrics_from(probs, ys)
        print(f'{log_prefix}[{backbone}] epoch {epoch}: train_loss={running / steps_per_epoch:.4f} '
              f'val_f1_macro={m["f1_macro"]:.4f} val_acc={m["accuracy"]:.4f} val_auc={m["auc"]:.4f}',
              flush=True)
        if m['f1_macro'] > best['f1_macro']:
            best.update(f1_macro=m['f1_macro'], epoch=epoch,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        val_metrics=m)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f'{log_prefix}early stopping at epoch {epoch} '
                      f'(best epoch {best["epoch"]}, val F1 {best["f1_macro"]:.4f})', flush=True)
                break

    model.load_state_dict(best['state'])
    return model, best


def evaluate_saved(backbone, df_eval, weights=None, device=None, batch_size=None, tag='test'):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = BACKBONES[backbone]
    model = load_finetuned(backbone, weights, device)
    ds = FundusSexDataset(df_eval, IMAGES_PATH, eval_transform(backbone))
    dl = DataLoader(ds, batch_size=batch_size or cfg['batch_size'], shuffle=False,
                    num_workers=8, pin_memory=True)
    probs, ys = predict(model, dl, device)
    m = metrics_from(probs, ys)
    print(f'[{backbone}] {tag}: acc={m["accuracy"]:.4f} f1_macro={m["f1_macro"]:.4f} '
          f'auc={m["auc"]:.4f}')
    print(classification_report(ys, probs.argmax(1), target_names=['Female (0)', 'Male (1)']))
    return m, probs, ys


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backbone', choices=list(BACKBONES), default='retfound_dinov2')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--backbone-lr', type=float, default=None)
    ap.add_argument('--head-lr', type=float, default=None)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--sweep', action='store_true',
                    help='select backbone LR on the val split with short runs first')
    ap.add_argument('--sweep-lrs', default='5e-6,1e-5,3e-5')
    ap.add_argument('--sweep-epochs', type=int, default=3)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--weights', default=None)
    ap.add_argument('--make-split-only', action='store_true')
    args = ap.parse_args()

    labels_df = pd.read_csv(LABELS_CSV_PATH)
    labels_df = attach_split(labels_df)
    if args.make_split_only:
        print(labels_df.groupby('split')[LABEL].agg(['count', 'mean']))
        return

    cfg = BACKBONES[args.backbone]
    df_train = labels_df[labels_df.split == 'train']
    df_val = labels_df[labels_df.split == 'val']
    df_test = labels_df[labels_df.split == 'test']
    print(f'images: train={len(df_train)} val={len(df_val)} test={len(df_test)}')

    if args.eval_only:
        m, probs, ys = evaluate_saved(args.backbone, df_test, args.weights)
        out = os.path.splitext(args.weights or cfg['weights'])[0] + '_test_metrics.json'
        with open(out, 'w') as f:
            json.dump(m, f, indent=2)
        print(f'wrote {out}')
        return

    backbone_lr = args.backbone_lr or cfg['backbone_lr']
    head_lr = args.head_lr or cfg['head_lr']

    if args.sweep:
        results = {}
        for lr in [float(s) for s in args.sweep_lrs.split(',')]:
            print(f'--- sweep: backbone_lr={lr} ---', flush=True)
            _, best = train_model(args.backbone, df_train, df_val,
                                  epochs=args.sweep_epochs, backbone_lr=lr,
                                  head_lr=head_lr, batch_size=args.batch_size,
                                  patience=args.sweep_epochs, log_prefix='[sweep] ')
            results[lr] = best['f1_macro']
        backbone_lr = max(results, key=results.get)
        print(f'sweep results (val F1 macro): {results}')
        print(f'selected backbone_lr={backbone_lr}')

    model, best = train_model(args.backbone, df_train, df_val, epochs=args.epochs,
                              backbone_lr=backbone_lr, head_lr=head_lr,
                              batch_size=args.batch_size, patience=args.patience)

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = args.weights or cfg['weights']
    torch.save(model.state_dict(), out_path)
    meta = dict(backbone=args.backbone, backbone_lr=backbone_lr, head_lr=head_lr,
                epochs_run=best['epoch'], val_metrics=best.get('val_metrics'),
                batch_size=args.batch_size or cfg['batch_size'], seed=SEED,
                split_csv=SPLIT_CSV)
    with open(os.path.splitext(out_path)[0] + '_train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'saved best model (epoch {best["epoch"]}, val F1 {best["f1_macro"]:.4f}) -> {out_path}')

    m, _, _ = evaluate_saved(args.backbone, df_test, out_path)
    with open(os.path.splitext(out_path)[0] + '_test_metrics.json', 'w') as f:
        json.dump(m, f, indent=2)


if __name__ == '__main__':
    main()
