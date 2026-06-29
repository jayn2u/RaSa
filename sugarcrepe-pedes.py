from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from models.model_person_search import ALBEF
from models.tokenization_bert import BertTokenizer
from models.vit import interpolate_pos_embed

def require_config_path(env_key: str, config_prefix: str) -> Path:
    override = os.environ.get(env_key, "").strip()
    if not override:
        supported = ", ".join(
            f"configs/{config_prefix}_{name}.yaml"
            for name in ("cuhk_pedes", "icfg_pedes", "rstp_reid")
        )
        raise ValueError(f"{env_key} is required. Set it to one of: {supported}")
    path = Path(override)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path
METHOD_REPLACE = "replace"
METHOD_SWAP = "swap"
METHOD_ADD = "add"
METHOD_OTHER = "other"

TYPE_OBJECT = "obj"
TYPE_ATTRIBUTE = "att"
TYPE_RELATION = "rel"
TYPE_OTHER = "other"

_TYPE_FULL_NAME = {
    TYPE_OBJECT: "object",
    TYPE_ATTRIBUTE: "attribute",
    TYPE_RELATION: "relation",
}

DATASET_SPECS = {
    "cuhk-pedes": ("CUHK-PEDES", "reid_raw.json", "file_path"),
    "rstpreid": ("RSTPReid", "data_captions.json", "img_path"),
    "icfg-pedes": ("ICFG-PEDES", "ICFG-PEDES.json", "file_path"),
}

_COLOR_WORDS = frozenset(
    {
        "black", "white", "red", "blue", "green", "yellow", "gray", "grey",
        "brown", "pink", "purple", "orange", "beige", "navy", "tan", "gold",
        "silver", "bright", "dark", "light", "neon", "khaki", "maroon", "violet",
        "cyan", "teal", "cream", "ivory",
    }
)
_PATTERN_WORDS = frozenset(
    {
        "striped", "plaid", "checkered", "checked", "floral", "plain",
        "patterned", "solid", "polka", "dotted", "camouflage", "camo", "denim",
        "leather", "cotton", "wool", "woolen", "knit", "knitted", "fur", "furry",
        "sleeveless", "long", "short", "sleeved", "tight", "loose", "baggy",
        "skinny", "slim", "hooded", "collared", "buttoned", "ripped", "torn",
        "shiny", "matte", "fluffy", "thick", "thin", "wide", "narrow", "curly",
        "straight", "wavy", "bald", "sleeve", "sleeves",
    }
)
_ATTRIBUTE_WORDS = _COLOR_WORDS | _PATTERN_WORDS
_GARMENT_WORDS = frozenset(
    {
        "shirt", "jacket", "coat", "pants", "jeans", "trousers", "shorts",
        "skirt", "dress", "shoes", "sneakers", "boots", "hat", "cap", "hoodie",
        "sweatshirt", "sweater", "cardigan", "blazer", "suit", "robe", "gown",
        "leggings", "tights", "tracksuit", "jersey", "polo", "blouse", "tunic",
        "overalls", "jumpsuit", "vest", "backpack", "bag", "top", "t-shirt",
        "tee", "sandals", "heels", "loafers", "slippers", "gloves", "scarf",
        "tie", "belt", "helmet", "mask", "glasses", "sunglasses", "watch",
        "bracelet", "necklace", "earrings", "umbrella", "phone", "cellphone",
        "handbag", "purse", "suitcase", "luggage", "briefcase", "satchel",
        "wallet", "book", "books", "bottle", "cup", "box", "basket", "stroller",
        "bicycle", "bike", "cart", "headphones", "earphones", "headband",
        "ponytail", "bun", "hair", "beard", "mustache",
    }
)
_OBJECT_WORDS = _GARMENT_WORDS
_RELATION_WORDS = frozenset(
    {
        "left", "right", "behind", "front", "above", "below", "under", "over",
        "beneath", "atop", "onto", "into", "beside", "next", "near", "top",
        "bottom", "side", "back", "ahead", "across", "along", "around",
        "between", "upon", "holding", "holds", "hold", "carrying", "carries",
        "carry", "carried", "riding", "rides", "ride", "pushing", "pushes",
        "push", "pulling", "pulls", "standing", "stands", "stand", "sitting",
        "sits", "sit", "walking", "walks", "walk", "running", "runs", "run",
        "leaning", "leans", "lean", "facing", "faces", "crossing", "crosses",
        "cross", "climbing", "kneeling", "bending", "crouching", "jumping",
        "lying",
    }
)
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "with", "is", "are", "was",
        "were", "be", "been", "being", "in", "on", "at", "by", "as", "for",
        "from", "this", "that", "these", "those", "he", "she", "it", "they",
        "his", "her", "their", "its", "him", "them", "has", "have", "had",
        "who", "which", "while", "also", "both", "very", "quite", "some",
        "there", "appears", "seems", "seem", "appear", "wearing", "wears",
        "wear", "worn", "dressed", "person", "man", "woman", "men", "women",
        "pedestrian", "guy", "lady", "girl", "boy", "individual", "someone",
        "people", "human", "subject",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)

CONFIG_DEFAULTS = {
    "env_file": "",
    "dataset": "",
    "dataset_root": "",
    "negative_pedestrians_root": "/mnt/data/negative-pedestrians/outputs",
    "negative_model": "gemma4:e4b",
    "negative_prompt": "tripletclip_reid.yaml",
    "negative_annotation": "auto",
    "ras_config": "",
    "checkpoint": "",
    "text_encoder": "bert-base-uncased",
    "test_split": "test",
    "batch_size": 128,
    "num_workers": 4,
    "device": "auto",
    "seed": 42,
    "max_probes": 0,
    "output_json": "",
    "no_amp": False,
}


@dataclass(frozen=True)
class CompositionalProbe:
    image_path: Path
    person_id: int
    positive_caption: str
    negative_caption: str
    caption_index: int
    perturbation: str
    method: str
    change_type: str


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower()
    if normalized not in DATASET_SPECS:
        supported = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unsupported dataset {dataset!r}. Expected one of: {supported}.")
    return normalized


def _read_env_key(env_file: str | Path, key: str) -> str | None:
    value = os.environ.get(key)
    if value is not None:
        value = value.strip().strip('"').strip("'")
        return value or None
    env_path = Path(env_file)
    if not env_path.exists():
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_key, env_value = line.split("=", 1)
        if env_key.strip() != key:
            continue
        env_value = env_value.strip().strip('"').strip("'")
        return env_value or None
    return None


def resolve_dataset_root(dataset_root: str, env_file: str) -> Path:
    if dataset_root.strip():
        path = Path(dataset_root).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"dataset_root does not exist: {path}")
    if env_file.strip():
        value = _read_env_key(env_file, "DATASET_ROOT")
        if value:
            path = Path(value).expanduser()
            if path.exists():
                return path
    for candidate in (Path("/mnt/data/lab_datasets"), Path("/data/jayn2u/lab_datasets")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve dataset root.")


def resolve_negative_pedestrians_root(config: SimpleNamespace) -> Path:
    if config.env_file.strip():
        value = _read_env_key(config.env_file, "NEGATIVE_REID_DATASET_PATH")
        if value:
            return Path(value).expanduser()
    return Path(config.negative_pedestrians_root).expanduser()


def resolve_negative_annotation_path(config: SimpleNamespace, dataset: str) -> Path:
    normalized = normalize_dataset_name(dataset)
    dataset_dir = resolve_negative_pedestrians_root(config) / normalized
    requested = str(config.negative_annotation).strip()
    if requested and requested.lower() != "auto":
        direct = Path(requested).expanduser()
        candidates = [direct]
        if not direct.is_absolute():
            candidates.append(dataset_dir / requested)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"Could not find negative annotation {requested!r} under {dataset_dir}.")
    if config.env_file.strip():
        model_tag = _read_env_key(config.env_file, "NEGATIVE_MODEL") or config.negative_model
        prompt = _read_env_key(config.env_file, "NEGATIVE_PROMPT") or config.negative_prompt
    else:
        model_tag = config.negative_model
        prompt = config.negative_prompt
    prompt_stem = Path(prompt).stem
    _, annotation_file, _ = DATASET_SPECS[normalized]
    negative_file = f"{Path(annotation_file).stem}_negative_{model_tag}_{prompt_stem}.json"
    candidate = dataset_dir / negative_file
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find negative captions file: {candidate}")


def tokenize_caption(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _content_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS
    ]


def _resolve_change_type(tokens: set[str]) -> str:
    has_relation = any(token in _RELATION_WORDS for token in tokens)
    has_object = any(token in _OBJECT_WORDS for token in tokens)
    has_attribute = any(token in _ATTRIBUTE_WORDS for token in tokens)
    if has_relation:
        return TYPE_RELATION
    if has_object:
        return TYPE_OBJECT
    if has_attribute:
        return TYPE_ATTRIBUTE
    return TYPE_OTHER


def _resolve_add_change_type(tokens: set[str]) -> str:
    if any(token in _OBJECT_WORDS for token in tokens):
        return TYPE_OBJECT
    if any(token in _ATTRIBUTE_WORDS for token in tokens):
        return TYPE_ATTRIBUTE
    return TYPE_OTHER


def classify_perturbation(
    positive_caption: str, negative_caption: str
) -> tuple[str, str]:
    positive_tokens = _content_tokens(positive_caption)
    negative_tokens = _content_tokens(negative_caption)
    if not positive_tokens or not negative_tokens:
        return METHOD_OTHER, TYPE_OTHER
    positive_counts = Counter(positive_tokens)
    negative_counts = Counter(negative_tokens)
    if positive_counts == negative_counts:
        if positive_tokens == negative_tokens:
            return METHOD_OTHER, TYPE_OTHER
        moved = {
            token
            for left, right in zip(positive_tokens, negative_tokens)
            if left != right
            for token in (left, right)
        }
        return METHOD_SWAP, _resolve_change_type(moved)
    added = set(negative_counts - positive_counts)
    removed = set(positive_counts - negative_counts)
    if added and not removed:
        return METHOD_ADD, _resolve_add_change_type(added)
    if removed and not added:
        return METHOD_ADD, _resolve_add_change_type(removed)
    if added == removed:
        return METHOD_SWAP, _resolve_change_type(added)
    return METHOD_REPLACE, _resolve_change_type(added | removed)


def perturbation_tag(method: str, change_type: str) -> str:
    if method == METHOD_OTHER:
        return METHOD_OTHER
    return f"{method}_{change_type}"


def load_compositional_probes(
    *,
    annotation_path: Path,
    dataset_root: Path,
    dataset: str,
    split: str,
    max_probes: int = 0,
) -> list[CompositionalProbe]:
    directory, _, image_key = DATASET_SPECS[normalize_dataset_name(dataset)]
    image_root = dataset_root / directory / "imgs"
    raw_records = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise TypeError(f"Expected a list in {annotation_path}.")
    probes: list[CompositionalProbe] = []
    for raw in raw_records:
        if not isinstance(raw, dict) or str(raw.get("split", "")) != split:
            continue
        pos_caps = [str(caption) for caption in raw.get("captions", []) if str(caption).strip()]
        neg_caps = [
            str(caption)
            for caption in (raw.get("negative_captions") or [])
            if str(caption).strip()
        ]
        if not pos_caps or not neg_caps:
            continue
        image_path = image_root / str(raw[image_key])
        if not image_path.exists():
            continue
        person_id = int(raw["id"])
        pair_count = min(len(pos_caps), len(neg_caps))
        for index in range(pair_count):
            positive_caption = pos_caps[index]
            negative_caption = neg_caps[index]
            method, change_type = classify_perturbation(
                positive_caption, negative_caption
            )
            probes.append(
                CompositionalProbe(
                    image_path=image_path,
                    person_id=person_id,
                    positive_caption=positive_caption,
                    negative_caption=negative_caption,
                    caption_index=index,
                    perturbation=perturbation_tag(method, change_type),
                    method=method,
                    change_type=change_type,
                )
            )
            if max_probes and len(probes) >= max_probes:
                return probes
    return probes


def collect_probe_vocabulary(probes: list[CompositionalProbe]) -> tuple[list[Path], list[str]]:
    image_paths = sorted({probe.image_path for probe in probes})
    captions: set[str] = set()
    for probe in probes:
        captions.add(probe.positive_caption)
        captions.add(probe.negative_caption)
    return image_paths, sorted(captions)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() * right.float()).sum().item())


def _bucket_metrics(
    correct_flags: list[bool], margins: list[float]
) -> dict[str, float]:
    total = len(correct_flags)
    if total == 0:
        return {
            "count": 0.0,
            "correct": 0.0,
            "discrimination_rate": 0.0,
            "mean_margin": 0.0,
        }
    correct = sum(1 for flag in correct_flags if flag)
    return {
        "count": float(total),
        "correct": float(correct),
        "discrimination_rate": correct / total,
        "mean_margin": sum(margins) / total,
    }


def evaluate_sugarcrepe_probes(
    probes: list[CompositionalProbe],
    *,
    image_features: dict[Path, torch.Tensor],
    text_features: dict[str, torch.Tensor],
) -> dict[str, object]:
    overall_correct: list[bool] = []
    overall_margins: list[float] = []
    pos_sims: list[float] = []
    neg_sims: list[float] = []
    fine_correct: dict[str, list[bool]] = defaultdict(list)
    fine_margins: dict[str, list[float]] = defaultdict(list)
    method_correct: dict[str, list[bool]] = defaultdict(list)
    method_margins: dict[str, list[float]] = defaultdict(list)
    type_correct: dict[str, list[bool]] = defaultdict(list)
    type_margins: dict[str, list[float]] = defaultdict(list)
    for probe in probes:
        image_feat = image_features[probe.image_path]
        pos_sim = _cosine(image_feat, text_features[probe.positive_caption])
        neg_sim = _cosine(image_feat, text_features[probe.negative_caption])
        correct = pos_sim > neg_sim
        margin = pos_sim - neg_sim
        overall_correct.append(correct)
        overall_margins.append(margin)
        pos_sims.append(pos_sim)
        neg_sims.append(neg_sim)
        fine_correct[probe.perturbation].append(correct)
        fine_margins[probe.perturbation].append(margin)
        method_correct[probe.method].append(correct)
        method_margins[probe.method].append(margin)
        if probe.change_type in _TYPE_FULL_NAME:
            type_key = _TYPE_FULL_NAME[probe.change_type]
            type_correct[type_key].append(correct)
            type_margins[type_key].append(margin)
    overall = {
        **_bucket_metrics(overall_correct, overall_margins),
        "mean_positive_similarity": (
            sum(pos_sims) / len(pos_sims) if pos_sims else 0.0
        ),
        "mean_negative_similarity": (
            sum(neg_sims) / len(neg_sims) if neg_sims else 0.0
        ),
    }
    return {
        "benchmark": "sugarcrepe",
        "description": (
            "Image-conditioned hard-caption discrimination. "
            "Given image I and captions (C+, C-), score 1 when sim(I, C+) > sim(I, C-). "
            "Perturbation method (replace/swap/add) and change type "
            "(object/attribute/relation) are inferred from the positive/negative "
            "token diff, mirroring the SugarCrepe taxonomy."
        ),
        "random_chance": 0.5,
        "overall": overall,
        "by_perturbation": {
            tag: _bucket_metrics(fine_correct[tag], fine_margins[tag])
            for tag in sorted(fine_correct)
        },
        "by_method": {
            method: _bucket_metrics(method_correct[method], method_margins[method])
            for method in sorted(method_correct)
        },
        "by_type": {
            type_key: _bucket_metrics(type_correct[type_key], type_margins[type_key])
            for type_key in sorted(type_correct)
        },
    }


def load_yaml_config(config_path: Path) -> SimpleNamespace:
    if len(sys.argv) > 1:
        raise ValueError("CLI arguments are not supported. Edit the YAML config file instead.")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file {config_path} must contain a mapping.")
    unknown_keys = set(data) - set(CONFIG_DEFAULTS)
    if unknown_keys:
        raise ValueError(f"Unknown keys in config {config_path}: {sorted(unknown_keys)}")
    config = dict(CONFIG_DEFAULTS)
    config.update(data)
    missing = [
        key
        for key in ("dataset", "checkpoint", "ras_config")
        if not str(config.get(key, "")).strip()
    ]
    if missing:
        raise ValueError(f"Missing required keys in config {config_path}: {missing}")
    return SimpleNamespace(**config)


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    return torch.device(device_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_results_date_dir(run_time: datetime | None = None) -> Path:
    run_time = run_time or datetime.now()
    return Path("results") / run_time.strftime("%m-%d")


def output_model_tag(checkpoint: str) -> str:
    return Path(checkpoint).with_suffix("").as_posix().replace("/", "-").replace(":", "-")


def build_eval_transform(image_res: int):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose(
        [
            transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )


class _UniqueImageDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "index": index}


class _UniqueTextDataset(Dataset):
    def __init__(self, captions: list[str]) -> None:
        self.captions = captions

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, index: int) -> dict:
        return {"caption": self.captions[index], "index": index}


def _collate_images(batch: list[dict]) -> dict:
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


def _collate_texts(batch: list[dict]) -> dict:
    return {
        "captions": [item["caption"] for item in batch],
        "indices": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


@torch.no_grad()
def encode_unique_images(
    model,
    image_paths: list[Path],
    *,
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    dataset = _UniqueImageDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_images,
    )
    features = torch.empty((len(image_paths), 0), dtype=torch.float32)
    for batch in tqdm(loader, desc="encode probe images", dynamic_ncols=True):
        images = batch["images"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            image_feat = model.visual_encoder(images)
            batch_features = F.normalize(model.vision_proj(image_feat[:, 0, :]), dim=-1).cpu()
        if features.numel() == 0:
            features = torch.empty((len(image_paths), batch_features.shape[1]))
        features[batch["indices"]] = batch_features
    return features


@torch.no_grad()
def encode_unique_texts(
    model,
    tokenizer,
    captions: list[str],
    *,
    max_words: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    dataset = _UniqueTextDataset(captions)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_collate_texts,
    )
    features = torch.empty((len(captions), 0), dtype=torch.float32)
    for batch in tqdm(loader, desc="encode probe texts", dynamic_ncols=True):
        text_input = tokenizer(
            batch["captions"],
            padding="max_length",
            truncation=True,
            max_length=max_words,
            return_tensors="pt",
        ).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            text_output = model.text_encoder.bert(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                mode="text",
            )
            batch_features = F.normalize(
                model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            ).cpu()
        if features.numel() == 0:
            features = torch.empty((len(captions), batch_features.shape[1]))
        features[batch["indices"]] = batch_features
    return features


def load_rasa_model(
    *,
    ras_config_path: str,
    checkpoint_path: Path,
    text_encoder: str,
    device: torch.device,
):
    yaml_loader = YAML(typ="rt")
    with open(ras_config_path, "r", encoding="utf-8") as handle:
        ras_config = yaml_loader.load(handle)
    tokenizer = BertTokenizer.from_pretrained(text_encoder)
    model = ALBEF(config=ras_config, text_encoder=text_encoder, tokenizer=tokenizer)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    pos_embed_reshaped = interpolate_pos_embed(
        state_dict["visual_encoder.pos_embed"], model.visual_encoder
    )
    state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
    m_pos_embed_reshaped = interpolate_pos_embed(
        state_dict["visual_encoder_m.pos_embed"], model.visual_encoder_m
    )
    state_dict["visual_encoder_m.pos_embed"] = m_pos_embed_reshaped
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer, ras_config


def _print_metrics(metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    assert isinstance(overall, dict)
    print("SugarCrepe-style compositional discrimination")
    print(f"overall discrimination_rate: {overall['discrimination_rate']:.4f}")
    print(f"overall mean_margin: {overall['mean_margin']:.4f}")
    for group_key, title in (
        ("by_method", "by method (replace / swap / add)"),
        ("by_type", "by type (object / attribute / relation)"),
        ("by_perturbation", "by fine-grained category"),
    ):
        group = metrics[group_key]
        assert isinstance(group, dict)
        print(title)
        for tag, payload in group.items():
            assert isinstance(payload, dict)
            print(
                f"  {tag}: discrimination_rate={payload['discrimination_rate']:.4f} "
                f"(n={int(payload['count'])})"
            )


def main() -> None:
    config = load_yaml_config(require_config_path("SUGARCREPE_CONFIG", "sugarcrepe"))
    seed_everything(config.seed)
    device = get_device(config.device)
    dataset_name = normalize_dataset_name(config.dataset)
    dataset_root = resolve_dataset_root(config.dataset_root, config.env_file)
    annotation_path = resolve_negative_annotation_path(config, dataset_name)
    probes = load_compositional_probes(
        annotation_path=annotation_path,
        dataset_root=dataset_root,
        dataset=dataset_name,
        split=config.test_split,
        max_probes=config.max_probes,
    )
    if not probes:
        raise RuntimeError(
            f"No compositional probes found for split={config.test_split!r} in {annotation_path}."
        )
    image_paths, captions = collect_probe_vocabulary(probes)
    checkpoint_path = Path(config.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    model, tokenizer, ras_config = load_rasa_model(
        ras_config_path=config.ras_config,
        checkpoint_path=checkpoint_path,
        text_encoder=config.text_encoder,
        device=device,
    )
    transform = build_eval_transform(int(ras_config["image_res"]))
    use_amp = device.type == "cuda" and not config.no_amp
    image_features_tensor = encode_unique_images(
        model,
        image_paths,
        transform=transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    text_features_tensor = encode_unique_texts(
        model,
        tokenizer,
        captions,
        max_words=int(ras_config["max_words"]),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        use_amp=use_amp,
    )
    image_feature_map = {path: image_features_tensor[index] for index, path in enumerate(image_paths)}
    text_feature_map = {caption: text_features_tensor[index] for index, caption in enumerate(captions)}
    metrics = evaluate_sugarcrepe_probes(
        probes,
        image_features=image_feature_map,
        text_features=text_feature_map,
    )
    _print_metrics(metrics)
    run_time = datetime.now()
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    results_dir = build_results_date_dir(run_time)
    model_tag = output_model_tag(str(checkpoint_path))
    output = {
        "benchmark": "sugarcrepe",
        "model": "RaSa",
        "dataset": dataset_name,
        "domain": "person-reid-compositional-probe",
        "checkpoint": str(checkpoint_path),
        "embedding_protocol": "coarse_cls_projection_no_itm",
        "split": config.test_split,
        "negative_annotation": str(annotation_path),
        "probes": len(probes),
        "unique_images": len(image_paths),
        "unique_captions": len(captions),
        "metrics": metrics,
    }
    default_output_path = (
        results_dir / f"{dataset_name}_sugarcrepe_compositional_{model_tag}_{timestamp}.json"
    )
    output_path = Path(config.output_json) if config.output_json else default_output_path
    if not output_path.is_absolute() and config.output_json:
        output_path = results_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
