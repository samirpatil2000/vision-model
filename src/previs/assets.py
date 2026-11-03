"""
Asset bank: character references, location plates, cast sheets and state-versioned anchors.

Every generated asset goes through engine.generate and is cached by content: a manifest
(assets/manifest.json) maps a hash of (kind, prompt, size, references, sampler settings) to the
file, so an unchanged request reuses the existing image and a changed one gets the next _v<N>.
Cast sheets are composed in Python (no engine call, no text) from the cached references.
People get a front-view full-body portrait plus a face crop; quadrupeds (Character.body) a
landscape side view of the whole animal, whose "face" file is a square thumbnail of it.

Layout (run directory):
  assets/cast/<ch_id>_full_v<N>.png   assets/cast/<ch_id>_face_v<N>.png
  assets/plates/<lo_id>_v<N>.png      assets/props/<pr_id>_v<N>.png
  assets/sheets/<kf_id>_cast.png      assets/manifest.json
"""
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from src import config
from src.config import GenerationDefaults, Tier
from src.engine.base import EngineClient, GenerationRequest
from src.previs import seeds
from src.previs.conditioning import KEY_LIGHT_PHRASES, TIME_OF_DAY_PHRASES, flag_phrase
from src.previs.schema import Character, Location, Prop

Entity = Union[Character, Prop]

SHEET_BG = (214, 214, 214)                 # light grey behind references and sheets
CHARACTER_SIZES: Dict[str, Tuple[int, int]] = {"draft": (512, 768), "final": (576, 864), "hero": (640, 960)}
QUADRUPED_SIZES: Dict[str, Tuple[int, int]] = {"draft": (768, 512), "final": (864, 576), "hero": (960, 640)}
PROP_SIZES: Dict[str, Tuple[int, int]] = {"draft": (512, 512), "final": (576, 576), "hero": (640, 640)}
FACE_FRACTION = 0.22                       # face crop = top 22% of the full-body reference
FACE_SIZE = 256
SHEET_FACE_FRACTION = 0.4                  # face tile height relative to the full body in a sheet
SHEET_GAP_FRACTION = 0.04
TRIM_TOLERANCE = 18                        # per-channel difference from the corner colour that counts as content
TRIM_MARGIN_FRACTION = 0.06
MAX_APPROVAL_TRIES = 3


def _round32(value: float) -> int:
    return max(32, int(round(value / 32.0)) * 32)


def _content_hash(payload: Dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def face_crop(full_body: Image.Image, size: int = FACE_SIZE) -> Image.Image:
    """Square crop centred horizontally in the top FACE_FRACTION band, scaled to `size`."""
    w, h = full_body.size
    side = max(1, min(w, int(round(h * FACE_FRACTION))))
    left = (w - side) // 2
    return full_body.crop((left, 0, left + side, side)).resize((size, size), Image.Resampling.LANCZOS)


def trim_sides(img: Image.Image) -> Image.Image:
    """Crop empty background columns left and right (full height kept), with a small margin."""
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    corner = arr[0, 0]
    content = np.any(np.abs(arr - corner) > TRIM_TOLERANCE, axis=2)
    cols = np.where(content.any(axis=0))[0]
    if cols.size == 0 or cols[-1] - cols[0] < img.width * 0.1:
        return img
    margin = int(img.width * TRIM_MARGIN_FRACTION)
    return img.crop((max(0, cols[0] - margin), 0, min(img.width, cols[-1] + 1 + margin), img.height))


def letterbox(img: Image.Image, size: Tuple[int, int], fill: Tuple[int, int, int] = SHEET_BG) -> Image.Image:
    """Scale `img` to fit inside `size` (aspect kept) and centre it on a `fill` canvas."""
    w, h = size
    scale = min(w / img.width, h / img.height)
    fitted = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w, h), fill)
    canvas.paste(fitted, ((w - fitted.width) // 2, (h - fitted.height) // 2))
    return canvas


class AssetBank:
    """Generates (through the engine) and caches the reference images a run needs."""

    def __init__(self, run_dir: Union[str, Path], engine: EngineClient, tier: Tier, style: str, project: str,
                 defaults: Optional[GenerationDefaults] = None):
        self.run_dir = Path(run_dir)
        self.engine = engine
        self.tier = tier
        self.style = style
        self.project = project
        self.defaults = defaults or GenerationDefaults()
        self.root = self.run_dir / "assets"
        self.manifest_path = self.root / "manifest.json"
        self.manifest: Dict[str, Dict] = (
            json.loads(self.manifest_path.read_text()) if self.manifest_path.is_file() else {}
        )
        self._entities: Dict[str, Entity] = {}
        self.references: Dict[str, Tuple[str, str]] = {}   # character id -> user-supplied (full, face)

    # ------------------------------------------------------------ hooks
    def approve(self, path: str) -> bool:
        """Accept a freshly generated asset. Override for checkpointed review; False forces a retake."""
        return True

    # ------------------------------------------------------------ public API
    def ensure_character(self, ch: Character) -> Tuple[str, str]:
        """(full_body_path, face_path): front-view full-body reference and its face crop (for a quadruped:
        a side view of the whole animal and a square thumbnail of it)."""
        self._entities[ch.id] = ch
        if ch.id in self.references:
            return self.references[ch.id]
        quadruped = ch.body == "quadruped"
        wardrobe = ch.wardrobe.strip().rstrip(".")
        if quadruped:
            prompt = (f"Full-body reference of one animal, {ch.appearance.strip().rstrip('.')}, "
                      + (f"wearing {wardrobe}, " if wardrobe else "")
                      + f"side view in profile, standing on all four legs, whole body from nose to tail in frame, "
                      f"plain light grey background, even studio light, {self.style}")
        else:
            prompt = (f"Full-body character reference of one person, {ch.appearance.strip().rstrip('.')}, "
                      f"wearing {wardrobe}, neutral standing pose, front view, arms relaxed, "
                      f"whole figure from head to feet in frame, plain light grey background, even studio light, "
                      f"{self.style}")
        full = self._generate("character", ch.id, self.root / "cast", f"{ch.id}_full", prompt,
                              self.character_size(ch), extra={"body": ch.body})
        face_path = self._face_path(full)
        if not face_path.is_file():
            self._write_face(full, face_path, quadruped)
        return full, str(face_path)

    def use_reference(self, ch: Character, image_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Use a user-supplied picture as the character's identity reference instead of generating one.
        It is letterboxed onto the sheet background at the character size; the face crop is derived from it.
        """
        self._entities[ch.id] = ch
        src = Path(image_path).expanduser()
        if not src.is_file():
            raise FileNotFoundError(f"Cast reference for {ch.id} not found: {src}")
        folder = self.root / "cast"
        folder.mkdir(parents=True, exist_ok=True)
        full = folder / f"{ch.id}_full_ref.png"
        with Image.open(src) as img:
            letterbox(img.convert("RGB"), self.character_size(ch)).save(full)
        face_path = self._face_path(str(full))
        self._write_face(str(full), face_path, ch.body == "quadruped")
        self.manifest[f"reference:{ch.id}"] = {"kind": "reference", "entity_id": ch.id, "path": str(full),
                                               "source": str(src)}
        self._save_manifest()
        self.references[ch.id] = (str(full), str(face_path))
        return self.references[ch.id]

    def ensure_prop(self, prop: Prop) -> str:
        """Reference image of a prop in its initial (unflagged) state."""
        self._entities[prop.id] = prop
        prompt = (f"Reference photo of a single {prop.name}, whole object centred in frame, plain light grey "
                  f"background, even studio light, {self.style}")
        side = PROP_SIZES.get(self.tier.name, (512, 512))
        return self._generate("prop", prop.id, self.root / "props", prop.id, prompt, side)

    def ensure_plate(self, location: Location, time_of_day: str) -> str:
        """Empty establishing view of a location (no people) at the tier's frame size."""
        light = ", ".join(p for p in (TIME_OF_DAY_PHRASES.get(time_of_day, ""),
                                      KEY_LIGHT_PHRASES.get(location.key_light, "")) if p)
        prompt = (f"Empty establishing view of the {location.name}, {location.description.strip().rstrip('.')}, "
                  f"{light}, no people, nobody in frame, {self.style}")
        return self._generate("plate", location.id, self.root / "plates", location.id, prompt,
                              (self.tier.width, self.tier.height))

    def cast_sheet(self, character_ids: Sequence[str], size: Tuple[int, int], name: Optional[str] = None) -> str:
        """
        Full body + face of each character side by side on light grey, letterboxed into `size`
        (no text). Characters must have been ensured first. Saved as assets/sheets/<name>.png.
        """
        if not character_ids:
            raise ValueError("cast_sheet needs at least one character id")
        sources = [self._character_files(cid) for cid in character_ids]
        if name is None:
            name = "cast_" + _content_hash({"ids": list(character_ids), "sources": sources, "size": list(size)})[:10]
        path = self.root / "sheets" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1. One cell per character: trimmed full body with the face tile beside it, top-aligned
        #    (a quadruped's side view is the whole cell: its "face" file is only a thumbnail of it).
        cells = []
        for cid, (full_path, face_path) in zip(character_ids, sources):
            full = trim_sides(Image.open(full_path).convert("RGB"))
            if self._is_quadruped(cid):
                cells.append(full)
                continue
            side = max(1, int(full.height * SHEET_FACE_FRACTION))
            face = Image.open(face_path).convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
            gap = int(full.height * SHEET_GAP_FRACTION)
            cell = Image.new("RGB", (full.width + gap + side, full.height), SHEET_BG)
            cell.paste(full, (0, 0))
            cell.paste(face, (full.width + gap, 0))
            cells.append(cell)
        # 2. Cells at a common height, side by side, letterboxed into the sheet size.
        height = max(c.height for c in cells)
        cells = [c if c.height == height else c.resize((int(c.width * height / c.height), height),
                                                       Image.Resampling.LANCZOS) for c in cells]
        gap = int(height * SHEET_GAP_FRACTION * 2)
        strip = Image.new("RGB", (sum(c.width for c in cells) + gap * (len(cells) - 1), height), SHEET_BG)
        x = 0
        for cell in cells:
            strip.paste(cell, (x, 0))
            x += cell.width + gap
        letterbox(strip, tuple(size)).save(path)
        return str(path)

    def version_for_flags(self, entity_id: str, flags: Sequence[str], entity: Optional[Entity] = None) -> str:
        """
        Anchor image of an entity in a damaged/filled state, edited from its current version:
        "same <thing>, now <flag phrase>". Cached by (entity, sorted flags).
        """
        if entity is not None:
            self._entities[entity.id] = entity
        if entity_id not in self._entities:
            raise KeyError(f"{entity_id}: call ensure_character/ensure_prop first or pass `entity`")
        thing = self._entities[entity_id]
        base = self.ensure_character(thing)[0] if isinstance(thing, Character) else self.ensure_prop(thing)
        wanted = sorted(set(flags))
        if not wanted or not flag_phrase(wanted):
            return base

        # 1. Cached (entity, flags) version, else the current one: the cached state with the most flags
        #    that are all still present (flags only accumulate, so edits chain from it).
        states = [r for r in self.manifest.values()
                  if r.get("kind") == "state" and r.get("entity_id") == entity_id and r.get("base") == base
                  and Path(r["path"]).is_file()]
        for record in states:
            if sorted(record.get("flags", [])) == wanted:
                return record["path"]
        current, current_flags = base, []
        for record in states:
            have = record.get("flags", [])
            if set(have) < set(wanted) and len(have) > len(current_flags):
                current, current_flags = record["path"], have
        new_flags = [f for f in wanted if f not in current_flags]

        # 2. Edit it with only the new flags.
        if isinstance(thing, Character):
            label = "animal" if thing.body == "quadruped" else "person"
        else:
            label = thing.name
        prompt = (f"Same {label} as in the first image, now {flag_phrase(new_flags)}. Plain light grey background, "
                  f"even studio light, {self.style}")
        with Image.open(current) as img:
            size = (_round32(img.width), _round32(img.height))
        folder = self.root / ("cast" if isinstance(thing, Character) else "props")
        stem = f"{entity_id}_full" if isinstance(thing, Character) else entity_id
        return self._generate("state", entity_id, folder, stem, prompt, size, ref_images=[current],
                              extra={"flags": wanted, "base": base})

    def character_size(self, ch: Optional[Character] = None) -> Tuple[int, int]:
        """Portrait for people, landscape for a quadruped's side view."""
        if ch is not None and ch.body == "quadruped":
            return QUADRUPED_SIZES.get(self.tier.name, (768, 512))
        return CHARACTER_SIZES.get(self.tier.name, (512, 768))

    # ------------------------------------------------------------ internals
    @staticmethod
    def _face_path(full_path: str) -> Path:
        path = Path(full_path)
        return path.with_name(path.name.replace("_full_v", "_face_v"))

    @staticmethod
    def _write_face(full_path: str, face_path: Path, quadruped: bool) -> None:
        with Image.open(full_path) as img:
            full = img.convert("RGB")
        face = letterbox(trim_sides(full), (FACE_SIZE, FACE_SIZE)) if quadruped else face_crop(full)
        face.save(face_path)

    def _character_records(self, ch_id: str):
        return [r for r in self.manifest.values()
                if r.get("kind") == "character" and r.get("entity_id") == ch_id and Path(r["path"]).is_file()]

    def _is_quadruped(self, ch_id: str) -> bool:
        """Body plan of a character: this session's Character, else the newest manifest record."""
        entity = self._entities.get(ch_id)
        if isinstance(entity, Character):
            return entity.body == "quadruped"
        records = self._character_records(ch_id)
        newest = max(records, key=lambda r: r.get("created_order", 0)) if records else {}
        return newest.get("body") == "quadruped"

    def _character_files(self, ch_id: str) -> Tuple[str, str]:
        """(full, face) of a character's base reference: this session's, else the newest in the manifest."""
        if isinstance(self._entities.get(ch_id), Character):
            return self.ensure_character(self._entities[ch_id])
        records = self._character_records(ch_id)
        if not records:
            raise KeyError(f"{ch_id}: no character reference; call ensure_character first")
        newest = max(records, key=lambda r: r.get("created_order", 0))
        full = newest["path"]
        face = self._face_path(full)
        if not face.is_file():
            self._write_face(full, face, newest.get("body") == "quadruped")
        return full, str(face)

    @staticmethod
    def _next_version(folder: Path, stem: str) -> int:
        pattern = re.compile(rf"^{re.escape(stem)}_v(\d+)\.png$")
        versions = [int(m.group(1)) for p in folder.glob(f"{stem}_v*.png") if (m := pattern.match(p.name))]
        return max(versions, default=0) + 1

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def _generate(self, kind: str, entity_id: str, folder: Path, stem: str, prompt: str, size: Tuple[int, int],
                  ref_images: Sequence[str] = (), extra: Optional[Dict] = None) -> str:
        """Generate one asset through the engine unless an identical request is already cached."""
        d = self.defaults
        sigmas = list(config.VIGGLE_SIGMAS) if d.use_custom_sigmas else None
        payload = {"kind": kind, "entity_id": entity_id, "prompt": prompt, "size": list(size),
                   "refs": list(ref_images), "steps": d.steps, "cfg": d.cfg_scale, "sampler": d.sampler,
                   "scheduler": d.scheduler, "sigmas": sigmas, "project": self.project}
        key = f"{kind}:{entity_id}:{_content_hash(payload)}"
        cached = self.manifest.get(key)
        if cached and Path(cached["path"]).is_file():
            return cached["path"]

        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}_v{self._next_version(folder, stem)}.png"
        result = None
        for attempt in range(MAX_APPROVAL_TRIES):
            request = GenerationRequest(
                prompt=prompt, output_path=str(path), width=size[0], height=size[1],
                seed=seeds.seed_for(self.project, "assets", stem, attempt),
                steps=d.steps, cfg_scale=d.cfg_scale, sampler=d.sampler, scheduler=d.scheduler, sigmas=sigmas,
                ref_images=list(ref_images), strength_as_noise_level=False, tag=f"asset_{stem}",
            )
            result = self.engine.generate(request)
            if Path(result.image_path).resolve() != path.resolve():
                shutil.copyfile(result.image_path, path)
            if self.approve(str(path)):
                break
        else:
            raise RuntimeError(f"Asset {stem} was rejected {MAX_APPROVAL_TRIES} times")

        self.manifest[key] = {
            "kind": kind, "entity_id": entity_id, "path": str(path), "prompt": prompt, "size": list(size),
            "seed": request.seed, "seconds": result.seconds, "created_order": len(self.manifest) + 1,
            **(extra or {}),
        }
        self._save_manifest()
        return str(path)
