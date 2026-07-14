"""
train_kaggle.py
----------------
Bu dosyanin icerigi, Kaggle Notebook'a HUCRE HUCRE yapistirilmak icin
yazildi (kodu asagidaki # === HUCRE N === yorumlarina gore boluyorsun).

NEDEN LOKALDE DEGIL DE KAGGLE'DA?
- Kaggle'in GPU'lari (T4 x2 veya P100) 16GB VRAM veriyor; senin lokal
  RTX 3060 Laptop'taki 6GB'a gore batch=4'e sikismak yerine batch=16-24
  ile rahat egitim yapilabiliyor. Buyuk batch = daha kararli gradyanlar
  = genelde daha iyi/daha hizli yakinsama.

NEDEN ONCE VERIYI /kaggle/working'e KOPYALIYORUZ?
- Kaggle'da "Input" olarak eklenen veri setleri SALT OKUNUR
  (/kaggle/input). Ultralytics egitim sirasinda "labels.cache" gibi
  dosyalar yazmaya calisir; salt okunur klasorde bu patlar. Bu yuzden
  once yazilabilir alana (/kaggle/working) kopyaliyoruz.
"""

# === HUCRE 1: kutuphane kurulumu ===
# DIKKAT: "-U" (upgrade) BILEREK KULLANILMIYOR. "-U" ultralytics'i
# guncellerken beraberinde cok yeni bir PyTorch/CUDA surumu de
# getiriyor; bu yeni surum P100 gibi eski mimarili GPU'lari (Pascal,
# "sm_60") artik desteklemiyor ve "no kernel image" hatasi veriyor.
# Kaggle'in GPU'ya onceden kurdugu, o GPU ile test edilmis PyTorch'a
# dokunmadan sadece ultralytics'i kuruyoruz.
!pip install -q ultralytics

# === HUCRE 2: iki veri setini de bul ve kendi hazirladigimizi kopyala ===
import shutil
from pathlib import Path

# Artik notebook'a 2 dataset ekli oldugu icin (kendi hazirladigimiz
# "fire-smoke-merged" + Kaggle'daki hazir "smoke-fire-detection-yolo" yani
# D-Fire), data.yaml'a gore genel arama artik iki sonuc dondurur ve hangisi
# hangisi karisir. Bu yuzden dataset SLUG'ina (klasor adina) gore
# ayirt ediyoruz.
all_data_yamls = list(Path("/kaggle/input").rglob("data.yaml"))
if not all_data_yamls:
    raise FileNotFoundError(
        "/kaggle/input altinda data.yaml bulunamadi. Dataset'lerin notebook'a "
        "'Add Input' ile eklendiginden ve session'in yeniden baslatildigindan emin ol."
    )

def find_dataset_root(slug_substring: str) -> Path:
    for p in all_data_yamls:
        if slug_substring in str(p):
            return p.parent
    raise FileNotFoundError(f"'{slug_substring}' icin dataset bulunamadi. Add Input ile eklendi mi?")

OUR_DATASET_DIR = find_dataset_root("fire-smoke-merged")
DFIRE_DATASET_DIR = find_dataset_root("smoke-fire-detection-yolo")
print(f"[OK] Kendi veri setimiz : {OUR_DATASET_DIR}")
print(f"[OK] D-Fire veri seti   : {DFIRE_DATASET_DIR}")

WORK_DIR = Path("/kaggle/working/merged_dataset")
if WORK_DIR.exists():
    shutil.rmtree(WORK_DIR)

shutil.copytree(OUR_DATASET_DIR, WORK_DIR)

print("[OK] Kendi veri setimiz /kaggle/working/merged_dataset icine kopyalandi.")
print(list(WORK_DIR.iterdir()))

# === HUCRE 2b: D-Fire'i ayni canonical siniflarla (0=fire, 1=smoke) icine kat ===
# D-Fire'in kendi sinif sirasi TERS: 0=Smoke, 1=Fire (bizimkinin ters yonu).
# prepare_dataset.py'deki mantikla ayni: isme gore esleştirip class id'yi
# YENIDEN YAZARAK ekliyoruz, index'e guvenmiyoruz.
DFIRE_NAME_TO_CANONICAL_ID = {"smoke": 1, "fire": 0}  # canonical: 0=fire, 1=smoke

# D-Fire'in kendi data.yaml'indaki sinif sirasini okuyup dogruluyoruz
# (varsayimla degil, dosyadan okuyarak).
import yaml as _yaml
with open(DFIRE_DATASET_DIR / "data.yaml") as f:
    dfire_yaml = _yaml.safe_load(f)
dfire_old_id_to_canonical = {
    old_idx: DFIRE_NAME_TO_CANONICAL_ID[name.strip().lower()]
    for old_idx, name in enumerate(dfire_yaml["names"])
}
print(f"[OK] D-Fire class remap: {dfire_old_id_to_canonical} (orijinal isimler: {dfire_yaml['names']})")


def rewrite_label(src_path: Path, dst_path: Path, remap: dict) -> None:
    lines_out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            parts[0] = str(remap[int(parts[0])])
            lines_out.append(" ".join(parts))
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


# D-Fire'in klasor derinligi Kaggle'da nasil mount edildigine gore
# degisebiliyor (bazen DFIRE_DATASET_DIR/train/images, bazen
# DFIRE_DATASET_DIR/data/train/images - kendi data.yaml'inda ikincisi
# yaziyordu). Elle tahmin etmek yerine, "<split>/images" desenini
# DFIRE_DATASET_DIR altinda ARAYIP buluyoruz; hangi derinlikte olursa olsun
# calisir.
def resolve_split_dir(root: Path, split: str, kind: str) -> Path | None:
    direct = root / split / kind
    if direct.exists():
        return direct
    matches = list(root.rglob(f"{split}/{kind}"))
    return matches[0] if matches else None


for split in ["train", "val", "test"]:
    src_img_dir = resolve_split_dir(DFIRE_DATASET_DIR, split, "images")
    src_lbl_dir = resolve_split_dir(DFIRE_DATASET_DIR, split, "labels")
    dst_img_dir = WORK_DIR / split / "images"
    dst_lbl_dir = WORK_DIR / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    if src_img_dir is None:
        print(f"[UYARI] D-Fire icinde '{split}/images' bulunamadi, atlaniyor.")
        continue
    print(f"[OK] D-Fire {split} kaynagi: {src_img_dir}")

    # NOT: shutil.copy2 yerine SYMLINK kullaniyoruz. Gorselleri fiziksel
    # olarak kopyalamak (~21k dosya, Kaggle'in yavas /kaggle/input baglantisinda)
    # cok uzun surer ve /kaggle/working disk kotasini gereksiz doldurur.
    # Symlink, dosyayi kopyalamadan sadece bir "kisayol" olusturur; Ultralytics
    # okurken farki anlamaz, egitim ayni sekilde calisir.
    n_copied = 0
    for img_file in src_img_dir.glob("*"):
        if not img_file.is_file():
            continue
        (dst_img_dir / f"dfire_{img_file.name}").symlink_to(img_file)

        # "None" (yangin/duman OLMAYAN) gorsellerin etiket dosyasi hic
        # olmayabilir; bu durumda BOS etiket dosyasi yaziyoruz. Bu goruntuler
        # tam da elimizde eksik olan negatif/hard-negative orneklerdir ve
        # Ultralytics bos .txt'yi otomatik negatif ornek olarak kullanir.
        lbl_file = (src_lbl_dir / (img_file.stem + ".txt")) if src_lbl_dir else None
        dst_lbl_file = dst_lbl_dir / f"dfire_{img_file.stem}.txt"
        if lbl_file is not None and lbl_file.exists():
            rewrite_label(lbl_file, dst_lbl_file, dfire_old_id_to_canonical)
        else:
            dst_lbl_file.write_text("", encoding="utf-8")
        n_copied += 1

        # Donmus gibi gorunmesin diye her 2000 dosyada bir ilerleme yazdir.
        if n_copied % 2000 == 0:
            print(f"  ... {split}: {n_copied} islendi")

    print(f"[OK] D-Fire {split}: {n_copied} gorsel eklendi.")

print("[OK] D-Fire birlestirme tamamlandi.")
for split in ["train", "val", "test"]:
    n = len(list((WORK_DIR / split / "images").glob("*")))
    print(f"  {split}: toplam {n} gorsel")

# === HUCRE 3: data.yaml'i mutlak yola gore yeniden yaz ===
# Lokaldeki data.yaml'da "path" kasitli olarak yoktu (Ultralytics'in kendi
# datasets_dir ayarina takilmasin diye). Kaggle'da da ayni riskten kacinmak
# icin burada path'i ACIKCA ve MUTLAK olarak veriyoruz; en garantili yontem.
import yaml

data_yaml = {
    "path": str(WORK_DIR),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 2,
    "names": ["fire", "smoke"],
}
with open(WORK_DIR / "data.yaml", "w") as f:
    yaml.safe_dump(data_yaml, f, default_flow_style=False)

print(open(WORK_DIR / "data.yaml").read())

# === HUCRE 4: egitim ===
from ultralytics import YOLO

# yolo11s.pt burada otomatik indirilir (Internet: On olmali).
model = YOLO("yolo11s.pt")

model.train(
    data=str(WORK_DIR / "data.yaml"),
    epochs=60,              # D-Fire eklenince veri ~12x buyudu (1964 -> ~24000
                            # gorsel); onceki 150 epoch tavani bu olcekte
                            # ~20-30 saate cikar, Kaggle'in tek oturum GPU
                            # suresini asar. Her epoch artik 12x daha fazla
                            # farkli goruntu gordugu icin ayni ogrenme icin
                            # cok daha az epoch yeterli olur; patience zaten
                            # gerekirse daha erken durduracak
    patience=15,
    imgsz=640,
    batch=48,               # once dene; veri buyudugu icin dataloader/IO
                            # baski yapabilir, OOM ya da asiri yavaslama
                            # gorursen 24'e dus
    cache="disk",           # RAM'de degil DISK'te cache: D-Fire ile veri
                            # ~7GB'a cikti, Kaggle'in standart RAM'i (13GB)
                            # icin riskli; disk cache hala ilk epoch'tan sonra
                            # hizlanma saglar ama RAM'i patlatmaz
    optimizer="AdamW",
    device=0,
    project="/kaggle/working/runs/detect",
    name="fire_smoke_yolo11s",
)

print("[OK] Egitim tamamlandi. Agirliklar: /kaggle/working/runs/detect/fire_smoke_yolo11s/weights/best.pt")

# === HUCRE 5: sonucu indirilebilir hale getir ===
# Kaggle notebook'un sag panelinden "Output" olarak best.pt'yi indirebilirsin;
# ekstra guvenlik icin ayrica zip'liyoruz.
import shutil as _shutil
_shutil.make_archive("/kaggle/working/fire_smoke_yolo11s_weights", "zip",
                      "/kaggle/working/runs/detect/fire_smoke_yolo11s/weights")
print("[OK] /kaggle/working/fire_smoke_yolo11s_weights.zip hazir, Output sekmesinden indir.")
