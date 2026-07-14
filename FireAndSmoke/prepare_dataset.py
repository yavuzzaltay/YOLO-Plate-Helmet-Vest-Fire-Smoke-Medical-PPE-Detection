"""
prepare_dataset.py
-------------------
Bu script, Roboflow'dan indirilen 2 farklı yangin/duman veri setini
TEK ve TUTARLI bir veri setine ("merged_dataset") donusturur.

NEDEN BU SCRIPT GEREKLI?
Elimizdeki iki veri setinin sinif isimleri farkli dillerde/etiketlerle geldi:
  1) fire-smoke-1        klasoru -> siniflar: ["HUO", "YAN"]      (Cince pinyin: huo=ates/fire, yan=duman/smoke)
  2) fire-and-smoke-1    klasoru -> siniflar: ["fire", "smoke"]   (Ingilizce)

YOLO egitiminde her etiket (.txt) dosyasinin ilk sutunu bir "class id"
(0, 1, 2 ...) tutar ve bu id, data.yaml icindeki "names" listesinin
INDEKSINE karsilik gelir. Iki veri setini oldugu gibi ust uste kopyalarsak,
class id'ler ayni sayida olsa bile (0 ve 1) hangi ismi temsil ettikleri
FARKLI data.yaml'lara bagli olabilir. Bu yuzden id'lere korku duymadan
guvenmek yerine, her veri setinin KENDI data.yaml'indaki isim listesini
okuyup, isimleri anlamca (HUO->fire, YAN->smoke) eslestirip, etiket
dosyalarindaki class id'lerini merged_dataset icin ORTAK/CANONICAL
bir siraya (0=fire, 1=smoke) göre YENIDEN YAZIYORUZ. Boylece hangi
veri setinden geldigine bakmaksizin 0 her zaman "fire", 1 her zaman
"smoke" anlamina gelir.

BU SCRIPT NE YAPMAZ:
- Model egitimi (model.train(...)) BURADA YOK. Bu script sadece veriyi
  hazirlar. Egitim kismi ayri bir dosyada (train.py) olacak; boylece
  veri hazirlama ile egitim mantigi birbirine karismaz ve veri
  hazirlama adimini egitimi baslatmadan tek basina calistirip
  kontrol edebilirsin.
"""

import shutil
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# 1) TEMEL YOL AYARLARI
# ---------------------------------------------------------------------------
# Bu script FireAndSmoke klasorunun icinde yasiyor; tum yollari bu klasore
# gore (relative) kuruyoruz ki proje nereye tasinirsa tasinsin calissin.
BASE_DIR = Path(__file__).resolve().parent

# Ham (indirilmis) Roboflow veri setlerinin klasor adlari.
# Bu klasorler zaten FireAndSmoke altinda mevcut (Roboflow'dan onceden indirildi).
RAW_DATASETS = [
    {
        "dir_name": "fire-smoke-1",         # HUO / YAN (Cince) etiketli veri seti
        "prefix": "fs1",                     # birlestirirken dosya adi cakismasini onlemek icin on ek
    },
    {
        "dir_name": "fire-and-smoke-1",     # fire / smoke (Ingilizce) etiketli veri seti
        "prefix": "fs2",
    },
]

MERGED_DIR = BASE_DIR / "merged_dataset"

# ---------------------------------------------------------------------------
# 2) SINIF ISIMLERINI ORTAK (CANONICAL) HALE GETIRME
# ---------------------------------------------------------------------------
# Nihai/ortak siniflarimiz ve sirasi: 0 = fire, 1 = smoke.
CANONICAL_CLASSES = ["fire", "smoke"]

# Veri setlerinde karsimiza cikabilecek her turlu isim varyasyonunu
# (Ingilizce, Cince pinyin, buyuk/kucuk harf farki vs.) ortak isme baglayan
# sozluk. Yeni bir veri seti eklenirse farkli bir etiket ismi gelirse
# bu sozluge tek satir eklemek yeterli olur.
NAME_SYNONYMS = {
    "fire": "fire",
    "huo": "fire",     # 火 (huo) = ates/yangin
    "smoke": "smoke",
    "yan": "smoke",    # 烟 (yan) = duman
}


def build_index_remap(dataset_yaml_names: list[str]) -> dict[int, int]:
    """Bir veri setinin kendi 'names' listesindeki eski class id'lerini
    CANONICAL_CLASSES icindeki yeni (ortak) class id'lerine eslestiren
    bir sozluk uretir. Ornek: ["HUO", "YAN"] -> {0: 0, 1: 1}
    (HUO=fire=canonical 0, YAN=smoke=canonical 1)
    """
    remap = {}
    for old_idx, raw_name in enumerate(dataset_yaml_names):
        key = raw_name.strip().lower()
        if key not in NAME_SYNONYMS:
            raise ValueError(
                f"Bilinmeyen sinif ismi: '{raw_name}'. "
                f"NAME_SYNONYMS sozlugune eklemeniz gerekiyor."
            )
        canonical_name = NAME_SYNONYMS[key]
        new_idx = CANONICAL_CLASSES.index(canonical_name)
        remap[old_idx] = new_idx
    return remap


# ---------------------------------------------------------------------------
# 3) TEK BIR ETIKET (.txt) DOSYASINI YENI CLASS ID'LERIYLE YENIDEN YAZMA
# ---------------------------------------------------------------------------
def rewrite_label_file(src_path: Path, dst_path: Path, remap: dict[int, int]) -> None:
    """Bir YOLO etiket dosyasini okur, her satirin basindaki class id'yi
    remap sozlugune gore degistirir ve yeni dosyaya yazar.

    Not: Bu veri setlerindeki etiketler poligon (segmentation) formatinda,
    yani "class_id x1 y1 x2 y2 x3 y3 ..." seklinde degisken sayida sutun
    icerebiliyor. Biz sadece ilk sutunu (class id) degistiriyoruz, geri
    kalan koordinatlara dokunmuyoruz; bu yuzden format ne olursa olsun
    (bbox ya da polygon) bu fonksiyon calisir.
    """
    lines_out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_class_id = int(parts[0])
            new_class_id = remap[old_class_id]
            parts[0] = str(new_class_id)
            lines_out.append(" ".join(parts))

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


# ---------------------------------------------------------------------------
# 4) BIR VERI SETININ TAMAMINI (train/valid/test) MERGED_DATASET'E KOPYALAMA
# ---------------------------------------------------------------------------
def merge_dataset(dir_name: str, prefix: str) -> None:
    src_root = BASE_DIR / dir_name
    yaml_path = src_root / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"{yaml_path} bulunamadi. '{dir_name}' veri setinin Roboflow'dan "
            f"indirilip FireAndSmoke klasorune cikartildigindan emin olun."
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)

    remap = build_index_remap(data_yaml["names"])
    print(f"[INFO] {dir_name}: sinif eslestirmesi -> {remap} "
          f"(orijinal isimler: {data_yaml['names']})")

    # Roboflow bazen "valid" bazen "val" klasor adini kullanir; ikisini de kontrol ediyoruz.
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}

    for src_split_name, dst_split_name in split_map.items():
        src_split_dir = src_root / src_split_name
        if not src_split_dir.exists():
            continue

        src_img_dir = src_split_dir / "images"
        src_lbl_dir = src_split_dir / "labels"
        dst_img_dir = MERGED_DIR / dst_split_name / "images"
        dst_lbl_dir = MERGED_DIR / dst_split_name / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Gorselleri oldugu gibi kopyala (on ek ekleyerek dosya adi cakismasini onle).
        if src_img_dir.exists():
            for img_file in src_img_dir.glob("*"):
                if img_file.is_file():
                    shutil.copy2(img_file, dst_img_dir / f"{prefix}_{img_file.name}")

        # Etiketleri KOPYALAMIYORUZ, class id'lerini yeniden yazarak aktariyoruz.
        if src_lbl_dir.exists():
            for lbl_file in src_lbl_dir.glob("*.txt"):
                rewrite_label_file(lbl_file, dst_lbl_dir / f"{prefix}_{lbl_file.name}", remap)


# ---------------------------------------------------------------------------
# 5) ANA AKIS
# ---------------------------------------------------------------------------
def main() -> None:
    # Onceki birlestirme sonucu varsa temizleyip sifirdan uretiyoruz ki
    # yarim kalmis / eski dosyalar karisikliga yol acmasin.
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for dataset in RAW_DATASETS:
        merge_dataset(dataset["dir_name"], dataset["prefix"])

    # Nihai data.yaml: YOLO'nun egitimde okuyacagi dosya.
    # "path" alanini BILEREK yazmiyoruz: Ultralytics, "path" relative
    # verilirse onu FireAndSmoke klasorune degil, kendi global
    # "datasets_dir" ayarina (Ultralytics/settings.json) gore cozuyor.
    # Bu da (ozellikle bilgisayarda baska YOLO projeleri de varsa, orn.
    # PlateDetection) yanlis klasorde veri aramasina yol aciyor.
    # "path" hic verilmezse Ultralytics data.yaml'in KENDI bulundugu
    # klasoru taban alir; bu yuzden her ortamda calisan, tasinabilir
    # bir cozum.
    merged_yaml = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CANONICAL_CLASSES),
        "names": CANONICAL_CLASSES,
    }
    with open(MERGED_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"[OK] Birlestirilmis veri seti hazir: {MERGED_DIR}")
    print(f"[OK] Ortak siniflar: {CANONICAL_CLASSES} (0=fire, 1=smoke)")


if __name__ == "__main__":
    main()
