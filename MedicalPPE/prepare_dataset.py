# -*- coding: utf-8 -*-
"""
prepare_dataset.py
-------------------
Roboflow'dan 3 farklı tıbbi PPE veri setini indirir ve TEK/TUTARLI bir
veri setine ("MedicalPPE/dataset") dönüştürür. best.pt'yi eğiten Kaggle
not defterindeki mantığın birebir yerel karşılığıdır (aynı MAPPING/
FINAL_NAMES) — böylece evaluate.py bu veri setiyle çalıştığında best.pt
ile TUTARLI sınıf sırasını ölçer.

NEDEN BU SCRIPT GEREKLİ? (FireAndSmoke/prepare_dataset.py ile aynı sorun)
3 ayrı Roboflow projesi aynı ekipmanları FARKLI isim/sırayla etiketlemiş
("Surgical Apron" vs "surgical-gown", büyük/küçük harf farkı vs.). YOLO
eğitiminde her etiket dosyasının class id'si data.yaml'daki "names"
listesinin İNDEKSİNE karşılık gelir; veri setlerini olduğu gibi üst üste
koyarsak id'ler çakışıp yanlış sınıflara karışır. Bu yüzden her veri
setinin KENDİ isim listesini okuyup FINAL_NAMES'teki ORTAK sıraya göre
class id'lerini YENİDEN YAZIYORUZ.

BU SCRIPT NE YAPMAZ: model eğitimi burada YOK (train.py'de).

API ANAHTARI NASIL VERİLİR? (repo public olduğu için ASLA koda gömülmez)
  1) Ortam değişkeni:  $env:ROBOFLOW_API_KEY = "..."  (PowerShell)
  2) Ya da MedicalPPE/.roboflow_key dosyasına TEK SATIR anahtarı yaz
     (.gitignore'da zaten hariç tutuluyor, commit'e asla girmez).
"""

import os
import shutil
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent
# DİKKAT — RAW_DIR proje klasörünün İÇİNDE DEĞİL, kısa bir yolda (C:\ppe_raw):
# Roboflow'un bazı projelerindeki dosya adları ~200 karakter olabiliyor;
# bu proje klasörünün derin/Unicode/boşluklu yoluyla ("...\<proje-kökü>\MedicalPPE\
# raw_datasets\dsX\valid\images\...") birleşince Windows'un 260 karakterlik
# MAX_PATH sınırını aşıp "FileNotFoundError" ile patlıyor (kayıt defteri
# değişikliği gerektirmeden çözüm: yolu kısalt). Birleştirme adımında
# (merge_all) zaten kısa/sıralı adlarla DATASET_DIR'e kopyalıyoruz, o yüzden
# bu kısa klasör yalnızca GEÇİCİ ham veri için kullanılıyor.
RAW_DIR = Path(os.environ.get("SystemDrive", "C:") + "\\") / "ppe_raw"
DATASET_DIR = BASE_DIR / "dataset"
KEY_FILE = BASE_DIR / ".roboflow_key"

# ---------------------------------------------------------------------------
# 1) İNDİRİLECEK PROJELER
# ---------------------------------------------------------------------------
# ds1-ds3: best.pt'yi eğiten Kaggle akışıyla birebir aynı ÇEKİRDEK kaynaklar.
#
# ds4-ds5: 2026-07-17'de eklenen TAKVİYE kaynaklar — Sensors 2025 hijyen uyumu
# çalışmasının (https://www.mdpi.com/1424-8220/25/19/6140, kaynakça 21-22)
# kullandığı, GERÇEK mutfak/işyeri ortamından çekilmiş projeler. Neden:
# zayıf sınıflarımızın (no-surgical-cap 0.253 / no-surgical-gloves 0.195 /
# no-facial-gear 0.304 mAP50-95) örnekleri o ana dek TEK kaynaktan (ds2,
# stüdyo/stok tarzı) geliyordu; bu projeler aynı ihlallerin (no_glove,
# no_hairnet, maskoff) doğal-ortam örnekleriyle tek-kaynak bağımlılığını
# kırıyor. İki kural (aşağıdaki alanlarla uygulanır):
#   only_final_ids: yalnızca ZAYIF sınıf içeren görüntüler alınır — binlerce
#     alakasız mutfak fotoğrafıyla veri setini şişirmemek için.
#   test_to_train=True: bu kaynakların HİÇBİR görüntüsü test'e GİRMEZ
#     (kendi test split'leri de train'e akar). Böylece benchmark test seti
#     (ds1-ds3'ün orijinal test'i) birebir sabit kalır ve yeniden eğitim
#     sonrası "iyileşti mi?" karşılaştırması elmayla elma olur.
ROBOFLOW_PROJECTS = [
    {"dir_name": "ds1", "workspace": "ribal", "project": "surgical-dataset-hsnio", "version": 1},
    {"dir_name": "ds2", "workspace": "ecrioobjectdetection1", "project": "meppe", "version": 1},
    {"dir_name": "ds3", "workspace": "new-workspace-n2w3n", "project": "medical-ppe-o6ot6-gesry", "version": 1},
    {"dir_name": "ds4", "workspace": "gp-zmz2y", "project": "gp-fqlbs", "version": 1,
     "only_final_ids": {2, 4, 10}, "test_to_train": True},
    {"dir_name": "ds5", "workspace": "nku-oyddi", "project": "final-work-nyuq2", "version": 1,
     "only_final_ids": {2, 4, 10}, "test_to_train": True},
]

# ---------------------------------------------------------------------------
# 2) SINIF İSİMLERİNİ ORTAK (CANONICAL) HALE GETİRME
# ---------------------------------------------------------------------------
# best.pt'nin GERÇEK sınıf sırası (14 sınıf) — bu sıra değişirse best.pt
# ile veri seti arasındaki eşleşme bozulur, DEĞİŞTİRME.
FINAL_NAMES = [
    "person", "surgical-cap", "no-surgical-cap", "surgical-gloves",
    "no-surgical-gloves", "surgical-mask", "surgical-gown", "shoe-covers",
    "surgical-scrubs", "face-shield", "no-facial-gear", "no-medical-attire",
    "goggles", "coverall",
]

# Her ham veri setinin KENDİ isimlerinden FINAL_NAMES indeksine eşleme.
# Anahtar = o veri setinin data.yaml'ındaki TAM isim (büyük/küçük harf
# duyarlı); değer = FINAL_NAMES içindeki indeks.
CLASS_MAPPING = {
    "ds1": {
        "Person": 0, "Shoe Covers": 7, "Surgical Apron": 6,
        "Surgical Cap": 1, "Surgical Gloves": 3, "Surgical Mask": 5,
        "Surgical Scrubs": 8,
    },
    "ds2": {
        "face-shield": 9, "no-facial-gear": 10, "no-medical-attire": 11,
        "no-surgical-cap": 2, "no-surgical-gloves": 4, "person": 0,
        "shoe-covers": 7, "surgical-cap": 1, "surgical-gloves": 3,
        "surgical-gown": 6, "surgical-mask": 5, "surgical-scrubs": 8,
    },
    "ds3": {
        "Coverall": 13, "Gloves": 3, "Goggles": 12, "Mask": 5,
    },
    # ds4 (GP) / ds5 (Final Work): mutfak-hijyen etiketlerinin tıbbi
    # karşılıkları. Eşleme gerekçeleri:
    #   hairnet/bone → surgical-cap: ikisi de bouffant/saç örtüsü, görsel
    #     olarak aynı ekipman ailesi; no_hairnet → no-surgical-cap.
    #   maskoff / no_mask → no-facial-gear: yüzü korumasız kişi — bizim
    #     şemada bunun karşılığı "no-facial-gear" (maske ayrı, yüz koruması
    #     bütünsel). Çene-altı maske görüntüleri de ihlal semantiğiyle uyumlu.
    # DİKKAT: buradaki adlar indirilen data.yaml'daki GERÇEK adlarla
    # eşleşmezse _build_remap uyarı basar — o zaman adları oradan düzelt.
    "ds4": {
        "glove": 3, "no_glove": 4, "hairnet": 1, "no_hairnet": 2,
        "maskon": 5, "maskoff": 10,
    },
    "ds5": {
        "gloves": 3, "no_gloves": 4, "Hairnet": 1, "no_hairnet": 2,
        "mask": 5, "no_mask": 10,   # "Hairnet" büyük H: data.yaml'da gerçekten böyle
    },
}

# Roboflow projeleri "valid" klasör adını kullanır; birleştirilmiş veri
# setinde de aynı adı koruyoruz (best.pt'yi eğiten Kaggle akışıyla birebir
# aynı klasör yapısı — evaluate.py bunu bu şekilde bekliyor).
SPLITS = ["train", "valid", "test"]


def _get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise RuntimeError(
        "Roboflow API anahtarı bulunamadı. Ya ROBOFLOW_API_KEY ortam "
        f"değişkenini ayarla ya da anahtarı '{KEY_FILE}' dosyasına "
        "tek satır olarak yaz (bu dosya .gitignore'da, commit'e girmez)."
    )


# ---------------------------------------------------------------------------
# 3) İNDİRME
# ---------------------------------------------------------------------------
def download_all(force: bool = False) -> None:
    """3 projeyi de RAW_DIR altına indirir.

    force=False ise (varsayılan) klasör zaten doluysa indirmeyi ATLAR —
    Roboflow'un ücretsiz plan indirme kotasını gereksiz tüketmemek için.
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=_get_api_key())

    for proj in ROBOFLOW_PROJECTS:
        target = RAW_DIR / proj["dir_name"]
        if target.exists() and any(target.iterdir()) and not force:
            print(f"[ATLA] {proj['dir_name']}: zaten indirilmiş ({target})")
            continue
        if target.exists():
            shutil.rmtree(target)

        print(f"[INFO] İndiriliyor: {proj['workspace']}/{proj['project']} "
              f"v{proj['version']} -> {target}")
        rf.workspace(proj["workspace"]).project(proj["project"]) \
          .version(proj["version"]).download("yolov11", location=str(target))

    print("[OK] Tüm veri setleri indirildi.")


# ---------------------------------------------------------------------------
# 4) SINIF ID'LERİNİ YENİDEN YAZIP BİRLEŞTİRME
# ---------------------------------------------------------------------------
def _build_remap(dir_name: str, old_names: list[str]) -> dict[int, int]:
    """Bu veri setinin eski class id'lerini FINAL_NAMES indekslerine eşler.

    Eşleşmeyen (CLASS_MAPPING'te olmayan) bir sınıf çıkarsa UYARIR — bu
    sınıfa ait tüm kutular sessizce atlanır, veri setinden habersiz nesne
    kaybı yaşanmasın diye açıkça bildiriyoruz.
    """
    class_map = CLASS_MAPPING[dir_name]
    remap = {}
    for old_idx, raw_name in enumerate(old_names):
        if raw_name in class_map:
            remap[old_idx] = class_map[raw_name]
        else:
            print(f"[UYARI] {dir_name}: '{raw_name}' sınıfı CLASS_MAPPING'te "
                  f"yok — bu sınıfa ait etiketler ATLANACAK.")

    unused = set(class_map) - set(old_names)
    if unused:
        print(f"[UYARI] {dir_name}: CLASS_MAPPING'teki {sorted(unused)} "
              f"veri setinin gerçek isimlerinde bulunamadı (yazım/versiyon "
              f"farkı olabilir) — kontrol et.")
    return remap


def _rewrite_label(src_path: Path, dst_path: Path, remap: dict[int, int]) -> None:
    """Bir etiket dosyasını, ilk sütunu (class id) yeniden yazarak kopyalar.

    Poligon/bbox fark etmez: sadece ilk sütuna dokunulur, geri kalan
    koordinatlar aynen aktarılır.
    """
    lines_out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_id = int(parts[0])
            if old_id not in remap:
                continue   # eşlenmeyen sınıf: bu kutu atlanır (yukarıda uyarıldı)
            parts[0] = str(remap[old_id])
            lines_out.append(" ".join(parts))
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))


def _label_final_ids(label_file: Path, remap: dict[int, int]) -> set[int]:
    """Bir etiket dosyasındaki kutuların FINAL_NAMES id'lerini döner
    (only_final_ids filtresi için)."""
    ids = set()
    if not label_file.exists():
        return ids
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and int(parts[0]) in remap:
            ids.add(remap[int(parts[0])])
    return ids


def merge_all() -> None:
    """RAW_DIR altındaki veri setlerini DATASET_DIR'e birleştirir."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in SPLITS:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    for proj in ROBOFLOW_PROJECTS:
        dir_name = proj["dir_name"]
        only_final_ids = proj.get("only_final_ids")
        test_to_train = proj.get("test_to_train", False)
        src_root = RAW_DIR / dir_name
        yaml_path = src_root / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"{yaml_path} bulunamadı. Önce download_all() ile veri "
                f"setlerini indir (python prepare_dataset.py)."
            )

        with open(yaml_path, "r", encoding="utf-8") as f:
            old_names = yaml.safe_load(f)["names"]
        remap = _build_remap(dir_name, old_names)
        print(f"[INFO] {dir_name}: {len(remap)}/{len(old_names)} sınıf eşlendi")

        kept = skipped = 0
        for split in SPLITS:
            src_img_dir = src_root / split / "images"
            src_lbl_dir = src_root / split / "labels"
            if not src_img_dir.exists():
                continue   # bu veri setinde bu split yok (ör. test klasörü olmayabilir)

            # test_to_train: takviye kaynakların test görüntüleri de train'e
            # akar — benchmark test seti yalnızca çekirdek (ds1-ds3)
            # kaynaklardan oluşur, sabit kalır (gerekçe: dosya başındaki
            # ROBOFLOW_PROJECTS yorumu).
            dst_split = "train" if (test_to_train and split == "test") else split
            dst_img_dir = DATASET_DIR / dst_split / "images"
            dst_lbl_dir = DATASET_DIR / dst_split / "labels"

            # DİKKAT: orijinal Roboflow dosya adları KISALTILIYOR (ds1_00001
            # gibi sıralı adlarla). Nedeni sadece kozmetik değil: bazı
            # Roboflow projelerinde dosya adı ~200 karakter oluyor; bu proje
            # klasörünün derin yoluyla ("...\<proje-kökü>\MedicalPPE\dataset\train\
            # images\...") birleşince Windows'un 260 karakterlik MAX_PATH
            # sınırı yine aşılırdı (RAW_DIR'i kısaltmak burada işe yaramaz,
            # DATASET_DIR proje içinde kalmalı). Kısa/sıralı ad hem sınırı
            # aşmayı önlüyor hem de dosya adında olabilecek Unicode/özel
            # karakter sorunlarını baştan eliyor.
            # (İsimde split ön eki var ki test_to_train train'e akıtırken
            # farklı split'lerden gelen sıra numaraları çakışmasın.)
            for idx, img_file in enumerate(sorted(p for p in src_img_dir.glob("*") if p.is_file())):
                label_file = src_lbl_dir / f"{img_file.stem}.txt"

                # only_final_ids filtresi: görüntüde hedef sınıflardan en az
                # biri yoksa alma (takviye kaynakları şişkinlik yapmasın).
                if only_final_ids is not None:
                    if not (_label_final_ids(label_file, remap) & only_final_ids):
                        skipped += 1
                        continue

                new_stem = f"{dir_name}_{split[:2]}{idx:05d}"
                new_name = new_stem + img_file.suffix.lower()
                shutil.copy2(img_file, dst_img_dir / new_name)
                kept += 1

                new_label_path = dst_lbl_dir / f"{new_stem}.txt"
                if label_file.exists():
                    _rewrite_label(label_file, new_label_path, remap)
                else:
                    # Etiketsiz görüntü = negatif/arka plan örneği (nesne
                    # yok); YOLO bunu boş etiket dosyasıyla kabul eder.
                    new_label_path.touch()

        if only_final_ids is not None:
            print(f"[INFO] {dir_name}: {kept} görüntü alındı, {skipped} tanesi "
                  f"hedef sınıf içermediği için atlandı")

    # Nihai data.yaml: "path" alanını BİLEREK yazmıyoruz (FireAndSmoke ile
    # aynı gerekçe) — Ultralytics'in global datasets_dir ayarına göre değil,
    # data.yaml'ın KENDİ bulunduğu klasöre göre yol çözmesi için; böylece
    # bilgisayarda başka YOLO projeleri (PlateDetection vs.) olsa da karışmaz.
    merged_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(FINAL_NAMES),
        "names": FINAL_NAMES,
    }
    with open(DATASET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_yaml, f, default_flow_style=False, allow_unicode=True)

    n_train = len(list((DATASET_DIR / "train" / "images").glob("*")))
    n_valid = len(list((DATASET_DIR / "valid" / "images").glob("*")))
    n_test = len(list((DATASET_DIR / "test" / "images").glob("*")))
    print(f"[OK] Birleştirilmiş veri seti hazır: {DATASET_DIR}")
    print(f"[OK] train={n_train}  valid={n_valid}  test={n_test}  görüntü")
    print(f"[OK] {len(FINAL_NAMES)} sınıf: {FINAL_NAMES}")


def main() -> None:
    download_all()
    merge_all()


if __name__ == "__main__":
    main()
