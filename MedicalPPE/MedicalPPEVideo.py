# -*- coding: utf-8 -*-
"""
MedicalPPEVideo.py — Tıbbi PPE (KKD) uyum izleme çekirdeği
===========================================================

AMAÇ: Sağlık ortamında çalışanların tıbbi ekipmanları (eldiven, bone,
önlük, gözlük, maske...) takıp takmadığını görüntü/videodan denetlemek.

BU MODÜLÜN GENEL MİMARİSİ (FireAndSmokeVideo.py ile aynı katman düzeni):

    KAYNAK (video dosyası / IP kamera / webcam)
       │  her kare işlenmez; app.py tarafı kare atlama (dosya) veya
       │  zaman bazlı örnekleme (canlı) ile kareleri seyreltir
       ▼
    1) YOLO TRACK  ── model.track(persist=True)
       │  ByteTrack her nesneye (özellikle KİŞİLERE) kalıcı kimlik verir;
       │  böylece "3 numaralı kişi 8 karedir eldivensiz" diyebiliriz.
       ▼
    2) SINIF ÇÖZÜMLEME + GÜVEN EŞİĞİ
       │  Model henüz eğitimde olduğu için sınıf adlarını ŞİMDİDEN
       │  bilmiyoruz. Bu modül model.names'i okuyup her sınıfı takma ad
       │  tablosuyla (ITEM_ALIASES) kanonik bir ekipmana eşler:
       │  "gloves"→eldiven, "hairnet"→bone, "no_gown"→önlük İHLALİ...
       │  Böylece eğitim hangi adlarla biterse bitsin kod değişmez.
       ▼
    3) UYUM MANTIĞI — model sınıflarına göre 3 moddan biri otomatik seçilir:
       │  • "direct" : modelde no_glove gibi NEGATİF sınıflar varsa
       │               her negatif tespit doğrudan ihlal adayıdır.
       │               (Sensors 2025 çalışmasının önerdiği düzen; en sağlıklısı)
       │  • "person" : model sadece kişi + pozitif ekipman veriyorsa
       │               ekipman kutuları kişilere atanır; kişide EKSİK
       │               kalan zorunlu ekipman ihlal adayı olur.
       │  • "presence": kişi de negatif sınıf da yoksa ihlal çıkarımı
       │               yapılamaz; yalnızca tespit/sayım yapılır.
       ▼
    4) N-of-M ZAMANSAL FİLTRE
       │  Tek karelik "eldiven yok" güvenilmez: el cebe girer, vücudun
       │  arkasında kalır, eldiven ten rengine karışır. Gerçek ihlal
       │  KALICIDIR. Kural: bir iz (kişi ya da negatif tespit), son M
       │  örnekleme karesinin en az N'inde ihlalli görüldüyse ONAYLANIR.
       │  Varsayılan 6/10: ~4 Hz örneklemede 2.5 saniyelik pencerede
       │  1.5 saniyelik kanıt (oklüzyon yüzünden yangından daha temkinli).
       ▼
    5) ONAY ANINDA KANIT KAYDI  (alarm/bildirim YOK — bilinçli tercih)
          • Anotasyonlu tam kare fotoğraf → IhlalKayitlari/ klasörüne
          • CSV satırı (tarih, video sn, iz id, eksik ekipman, güven, dosya)
          Aynı iz + aynı eksik ekipman için kanıt SADECE BİR KEZ alınır.

EKRANDAKİ RENK DİLİ:
    • İnce yeşil kutu  = takılı ekipman (bilgi amaçlı)
    • İnce sarı kutu   = ADAY ihlal (henüz N-of-M onayı yok)
    • Kalın kırmızı    = ONAYLI ihlal
    • Yeşil kişi kutusu= tüm zorunlu ekipmanları TAM olan kişi

Bu modül arayüz İÇERMEZ: Streamlit (app.py) veya komut satırı (main())
buradaki MedicalPPEMonitor sınıfını kullanır.

NOT — MODEL HENÜZ YOK: eğitim sürüyor. best.pt hazır olunca ya
MedicalPPE/MedicalPPE.pt olarak buraya kopyala ya da eğitimi
MedicalPPE/runs/detect altında bitir; find_weights() ikisini de bulur.
"""

import csv
import re
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent


def find_weights():
    """Tıbbi PPE model ağırlığını bulur.

    Öncelik sırası:
      1) MedicalPPE/MedicalPPE.pt      (elle bırakılan hazır model)
      2) MedicalPPE/runs/detect/*/weights/best.pt  (en son eğitim, henüz
         adlandırılmamış ham Ultralytics çıktısı)
    """
    direct = BASE_DIR / "MedicalPPE.pt"
    if direct.exists():
        return str(direct)

    candidates = list((BASE_DIR / "runs" / "detect").glob("*/weights/best.pt"))
    if candidates:
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    raise FileNotFoundError(
        f"{BASE_DIR} altında model bulunamadı. Eğitim bittiğinde best.pt "
        f"dosyasını '{BASE_DIR / 'MedicalPPE.pt'}' olarak kopyala (veya eğitimi "
        f"'{BASE_DIR / 'runs' / 'detect'}' altında çalıştır)."
    )


# ─────────────────────── SINIF ADI ÇÖZÜMLEME ───────────────────────
# Model hangi veri setiyle eğitilirse eğitilsin sınıf adları değişebilir
# ("gloves", "Glove", "tibbi-eldiven", "no_gown"...). Aşağıdaki tablo
# olası adları kanonik ekipman adına eşler. Eşleşmeyen bir ad çıkarsa
# ilgili kümeye eklemen yeterli.
#
# Eğitilen modelin GERÇEK sınıfları (yolo11s, 2026-07-16, mAP50 0.771):
#   person, surgical-cap/no-surgical-cap, surgical-gloves/no-surgical-gloves,
#   surgical-mask, surgical-gown, shoe-covers, surgical-scrubs, face-shield,
#   no-facial-gear, no-medical-attire, goggles, coverall
# Hepsi aşağıdaki tabloyla çözülür; no_* sınıfları bulunduğu için izleme
# "direct" modda çalışır (en sağlıklı düzen).
ITEM_ALIASES = {
    "eldiven": {"eldiven", "glove", "gloves", "medicalglove", "medicalgloves",
                "surgicalglove", "surgicalgloves", "latexglove", "latexgloves",
                "tibbieldiven", "handglove", "handgloves"},
    "bone": {"bone", "bonnet", "hairnet", "haircap", "bouffant", "bouffantcap",
             "cap", "surgicalcap", "hat", "headcover", "headcap"},
    # DİKKAT: scrubs (forma) ayrı bir kanonik ekipmandır, önlük DEĞİLDİR —
    # forma giymiş ama önlüksüz birini "önlüklü" saymak yanlış olurdu.
    "onluk": {"onluk", "gown", "gowns", "apron", "coverall", "coveralls",
              "labcoat", "medicalgown", "isolationgown", "surgicalgown"},
    "forma": {"forma", "scrub", "scrubs", "surgicalscrubs", "medicalscrubs",
              "scrubsuit"},
    "galos": {"galos", "shoecover", "shoecovers", "bootcover", "bootcovers",
              "overshoe", "overshoes"},
    "gozluk": {"gozluk", "goggle", "goggles", "safetyglasses", "glasses",
               "eyewear", "safetygoggles", "protectiveglasses"},
    "maske": {"maske", "mask", "facemask", "maskon", "surgicalmask",
              "medicalmask", "n95"},
    "siperlik": {"siperlik", "faceshield", "shield", "visor"},
    # "no-facial-gear" = yüzde hiçbir koruma yok (maske/siperlik/gözlük);
    # "no-medical-attire" = medikal kıyafet (önlük/forma) yok. İkisinin de
    # yalnızca NEGATİF hali var; pozitif karşılıkları ayrı sınıflarda zaten.
    "yuz-korumasi": {"yuzkorumasi", "facialgear", "facegear",
                     "facialprotection", "faceprotection"},
    "medikal-kiyafet": {"medikalkiyafet", "medicalattire", "attire",
                        "medicalclothing", "medicalclothes"},
}
PERSON_ALIASES = {"person", "human", "people", "kisi", "insan", "worker",
                  "staff", "personel", "doctor", "nurse", "medicalstaff"}
# "no_glove", "without-mask", "incorrect_mask" gibi adlardaki olumsuzluk
# ekleri: ad bu öneklerden biriyle başlıyorsa (veya "off" ile bitiyorsa)
# kalan kısım bir ekipmana eşleşiyorsa NEGATİF (ihlal) sınıfı sayılır.
# "incorrect_mask" da ihlaldir: takılı ama YANLIŞ takılı (Sensors 2025
# çalışması bunu ayrı sınıf olarak modellemeyi öneriyor).
NEGATION_PREFIXES = ("no", "non", "without", "incorrect", "wrong", "improper", "missing")

_TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def _normalize(name: str) -> str:
    """'No-Glove' → 'noglove', 'Tıbbi Eldiven' → 'tibbieldiven'."""
    return re.sub(r"[^a-z0-9]", "", str(name).translate(_TR_MAP).lower())


class ClassInfo:
    """Modeldeki TEK bir sınıfın çözümlenmiş anlamı.

    role     : "person" | "item" | "unknown"
    item     : kanonik ekipman adı (role=="item" ise; ör. "eldiven")
    positive : True = ekipman TAKILI sınıfı, False = ihlal sınıfı (no_*)
    """

    def __init__(self, raw, role, item=None, positive=True):
        self.raw = raw
        self.role = role
        self.item = item
        self.positive = positive


def resolve_model_classes(names):
    """model.names sözlüğünü {indeks: ClassInfo} tablosuna çevirir.

    Eşleşmeyen sınıflar "unknown" kalır: çizilir ve sayılır ama uyum
    mantığına girmez (yanlış varsayım yapmaktansa dışarıda bırakmak
    daha güvenli).
    """
    resolved = {}
    for idx, raw in names.items():
        norm = _normalize(raw)

        if norm in PERSON_ALIASES:
            resolved[idx] = ClassInfo(raw, "person")
            continue

        matched = False
        # Önce pozitif eşleşme dene ("gloves" → eldiven)
        for item, aliases in ITEM_ALIASES.items():
            if norm in aliases:
                resolved[idx] = ClassInfo(raw, "item", item, positive=True)
                matched = True
                break
        if matched:
            continue

        # Sonra olumsuz kalıpları dene ("noglove", "incorrectmask", "maskoff")
        candidate = None
        for prefix in NEGATION_PREFIXES:
            if norm.startswith(prefix):
                candidate = norm[len(prefix):]
                break
        if candidate is None and norm.endswith("off"):
            candidate = norm[:-3]
        if candidate:
            for item, aliases in ITEM_ALIASES.items():
                if candidate in aliases:
                    resolved[idx] = ClassInfo(raw, "item", item, positive=False)
                    matched = True
                    break
        if not matched:
            resolved[idx] = ClassInfo(raw, "unknown")
    return resolved


# ─────────────────────────── AYARLAR ───────────────────────────
# evaluate.py'nin 2026-07-17'de MedicalPPE/dataset test setinde (535
# görüntü) ürettiği F1-Confidence eğrisinden bulundu (FireAndSmoke'ta
# fire 0.37/smoke 0.23 aynı yöntemle bulunmuştu). Genel özet: mAP50
# 0.800, precision 0.763, recall 0.803. En zayıf sınıflar beklendiği gibi
# NEGATİF sınıflar (no-surgical-gloves F1 0.53, no-surgical-cap F1 0.66)
# — "yokluk" tespiti literatürde de hep daha zor çıkıyor (bkz. README).
#
# NOT (2026-07-22): "best_model_v2.zip" olarak gelen alternatif model
# denendi ve AYNI test setinde bu v1'den (neredeyse tüm sınıflarda,
# 14 sınıfın 13'ünde mAP50-95 gerilemiş) daha kötü çıktığı için
# reddedildi, bu sözlük v1 değerlerine GERİ ALINDI. Ayrıntılı kıyaslama
# tablosu: README.md "v1 ↔ v2 kıyaslaması". Yedek dosya:
# best_v1_640px_2026-07-16.pt (şu an best.pt ile aynı içerik).
DEFAULT_CONF = 0.40
DEFAULT_THRESHOLDS = {
    "person": 0.36,
    "surgical-cap": 0.43,
    "no-surgical-cap": 0.26,
    "surgical-gloves": 0.43,
    "no-surgical-gloves": 0.37,
    "surgical-mask": 0.34,
    "surgical-gown": 0.49,
    "shoe-covers": 0.21,
    "surgical-scrubs": 0.36,
    "face-shield": 0.46,
    "no-facial-gear": 0.25,
    "no-medical-attire": 0.21,
    "goggles": 0.36,
    "coverall": 0.18,
}

# FOTOĞRAF hattının kendi eşikleri. NEDEN AYRI SET: fotoğraf modunda
# tespit TTA + dilimli geçişlerden geçiyor; TTA doğru tespitlerin güvenini
# sistematik yükseltiyor ve F1 eğrisinin tepesi sağa kayıyor (2026-07-22,
# 535 görüntülük testte GERÇEK hattın F1 eğrisinden ölçüldü — ortalama
# +0.15). Video hattında TTA/dilim YOK; oradaki güven dağılımı düz
# model.val'e yakın olduğundan video DEFAULT_THRESHOLDS ile kalır.
# Eşiklerin yüksek olması, alakasız sahnelerde (v2 denemesinde görülen
# "yangın fotoğrafında eldiven-yok alarmı" türü) yanlış alarm riskini de
# ayrıca azaltır.
PHOTO_THRESHOLDS = {
    "person": 0.42,
    "surgical-cap": 0.50,
    "no-surgical-cap": 0.37,
    "surgical-gloves": 0.54,
    "no-surgical-gloves": 0.49,
    "surgical-mask": 0.47,
    "surgical-gown": 0.51,
    "shoe-covers": 0.48,
    "surgical-scrubs": 0.35,
    "face-shield": 0.63,
    "no-facial-gear": 0.47,
    "no-medical-attire": 0.60,
    "goggles": 0.56,
    "coverall": 0.55,
}

# N-of-M ihlal onayı: son M örnekleme karesinin en az N'inde ihlal.
# Yangına (4/8) göre daha temkinli (6/10): kol/el oklüzyonu tek karelik
# sahte "eldiven yok" üretebilir; 4 Hz örneklemede bu ~1.5 sn kanıt demek.
DEFAULT_NOFM = (6, 10)

# Bir iz bu kadar örnekleme karesi boyunca hiç görünmezse unutulur.
TRACK_FORGET_LIMIT = 12

# Kişi kutusu, eldiven/gözlük gibi uçlarda kalan ekipmanları da kapsasın
# diye eşleştirme sırasında her yöne %15 genişletilir.
PERSON_BOX_EXPAND = 0.15

# Kenarı bundan kısa kutular elenir: 640px'lik karede ~12px'lik bir tespit
# (eldiven bile olsa) güvenilir sınıflandırılamaz; bu boyuttaki kutular
# pratikte hep doku/gölge kaynaklı yanlış alarmdır.
MIN_BOX_PX = 12

# Pozitif/negatif çakışma bastırma eşiği: aynı ekipmanın "var" ve "yok"
# tespiti aynı bölgeye düşerse (IoU bu değerin üstündeyse) model kararsız
# demektir; yalnızca yüksek güvenli olan tutulur (aşağıda _suppress_conflicts).
CONFLICT_IOU = 0.45

# Çizim renkleri (BGR)
COLOR_ITEM_OK = (0, 180, 0)        # takılı ekipman: ince yeşil
COLOR_CANDIDATE = (0, 220, 220)    # aday ihlal: sarı
COLOR_VIOLATION = (0, 0, 255)      # onaylı ihlal: kırmızı
COLOR_PERSON_OK = (0, 200, 0)      # tam ekipmanlı kişi: yeşil


def _iou(box_a, box_b):
    """İki (x1,y1,x2,y2) kutusunun kesişim/birleşim oranı."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / float(area_a + area_b - inter)


# ─────────────── DİLİMLİ TESPİT (küçük nesne güçlendirme) ───────────────
# NEDEN: en zayıf sınıflarımızın (no-surgical-gloves, no-surgical-cap,
# no-facial-gear) medyan kutu alanı görüntünün %0.9-1.6'sı. Tam kare 640px'e
# küçültülünce bu nesneler ~10-20 piksele düşüyor. Dilimli tespit (SAHI
# yaklaşımı) kareyi %20 örtüşmeli 2x2 parçaya böler ve HER PARÇAYI ayrı ayrı
# 640px'te modele sokar — küçük nesne, modelin gözünde ~2 kat büyür. Tam kare
# geçişi de korunur (kişi/önlük gibi büyük nesneler parçalara sığmaz);
# ardından örtüşen kutular tekilleştirilir. Bedeli ~5 kat çıkarım süresi —
# bu yüzden yalnızca FOTOĞRAF modunda açık (videoda zamansal fazlalık zaten
# kaçan kareyi telafi ediyor, hız kritik).
TILE_GRID = (2, 2)
TILE_OVERLAP = 0.20
# Kısa kenarı bundan küçük görüntülerde dilim, 640'ın altına iner ve
# büyütme kazancı interpolasyon bulanıklığına dönüşür — dilimleme atlanır.
TILE_MIN_SIDE = 500
# Dilim geçişlerinden SADECE küçük kutular kabul edilir (tam kare alanının
# en fazla %5'i). NEDEN — ÖLÇÜLMÜŞ DERS (2026-07-17, 535 görüntülük test):
# filtresiz dilimleme küçük sınıfları +3..+6 puan iyileştirirken BÜYÜK
# nesneleri çökertti (person mAP50 0.880->0.249, gown 0.897->0.571):
# dilim, kişiyi/önlüğü ortadan kesiyor, yarım gövde "person" diye yüksek
# güvenle tespit edilip yanlış alarma dönüşüyordu. Büyük nesnelerin görevi
# zaten TAM KARE geçişinin; dilimlerin tek görevi küçükleri yakalamak.
TILE_MAX_AREA_FRAC = 0.05
# İç dilim kenarına bu kadar yaklaşan kutu "kesilmiş nesne" sayılıp atılır
# (kare sınırındaki kenarlar hariç — orada kutu meşru olarak kenara değer).
TILE_EDGE_MARGIN = 4
# Dilim geçişlerinin model giriş boyutu. Tam kare 640'ta kalırken dilimi
# daha büyük koşmak, yakınlaştırma kazancını YALNIZCA küçük nesne geçişine
# uygular. ÖLÇÜLMÜŞ KARAR (2026-07-22, 535 görüntülük test):
#   • tam kareyi 768/960'a çıkarmak: küçük sınıflar +3..+7 ama büyük
#     sınıflar -3..-7 (face-shield 0.943->0.895, person 0.917->0.848)
#     ve genel mAP50-95 düşüyor -> REDDEDİLDİ
#   • 3x3 ızgara (dilim 640): 246px dilim 2.6x büyütülünce bulanıyor,
#     genel mAP50 0.790->0.786 -> REDDEDİLDİ
#   • 2x2 ızgara + dilim 768 (bu ayar): genel mAP50 0.790->0.792,
#     mAP50-95 0.471->0.472, en zayıf sınıf no-surgical-gloves +2.0
#     puan; hiçbir sınıf 0.2 puandan fazla gerilemedi -> KABUL
TILE_IMGSZ = 768


def _iter_tiles(frame):
    """%20 örtüşmeli 2x2 dilimleri (kirpim, x_ofset, y_ofset) olarak üretir."""
    h, w = frame.shape[:2]
    cols, rows = TILE_GRID
    tile_w = int(w / (cols - (cols - 1) * TILE_OVERLAP))
    tile_h = int(h / (rows - (rows - 1) * TILE_OVERLAP))
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))
    for r in range(rows):
        for c in range(cols):
            x1 = min(c * step_x, w - tile_w)
            y1 = min(r * step_y, h - tile_h)
            yield frame[y1:y1 + tile_h, x1:x1 + tile_w], x1, y1


def _tile_detections(model, frame, class_map, min_conf):
    """Dilim geçişlerini koşar ve YALNIZCA güvenilir KÜÇÜK tespitleri döner.

    Üç filtre (gerekçeler TILE_MAX_AREA_FRAC/TILE_EDGE_MARGIN yorumlarında;
    3. filtre için aşağıya bak):
      1) kutu alanı, tam kare alanının %5'inden büyükse ATILIR
      2) kutu, İÇ dilim kenarına değiyorsa ATILIR (kesilmiş nesne)
      3) yalnızca NEGATİF (ihlal) sınıflar kabul edilir — ÖLÇÜLMÜŞ KARAR
         (2026-07-17, 535 görüntülük test): dilim katkısı sınıf sınıf
         incelendiğinde İSTİSNASIZ bütün negatif sınıflar iyileşti
         (no-facial-gear +10.4, no-surgical-cap +2.8, no-surgical-gloves
         +2.6, no-medical-attire +2.5 mAP50 puanı) ve bütün pozitif
         sınıflar ya sabit kaldı ya hafif geriledi (mask -2.7, scrubs
         -2.8). Pozitif ekipman zaten tam kare geçişinde güçlü; dilimlerin
         katma değeri küçük İHLAL bölgelerini (açık el, çıplak baş,
         korumasız yüz) yakalamak. Bu yüzden dilimler sadece o işi yapar.
    Dönen kutular tam kare koordinatındadır.
    """
    h, w = frame.shape[:2]
    frame_area = float(h * w)
    detections = []
    for crop, ox, oy in _iter_tiles(frame):
        th, tw = crop.shape[:2]
        result = model.predict(crop, conf=min_conf, imgsz=TILE_IMGSZ, verbose=False)[0]
        for det in _parse_result(result, class_map, min_conf):
            if det["info"].role != "item" or det["info"].positive:
                continue   # filtre 3: yalnız negatif (ihlal) sınıflar
            x1, y1, x2, y2 = det["box"]  # dilim koordinatında
            if (x2 - x1) * (y2 - y1) > TILE_MAX_AREA_FRAC * frame_area:
                continue
            # İç kenar kontrolü: dilimin bu kenarı karenin kendi sınırı
            # DEĞİLSE ve kutu ona değiyorsa nesne kesilmiş demektir.
            if x1 <= TILE_EDGE_MARGIN and ox > 0:
                continue
            if y1 <= TILE_EDGE_MARGIN and oy > 0:
                continue
            if x2 >= tw - TILE_EDGE_MARGIN and ox + tw < w:
                continue
            if y2 >= th - TILE_EDGE_MARGIN and oy + th < h:
                continue
            det["box"] = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)
            detections.append(det)
    return detections


def _merge_duplicates(base_detections, extra_detections):
    """Dilim geçişinden gelen EK kutuları, ana (tam kare) kutulara karşı
    tekilleştirerek ekler. ANA KUTULARA ASLA DOKUNULMAZ.

    NEDEN TEK YÖNLÜ (ölçülmüş ders, 2026-07-17): ilk sürüm TÜM kutuları
    (tam kare dahil) birbirine karşı tekilleştiriyordu; bu, dilim kullanmayan
    moda göre pozitif sınıflarda küçük ama gereksiz gerilemeler yarattı
    (scrubs -3.3, shoe-covers -2.2 mAP50) çünkü tam karenin NMS'ten sağ
    çıkmış yakın-komşu kutuları da birbirini eliyordu. Dilimlerin görevi
    yalnızca EK (kaçmış küçük ihlal) kutusu getirmek; ana geçişin çıktısını
    değiştirmeye hakkı yok.

    İki kutu aynı sayılır: aynı ham sınıf + (IoU > 0.55 VEYA küçük kutunun
    %80'i büyüğün içinde). Kapsama kuralı şart çünkü dilimde kesilen yarım
    kutu, tam karedeki bütün kutuyla düşük IoU verir ama onun içinde kalır.
    """
    def _find_duplicates(det, others):
        found = []
        for k, other in enumerate(others):
            if det["info"].raw != other["info"].raw:
                continue
            iou = _iou(det["box"], other["box"])
            ix1 = max(det["box"][0], other["box"][0])
            iy1 = max(det["box"][1], other["box"][1])
            ix2 = min(det["box"][2], other["box"][2])
            iy2 = min(det["box"][3], other["box"][3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            area_b = (other["box"][2] - other["box"][0]) * (other["box"][3] - other["box"][1])
            containment = inter / float(min(area_a, area_b)) if inter else 0.0
            if iou > 0.55 or containment > 0.80:
                found.append(k)
        return found

    # Çift bulunursa GÜVENİ YÜKSEK olan kazanır (yakınlaştırılmış dilimde
    # küçük ihlal kutusu genelde hem daha isabetli hem daha güvenlidir;
    # tek yönlü "hep tam kare kazanır" kuralı denendi, negatif kazancın
    # yarısını geri veriyordu). Dilimler yalnızca negatif sınıf ürettiği
    # için bu değiştirme pozitif sınıf kutularına yine dokunAMAZ.
    merged = list(base_detections)
    for det in sorted(extra_detections, key=lambda d: -d["conf"]):
        dups = _find_duplicates(det, merged)
        if not dups:
            merged.append(det)
        elif det["conf"] > max(merged[k]["conf"] for k in dups):
            merged = [d for k, d in enumerate(merged) if k not in dups]
            merged.append(det)
    return merged


def _parse_result(result, class_map, min_conf, offset=(0, 0)):
    """Bir predict sonucunu {'info','conf','box'} listesine çevirir.

    offset: dilim kırpımının tam karedeki sol-üst köşesi — kutular tam kare
    koordinatına taşınır. Eşik/boyut filtreleri BURADA UYGULANMAZ (birleştirme
    sonrasında bir kez uygulanır, çift filtre olmasın).
    """
    ox, oy = offset
    detections = []
    if result.boxes is None:
        return detections
    for i in range(len(result.boxes)):
        info = class_map.get(int(result.boxes.cls[i]))
        if info is None:
            continue
        conf = float(result.boxes.conf[i])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
        detections.append({"info": info, "conf": conf,
                           "box": (x1 + ox, y1 + oy, x2 + ox, y2 + oy)})
    return detections


def _suppress_conflicts(detections):
    """Aynı ekipmanın POZİTİF ve NEGATİF tespiti aynı bölgeye düşerse
    düşük güvenli olanı eler.

    NEDEN GEREKLİ: pozitif+negatif sınıf şemasında ("surgical-gloves" /
    "no-surgical-gloves") model sınır durumlarında AYNI ele iki kutu
    birden verebilir — biri "eldiven var" biri "eldiven yok". Standart NMS
    bunu ELEYEMEZ çünkü NMS sınıf içinde çalışır, bunlar farklı sınıflar.
    İkisi birden kalırsa hem ekranda çelişkili kutular görünür hem de
    sahte "yok" tespiti N-of-M penceresini kirletir. Kural basit: aynı
    kanonik ekipman + zıt kutup + IoU > CONFLICT_IOU → güveni yüksek olan
    kazanır (model hangisinden daha eminse ona inanıyoruz).
    """
    dropped = set()
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            a, b = detections[i], detections[j]
            info_a, info_b = a["info"], b["info"]
            if info_a.role != "item" or info_b.role != "item":
                continue
            if info_a.item != info_b.item or info_a.positive == info_b.positive:
                continue
            if _iou(a["box"], b["box"]) <= CONFLICT_IOU:
                continue
            dropped.add(j if a["conf"] >= b["conf"] else i)
    return [d for k, d in enumerate(detections) if k not in dropped]


def _draw_box_with_label(canvas, box, color, thickness, label):
    """Kutu + etiket çizer; etiket üstte yer yoksa kutunun İÇİNE alınır.

    (Kare sınırı taşma sorunu ve çözümü FireAndSmokeVideo.py'de ayrıntılı
    anlatılıyor; aynı davranış buraya taşındı.)
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_h = th + 8

    if y1 - label_h >= 0:
        top = y1 - label_h
        text_y = y1 - 5
    else:
        top = y1
        text_y = y1 + th + 3

    cv2.rectangle(canvas, (x1, top), (x1 + tw + 4, top + label_h), color, -1)
    cv2.putText(canvas, label, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


class TrackState:
    """Tek bir izin (kişi veya negatif tespit) zamansal geçmişi.

    windows: {ekipman: deque(maxlen=M)} — her örnekleme karesi için
    1=ihlal görüldü, 0=uyumlu. "direct" modda tek anahtar kullanılır
    (negatif sınıfın kendisi), "person" modda zorunlu her ekipman için
    ayrı pencere tutulur (kişi aynı anda hem eldivensiz hem gözlüksüz
    olabilir; ikisi bağımsız onaylanır).
    """

    def __init__(self, track_id, items, window_size):
        self.track_id = track_id
        self.windows = {item: deque(maxlen=window_size) for item in items}
        self.confirmed = set()        # N-of-M'i sağlamış ekipmanlar
        self.evidence_saved = set()   # kanıtı yazılmış ekipmanlar
        self.miss_count = 0           # üst üste kaç karedir görünmüyor
        self.last_box = None
        self.last_conf = 0.0
        self.last_missing = set()     # bu karede eksik görünen ekipmanlar


class MedicalPPEMonitor:
    """Video karelerini tek tek alır, mimarideki 1-5 katmanlarını uygular.

    Kullanım (arayüzden bağımsız):
        monitor = MedicalPPEMonitor()          # modeli kendisi bulur
        for frame in frames:
            drawn, new_events = monitor.process_frame(frame, video_sec)
    """

    def __init__(self, model_path=None, thresholds=None, default_conf=None,
                 nofm=None, required_items=None, record_dir=None,
                 csv_path=None, model=None):
        if model is not None:
            # Streamlit modeli bir kez yükleyip (cache) her oturum için
            # yeni izleyici kurar; hazır nesneyi kabul edip ByteTrack
            # hafızasını sıfırlıyoruz (detay FireAndSmokeVideo.py'de).
            self.model = model
            try:
                for tr in self.model.predictor.trackers:
                    tr.reset()
            except Exception:
                pass  # henüz hiç track çağrılmadıysa predictor yoktur
        else:
            from ultralytics import YOLO
            if model_path is None:
                model_path = find_weights()
            self.model = YOLO(model_path)

        self.thresholds = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.default_conf = default_conf if default_conf is not None else DEFAULT_CONF
        self.nofm = nofm or DEFAULT_NOFM

        # ─── Sınıf çözümleme + mod seçimi (mimarideki katman 2-3) ───
        self.class_map = resolve_model_classes(self.model.names)
        self.neg_items = {ci.item for ci in self.class_map.values()
                          if ci.role == "item" and not ci.positive}
        self.pos_items = {ci.item for ci in self.class_map.values()
                          if ci.role == "item" and ci.positive}
        self.has_person = any(ci.role == "person" for ci in self.class_map.values())

        if self.neg_items:
            self.mode = "direct"
        elif self.has_person and self.pos_items:
            self.mode = "person"
        else:
            self.mode = "presence"

        # "person" modunda denetlenecek zorunlu ekipman seti: varsayılan,
        # modelin tanıyabildiği TÜM pozitif ekipmanlar. (Modelin hiç
        # göremeyeceği bir şeyi zorunlu tutmak her kişiyi ihlalli yapardı.)
        self.required_items = set(required_items) if required_items else set(self.pos_items)

        self.record_dir = Path(record_dir or (BASE_DIR / "IhlalKayitlari"))
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(csv_path or (BASE_DIR / "medikal_ihlal_log.csv"))

        # Aktif izler: {track_id: TrackState}
        self.tracks: dict[int, TrackState] = {}

    # ─────────────── KATMAN 1+2: track + sınıf bazlı eşik ───────────────
    def _detect(self, frame):
        """Bir karede takipli tespit yapar; eşik altını eler.

        Dönen liste elemanı: {"track_id", "info": ClassInfo, "conf", "box"}
        """
        all_conf = list(self.thresholds.values()) + [self.default_conf]
        result = self.model.track(
            frame,
            persist=True,
            conf=min(all_conf),   # ön eleme: en düşük eşikten azını hiç getirme
            imgsz=640,            # eğitimle aynı çözünürlük → tutarlı skorlar
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections
        for i in range(len(result.boxes)):
            cls_idx = int(result.boxes.cls[i])
            info = self.class_map.get(cls_idx)
            if info is None:
                continue
            conf = float(result.boxes.conf[i])
            if conf < self.thresholds.get(info.raw, self.default_conf):
                continue
            x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
            # Minik kutu filtresi: bu boyutta sınıf kararı güvenilmez (ayar
            # MIN_BOX_PX'in yanındaki yorumda).
            if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
                continue
            track_id = None
            if result.boxes.id is not None:
                track_id = int(result.boxes.id[i])
            detections.append({
                "track_id": track_id,
                "info": info,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })
        # Pozitif/negatif çelişkilerini bastır ("eldiven var" ile "eldiven
        # yok" aynı ele düşmüşse yüksek güvenli olan kalır).
        return _suppress_conflicts(detections)

    # ─────────────── KATMAN 3: ekipman → kişi eşleştirme ───────────────
    @staticmethod
    def _find_owner(item_box, persons):
        """Bir ekipman kutusunu en uygun kişiye atar.

        Kural: ekipman kutusunun merkezi, %15 genişletilmiş kişi kutusunun
        içindeyse o kişi adaydır; birden çok aday varsa merkezi ekipmana
        en yakın kişi kazanır. (Eldiven elde, yani kişi kutusunun tam
        kenarında durur — genişletme bu yüzden var.)
        """
        cx = (item_box[0] + item_box[2]) / 2
        cy = (item_box[1] + item_box[3]) / 2
        best, best_dist = None, None
        for person in persons:
            x1, y1, x2, y2 = person["box"]
            ex = (x2 - x1) * PERSON_BOX_EXPAND
            ey = (y2 - y1) * PERSON_BOX_EXPAND
            if not (x1 - ex <= cx <= x2 + ex and y1 - ey <= cy <= y2 + ey):
                continue
            pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
            if best_dist is None or dist < best_dist:
                best, best_dist = person, dist
        return best

    # ─────────────── KATMAN 4: N-of-M zamansal filtre ───────────────
    def _update_tracks(self, observations):
        """Her izin pencerelerine bu karenin gözlemini işler.

        observations: {track_id: {"missing": set, "box": ..., "conf": ...}}
        — bu karede GÖRÜNEN izlerin ihlal durumu. (\"direct\" modda missing
        negatif sınıfın ekipman adıdır; \"person\" modda kişide eksik
        kalan zorunlu ekipmanların kümesidir.)

        Döner: [(track, yeni onaylanan ekipman seti), ...]
        AŞAMA A/B ayrımının gerekçesi FireAndSmokeVideo._update_tracks'te
        ayrıntılı anlatılıyor; aynı iskelet burada da geçerli.
        """
        required_n, window_size = self.nofm

        # ─── AŞAMA A: bu karede görülen izleri işle ───
        for track_id, obs in observations.items():
            if track_id not in self.tracks:
                items = obs["items"]
                self.tracks[track_id] = TrackState(track_id, items, window_size)
            track = self.tracks[track_id]
            track.miss_count = 0
            track.last_box = obs["box"]
            track.last_conf = obs["conf"]
            track.last_missing = obs["missing"]
            for item, window in track.windows.items():
                window.append(1 if item in obs["missing"] else 0)

        # ─── AŞAMA B: görünmeyenleri say, unut, N-of-M kontrolü ───
        newly_confirmed = []
        for track_id, track in list(self.tracks.items()):
            if track_id not in observations:
                track.miss_count += 1
                if track.miss_count > TRACK_FORGET_LIMIT:
                    del self.tracks[track_id]
                    continue
                # Kişi bu karede görünmediyse ekipmanları hakkında YENİ
                # bilgi yok; pencereye 0/1 yazmıyoruz (yangındaki gibi
                # "ıska=0" yazmak burada kişiyi haksız yere aklardı).

            new_items = set()
            for item, window in track.windows.items():
                if item not in track.confirmed and sum(window) >= required_n:
                    track.confirmed.add(item)
                    new_items.add(item)
            if new_items:
                newly_confirmed.append((track, new_items))
        return newly_confirmed

    # ─────────────── KATMAN 5: kanıt kaydı (fotoğraf + CSV) ───────────────
    def _save_evidence(self, track, items, drawn_frame, video_sec):
        """Onaylanan ihlal için anotasyonlu kareyi diske, olayı CSV'ye yazar.

        Aynı iz + aynı ekipman için bir kez çağrılır (evidence_saved
        kümesi); alarm/bildirim bilinçli olarak yok — görev kanıt
        biriktirmek (FireAndSmoke ile aynı tercih).
        """
        items_text = "+".join(sorted(items))
        timestamp = datetime.now()
        file_name = f"ihlal_{items_text}_iz{track.track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(self.record_dir / file_name), drawn_frame)

        is_new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(["Tarih_Saat", "Video_sn", "Iz_ID",
                                 "Eksik_Ekipman", "Guven_%", "Kanit_Dosyasi"])
            writer.writerow([
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{video_sec:.1f}",
                track.track_id,
                items_text,
                f"{track.last_conf * 100:.1f}",
                file_name,
            ])
        track.evidence_saved.update(items)
        return file_name

    # ─────────────── ÇİZİM: durumu görselleştir ───────────────
    def _draw_frame(self, frame, detections):
        """Takılı ekipman yeşil-ince, aday ihlal sarı, onaylı ihlal kırmızı-kalın."""
        canvas = frame.copy()
        for det in detections:
            info = det["info"]
            conf = det["conf"]
            box = det["box"]
            track = self.tracks.get(det["track_id"]) if det["track_id"] is not None else None

            if info.role == "person" and self.mode == "person":
                if track is not None and track.confirmed:
                    color, thickness = COLOR_VIOLATION, 3
                    label = f"kisi#{track.track_id} IHLAL: {','.join(sorted(track.confirmed))}"
                elif track is not None and track.last_missing:
                    color, thickness = COLOR_CANDIDATE, 2
                    label = f"kisi#{track.track_id} eksik?: {','.join(sorted(track.last_missing))}"
                else:
                    color, thickness = COLOR_PERSON_OK, 1
                    tid = track.track_id if track else "?"
                    label = f"kisi#{tid} TAM"
            elif info.role == "item" and not info.positive:
                # "direct" mod: negatif sınıf tespiti = ihlal adayı/onaylısı
                confirmed = track is not None and track.confirmed
                color = COLOR_VIOLATION if confirmed else COLOR_CANDIDATE
                thickness = 3 if confirmed else 1
                status = "ONAYLI" if confirmed else "aday"
                label = f"{info.item or info.raw} yok {conf:.2f} [{status}]"
            else:
                # takılı ekipman / kişi (direct modda) / bilinmeyen sınıf
                color, thickness = COLOR_ITEM_OK, 1
                label = f"{info.item or info.raw} {conf:.2f}"

            _draw_box_with_label(canvas, box, color, thickness, label)
        return canvas

    # ─────────────── DIŞ DÜNYAYA AÇILAN TEK KAPI ───────────────
    def process_frame(self, frame, video_sec: float):
        """Bir kareyi uçtan uca işler; arayüzlerin çağırdığı tek fonksiyon.

        Dönenler:
          drawn_frame : kutuları çizilmiş kare
          new_events  : bu karede İLK KEZ onaylanan ihlaller;
                        her öğe {"track_id", "missing", "conf", "file_name"}
        """
        detections = self._detect(frame)

        # ─── Mod'a göre bu karenin ihlal gözlemlerini çıkar ───
        observations = {}
        if self.mode == "person":
            persons = [d for d in detections
                       if d["info"].role == "person" and d["track_id"] is not None]
            ppe_items = [d for d in detections
                         if d["info"].role == "item" and d["info"].positive]

            present = {p["track_id"]: set() for p in persons}
            for ppe in ppe_items:
                owner = self._find_owner(ppe["box"], persons)
                if owner is not None:
                    present[owner["track_id"]].add(ppe["info"].item)

            for person in persons:
                tid = person["track_id"]
                observations[tid] = {
                    "missing": self.required_items - present[tid],
                    "items": self.required_items,
                    "box": person["box"],
                    "conf": person["conf"],
                }
        elif self.mode == "direct":
            for det in detections:
                info = det["info"]
                if info.role == "item" and not info.positive and det["track_id"] is not None:
                    item = info.item or info.raw
                    observations[det["track_id"]] = {
                        "missing": {item},
                        "items": {item},
                        "box": det["box"],
                        "conf": det["conf"],
                    }
        # "presence" modunda ihlal çıkarımı yapılamaz; observations boş kalır.

        newly_confirmed = self._update_tracks(observations)
        drawn_frame = self._draw_frame(frame, detections)

        new_events = []
        for track, items in newly_confirmed:
            unsaved = items - track.evidence_saved
            if unsaved:
                file_name = self._save_evidence(track, unsaved, drawn_frame, video_sec)
                new_events.append({
                    "track_id": track.track_id,
                    "missing": "+".join(sorted(unsaved)),
                    "conf": track.last_conf,
                    "file_name": file_name,
                })
        return drawn_frame, new_events

    def get_status(self):
        """Arayüzün durum satırı için kısa özet."""
        violations = sum(1 for t in self.tracks.values() if t.confirmed)
        return {"active_tracks": len(self.tracks), "violations": violations}


# ─────────────────────────── TEK FOTOĞRAF MODU ───────────────────────────
def process_photo(model, frame, thresholds=None, default_conf=None, tiled=True):
    """Tek bir fotoğrafta tespit + uyum özeti (Streamlit'in 'Görsel' sekmesi).

    Zamansal filtre (N-of-M) tek karede uygulanamaz; bu yüzden fotoğraf
    modunda yalnızca eşik uygulanır ve anlık uyum durumu raporlanır.
    TTA (augment=True) her zaman açık: tek fotoğrafta hız kaygısı yok,
    isabet önemli (FireAndSmoke'taki tercihle aynı).

    tiled=True: dilimli tespit (küçük nesne güçlendirme — gerekçe ve ölçülen
    etkisi _iter_tiles üstündeki blokta ve README'de). Görüntü küçükse
    (kısa kenar < TILE_MIN_SIDE) kendiliğinden atlanır.

    Dönenler: (drawn_frame, counts, violations)
      counts     : {"eldiven": 2, "kisi": 1, ...} sınıf sayıları
      violations : ["kisi#1 eksik: gozluk", ...] — modelin izin verdiği
                   modda anlık ihlal özeti (yoksa boş liste)
    """
    # Fotoğraf hattı kendi (TTA+dilim üzerinde kalibre) eşiklerini kullanır —
    # gerekçe PHOTO_THRESHOLDS tanımının üstünde.
    thresholds = dict(PHOTO_THRESHOLDS) if thresholds is None else dict(thresholds)
    default_conf = default_conf if default_conf is not None else DEFAULT_CONF

    class_map = resolve_model_classes(model.names)
    min_conf = min(list(thresholds.values()) + [default_conf])

    # 1) TAM KARE geçişi (TTA açık): büyük nesneler + genel bağlam.
    result = model.predict(frame, conf=min_conf, imgsz=640,
                           augment=True, verbose=False)[0]
    detections = _parse_result(result, class_map, min_conf)

    # 2) DİLİMLİ geçişler (küçük nesne güçlendirme; yalnız küçük/kesilmemiş
    #    kutular kabul edilir — gerekçe _tile_detections'ta). Dilimlerde TTA
    #    kapalı: dilim zaten yakınlaştırma sağlıyor.
    if tiled and min(frame.shape[:2]) >= TILE_MIN_SIDE:
        extra = _tile_detections(model, frame, class_map, min_conf)
        detections = _merge_duplicates(detections, extra)

    # 3) Eşik + boyut filtreleri (birleştirmeden SONRA, tek sefer)
    detections = [
        d for d in detections
        if d["conf"] >= thresholds.get(d["info"].raw, default_conf)
        and (d["box"][2] - d["box"][0]) >= MIN_BOX_PX
        and (d["box"][3] - d["box"][1]) >= MIN_BOX_PX
    ]
    detections = _suppress_conflicts(detections)

    pos_items = {ci.item for ci in class_map.values() if ci.role == "item" and ci.positive}
    neg_items = {ci.item for ci in class_map.values() if ci.role == "item" and not ci.positive}
    has_person = any(ci.role == "person" for ci in class_map.values())

    # Mod seçimi MedicalPPEMonitor ile BİREBİR AYNI kural: negatif sınıflar
    # varsa direct mod kazanır. NEDEN ÖNEMLİ: negatif sınıflar zaten "X yok"
    # diye açıkça bildiriyor; bunun ÜSTÜNE kişi-ekipman kutu eşleştirmesiyle
    # "kişide görünmeyen her ekipman eksiktir" de dersek aynı ihlali iki kez
    # (biri doğru sinyalle, biri sahte "görünmüyor" varsayımıyla) raporlarız
    # — üstelik tek fotoğrafta ekipmanın kadraj dışında kalması (ör. ayakkabı
    # kadraja hiç girmemiş) gerçek bir ihlal değildir.
    if neg_items:
        mode = "direct"
    elif has_person and pos_items:
        mode = "person"
    else:
        mode = "presence"

    canvas = frame.copy()
    counts = {}
    violations = []

    persons = [d for d in detections if d["info"].role == "person"]
    ppe_items = [d for d in detections if d["info"].role == "item" and d["info"].positive]

    # Kişi bazlı anlık uyum SADECE "person" modda (modelde negatif sınıf yok,
    # kişi + pozitif ekipman var — eksik ekipmanı ancak kutu eşleştirmesiyle
    # çıkarabiliriz).
    person_missing = {}
    if mode == "person":
        for idx, person in enumerate(persons, start=1):
            present = set()
            for ppe in ppe_items:
                if MedicalPPEMonitor._find_owner(ppe["box"], [person]) is not None:
                    present.add(ppe["info"].item)
            missing = pos_items - present
            person_missing[id(person)] = (idx, missing)
            if missing:
                violations.append(f"kisi#{idx} eksik: {', '.join(sorted(missing))}")

    for det in detections:
        info = det["info"]
        name = info.item or info.raw
        if info.role == "person":
            name = "kisi"
        elif info.role == "item" and not info.positive:
            # Pozitif ve negatif tespiti AYNI anahtar altında saymıyoruz:
            # "eldiven: 2" hem "2 eldiven var" hem "1 var+1 yok" anlamına
            # gelebilirdi — kullanıcıyı yanıltırdı. Ayrı anahtar (ör.
            # "eldiven-YOK") ihlal listesindeki "X yok" ifadesiyle de tutarlı.
            name = f"{name}-YOK"
        counts[name] = counts.get(name, 0) + 1

        if info.role == "person" and id(det) in person_missing:
            idx, missing = person_missing[id(det)]
            if missing:
                color, thickness = COLOR_VIOLATION, 2
                label = f"kisi#{idx} eksik: {','.join(sorted(missing))}"
            else:
                color, thickness = COLOR_PERSON_OK, 1
                label = f"kisi#{idx} TAM"
        elif info.role == "item" and not info.positive:
            color, thickness = COLOR_VIOLATION, 2
            label = f"{info.item or info.raw} yok {det['conf']:.2f}"
            violations.append(f"{info.item or info.raw} yok ({det['conf']:.2f})")
        else:
            color, thickness = COLOR_ITEM_OK, 1
            label = f"{name} {det['conf']:.2f}"
        _draw_box_with_label(canvas, det["box"], color, thickness, label)

    return canvas, counts, violations


# ─────────────────── GERÇEK HATTIN TEST SETİ ÖLÇÜMÜ ───────────────────
def evaluate_pipeline(model, img_dir, lbl_dir, tiled=True, progress_cb=None):
    """Fotoğraf hattını (TTA + dilimli tespit) etiketli bir set üzerinde ölçer.

    NEDEN VAR: model.val() modelin HAM halini ölçer; üretimde ise fotoğraf
    TTA + dilimli tespit + tekilleştirme katmanlarından geçiyor. Kullanıcıya
    gösterilen metrik, kullanıcının gerçekten çalıştırdığı hattın metriği
    olmalı (plaka modülündeki tercihle aynı ilke). Eşik/boyut filtreleri
    burada KAPALI tutulur: AP matematiği tüm güven aralığına bakar, eşik
    uygulamak eğriyi keserdi.

    Ölçüm matematiği Ultralytics'in kendisinden (ap_per_class + box_iou);
    IoU eşleştirme algoritması val() ile birebir aynı.

    progress_cb: her görüntüden sonra (islenen, toplam) ile çağrılır
    (Streamlit ilerleme çubuğu için).

    Döner: {"precision","recall","map50","map50_95","per_class":[...]}
    — app.py'deki _metrics_from_val ile aynı şekil.
    """
    import numpy as np
    import torch
    from ultralytics.utils.metrics import ap_per_class, box_iou

    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    min_conf = 0.05          # AP eğrisi için düşük taban; eşikler uygulanmaz
    iouv = np.linspace(0.5, 0.95, 10)

    class_map = resolve_model_classes(model.names)
    raw_to_idx = {v: k for k, v in model.names.items()}

    img_paths = sorted(p for p in img_dir.glob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []

    for done, img_path in enumerate(img_paths, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        gt_boxes, gt_cls = [], []
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                gt_boxes.append([cx - bw / 2, cy - bh / 2,
                                 cx + bw / 2, cy + bh / 2])
                gt_cls.append(int(parts[0]))
        all_target_cls.extend(gt_cls)

        # Üretim fotoğraf hattı: tam kare + TTA, ardından dilimler
        result = model.predict(frame, conf=min_conf, imgsz=640,
                               augment=True, verbose=False)[0]
        dets = _parse_result(result, class_map, min_conf)
        if tiled and min(h, w) >= TILE_MIN_SIDE:
            extra = _tile_detections(model, frame, class_map, min_conf)
            dets = _merge_duplicates(dets, extra)

        if progress_cb:
            progress_cb(done, len(img_paths))
        if not dets:
            continue

        pred_boxes = torch.tensor([list(d["box"]) for d in dets],
                                  dtype=torch.float32)
        pred_cls = np.array([raw_to_idx[d["info"].raw] for d in dets])
        confs = np.array([d["conf"] for d in dets])

        # val() ile aynı eşleştirme: IoU matrisi -> eşik başına, IoU'su en
        # yüksek eşleşmeler önce; her tahmin ve her GT en fazla 1 kez.
        tp = np.zeros((len(dets), len(iouv)), dtype=bool)
        if gt_boxes:
            iou = box_iou(torch.tensor(gt_boxes, dtype=torch.float32),
                          pred_boxes).numpy()
            iou = iou * (np.array(gt_cls)[:, None] == pred_cls[None, :])
            for ti, thr in enumerate(iouv):
                matches = np.array(np.nonzero(iou >= thr)).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        order = iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        matches = matches[order]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    tp[matches[:, 1].astype(int), ti] = True

        all_tp.append(tp)
        all_conf.extend(confs)
        all_pred_cls.extend(pred_cls)

    if not all_tp:
        raise RuntimeError(f"{img_dir} altında ölçülecek görüntü bulunamadı.")

    tp_arr = np.concatenate(all_tp, axis=0)
    _, _, p, r, f1, ap, uniq, *_ = ap_per_class(
        tp_arr, np.array(all_conf), np.array(all_pred_cls),
        np.array(all_target_cls))

    per_class = []
    for i, cls_idx in enumerate(uniq):
        pc_p, pc_r = float(p[i]), float(r[i])
        per_class.append({
            "Sınıf": model.names[int(cls_idx)],
            "Precision": pc_p,
            "Recall": pc_r,
            "F1": (2 * pc_p * pc_r / (pc_p + pc_r)) if (pc_p + pc_r) > 0 else 0.0,
            "mAP50": float(ap[i, 0]),
            "mAP50-95": float(ap[i].mean()),
        })
    return {
        "precision": float(p.mean()),
        "recall": float(r.mean()),
        "map50": float(ap[:, 0].mean()),
        "map50_95": float(ap.mean()),
        "per_class": per_class,
    }


# ─────────────────────────── KOMUT SATIRI TESTİ ───────────────────────────
# Streamlit olmadan hızlı deneme: python MedicalPPEVideo.py video.mp4
# (0 verirsen bilgisayarın webcam'ini açar)
if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    monitor = MedicalPPEMonitor()
    print(f"[BILGI] Mod: {monitor.mode}   Zorunlu ekipman: {sorted(monitor.required_items)}")
    if monitor.mode == "presence":
        print("[UYARI] Modelde ne kişi ne de no_* sınıfı var; ihlal çıkarımı "
              "yapılamaz, yalnızca tespitler gösterilir.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Kaynak açılamadı: {source}")

    FRAME_SKIP = 5  # her 5 kareden 1'ini işle (app.py dosya moduyla aynı seyreltme)
    frame_no = -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % FRAME_SKIP != 0:
            continue
        drawn, events = monitor.process_frame(frame, frame_no / fps)
        for event in events:
            print(f"[ONAY] iz {event['track_id']} eksik: {event['missing']} "
                  f"(guven {event['conf']:.2f}) -> {event['file_name']}")
        cv2.imshow("Tibbi PPE Izleme - cikis: Q", drawn)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
