# -*- coding: utf-8 -*-
"""
FireAndSmokeVideo.py — Gerçek zamanlı yangın/duman izleme çekirdeği
====================================================================

BU MODÜLÜN GENEL MİMARİSİ (veri bir kareden kayda kadar şu katmanlardan geçer):

    KAYNAK (video dosyası / IP kamera / webcam)
       │  her kare işlenmez; app.py tarafı kare atlama (dosya) veya
       │  zaman bazlı örnekleme (canlı) ile kareleri seyreltir
       ▼
    1) YOLO TRACK  ── model.track(persist=True)
       │  Sıradan predict'ten farkı: ByteTrack algoritması her nesneye
       │  kareler arasında KALICI bir kimlik (track id) verir. Böylece
       │  "5 numaralı duman bulutu 12 karedir görünüyor" diyebiliriz.
       ▼
    2) SINIF BAZLI GÜVEN EŞİĞİ
       │  fire  → yüksek eşik (parlak/turuncu şeylere yanlış alarm vermesin)
       │  smoke → düşük eşik  (duman doğası gereği düşük skor alır, kaçmasın)
       ▼
    3) N-of-M ZAMANSAL FİLTRE  (bu modülün asıl katma değeri)
       │  Tek karelik tespit güvenilmez: far yansıması, turuncu kıyafet,
       │  parıltı tek karede "fire" çıkabilir. Gerçek yangın ise KALICIDIR.
       │  Kural: bir track id, son M örnekleme karesinin en az N'inde
       │  eşiği geçen bir tespitle görüldüyse "ONAYLANMIŞ" sayılır.
       │  Fire için kısa pencere (hızlı tepki), smoke için uzun pencere
       │  (daha fazla kanıt) kullanıyoruz.
       ▼
    4) ONAY ANINDA KANIT KAYDI  (alarm/bildirim YOK — bilinçli tercih)
          • Anotasyonlu tam kare fotoğraf → YanginKayitlari/ klasörüne
          • CSV satırı (tarih, video sn, sınıf, güven, iz id, dosya adı)
          Aynı track id için kanıt SADECE BİR KEZ alınır; yoksa aynı
          yangın her karede yeni fotoğraf üretip diski doldurur.

EKRANDAKİ RENK DİLİ:
    • İnce sarı kutu   = ADAY   (tespit var ama henüz N-of-M onayı yok)
    • Kalın kırmızı    = ONAYLI fire
    • Kalın turuncu    = ONAYLI smoke
Bu ayrım kullanıcıya "model bir şey gördü" ile "sistem bundan emin"
arasındaki farkı canlı canlı gösterir.

Bu modül arayüz İÇERMEZ: Streamlit (app.py) veya komut satırı (main())
buradaki YanginDumanIzleyici sınıfını kullanır. Böylece izleme mantığı
tek yerde yaşar, her arayüz aynı davranışı gösterir.
"""

import csv
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

# En son eğitimin best.pt'sini bulan fonksiyon FireAndSmokeDetection.py'de
# zaten var; kopyalamak yerine oradan alıyoruz (tek doğru kaynak ilkesi).
from FireAndSmokeDetection import find_latest_best_weights

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────── AYARLAR ───────────────────────────
# Sınıf bazlı güven eşikleri (neden farklı olduklarının açıklaması yukarıda).
VARSAYILAN_ESIKLER = {"fire": 0.45, "smoke": 0.30}

# N-of-M pencere ayarı: {sınıf: (N=en az kaç isabet, M=pencere boyu)}.
# Örnekleme hızımız saniyede ~4 kare olduğundan fire 4/8 ≈ "2 saniyelik
# pencerede 1 saniyelik kanıt", smoke 5/10 ≈ "2.5 saniyede 1.25 sn kanıt".
VARSAYILAN_NOFM = {"fire": (4, 8), "smoke": (5, 10)}

# Bir iz bu kadar örnekleme karesi boyunca hiç görünmezse unutulur
# (nesne sahneden çıktı ya da yanlış alarmdı demektir).
IZ_UNUTMA_SINIRI = 12

# Çizim renkleri (BGR): aday sarı, onaylı fire kırmızı, onaylı smoke turuncu.
RENK_ADAY = (0, 220, 220)
RENK_ONAYLI = {"fire": (0, 0, 255), "smoke": (0, 140, 255)}


class IzDurumu:
    """Tek bir track id'nin zamansal geçmişini tutar.

    pencere: son M örnekleme karesinin isabet kaydı (1=bu karede eşiği
    geçen tespit vardı, 0=yoktu). deque(maxlen=M) sayesinde eski kayıtlar
    kendiliğinden düşer — kayan pencere (sliding window) budur.
    """

    def __init__(self, iz_id: int, sinif_adi: str, pencere_boyu: int):
        self.iz_id = iz_id
        self.sinif_adi = sinif_adi
        self.pencere = deque(maxlen=pencere_boyu)
        self.onaylandi = False        # N-of-M sağlandı mı?
        self.kanit_alindi = False     # fotoğraf+CSV bir kez yazıldı mı?
        self.kayip_sayaci = 0         # üst üste kaç karedir görünmüyor
        self.son_kutu = None          # en son görüldüğü (x1,y1,x2,y2)
        self.son_guven = 0.0


class YanginDumanIzleyici:
    """Video karelerini tek tek alır, mimarideki 1-4 katmanlarını uygular.

    Kullanım (arayüzden bağımsız):
        izleyici = YanginDumanIzleyici(model_yolu)
        for kare in kareler:
            cizili_kare, yeni_onaylar = izleyici.kare_isle(kare, video_sn)
    """

    def __init__(self, model_yolu=None, esikler=None, nofm=None,
                 kayit_klasoru=None, csv_yolu=None, model=None):
        # Model burada import ediliyor ki modülü sadece ayarları için
        # import edenler (ör. testler) ultralytics yüklemek zorunda kalmasın.
        from ultralytics import YOLO

        if model is not None:
            # Streamlit gibi arayüzler modeli bir kez yükleyip (cache)
            # her izleme oturumu için yeni izleyici kurar; modeli yeniden
            # diskten yüklememek için hazır nesneyi kabul ediyoruz.
            self.model = model
            # Aynı model nesnesi önceki oturumda track için kullanıldıysa
            # ByteTrack'in hafızasında eski izler kalır; yeni videonun ilk
            # kareleri yanlışlıkla onlara eşleşmesin diye sıfırlıyoruz.
            try:
                for tr in self.model.predictor.trackers:
                    tr.reset()
            except Exception:
                pass  # henüz hiç track çağrılmadıysa predictor yoktur, sorun değil
        else:
            if model_yolu is None:
                model_yolu = find_latest_best_weights()
            self.model = YOLO(model_yolu)
        self.esikler = esikler or dict(VARSAYILAN_ESIKLER)
        self.nofm = nofm or dict(VARSAYILAN_NOFM)

        # Kanıt fotoğraflarının klasörü ve olay CSV'si
        self.kayit_klasoru = Path(kayit_klasoru or (BASE_DIR / "YanginKayitlari"))
        self.kayit_klasoru.mkdir(parents=True, exist_ok=True)
        self.csv_yolu = Path(csv_yolu or (BASE_DIR / "yangin_olay_log.csv"))

        # Aktif izler: {track_id: IzDurumu}
        self.izler: dict[int, IzDurumu] = {}

    # ─────────────── KATMAN 1+2: track + sınıf bazlı eşik ───────────────
    def _tespit_et(self, kare):
        """Bir karede takipli tespit yapar, eşik altını eler.

        Dönen liste: (iz_id, sinif_adi, guven, (x1,y1,x2,y2)) dörtlüleri.
        iz_id None olabilir (tracker yeni nesneye henüz id atamadıysa);
        bunları zamansal filtreye sokamayacağımız için sadece ekranda
        aday olarak gösterir, sayıma katmayız.
        """
        sonuc = self.model.track(
            kare,
            persist=True,                 # iz durumunu kareler arasında koru
            conf=min(self.esikler.values()),  # ön eleme: en düşük eşikten azını hiç getirme
            imgsz=640,                    # eğitimle aynı çözünürlük → tutarlı skorlar
            verbose=False,
        )[0]

        tespitler = []
        if sonuc.boxes is None:
            return tespitler
        for i in range(len(sonuc.boxes)):
            sinif_adi = sonuc.names[int(sonuc.boxes.cls[i])]
            guven = float(sonuc.boxes.conf[i])
            # Katman 2: sınıfın KENDİ eşiğini uygula
            if guven < self.esikler.get(sinif_adi, 0.5):
                continue
            iz_id = None
            if sonuc.boxes.id is not None:
                iz_id = int(sonuc.boxes.id[i])
            x1, y1, x2, y2 = (int(v) for v in sonuc.boxes.xyxy[i].tolist())
            tespitler.append((iz_id, sinif_adi, guven, (x1, y1, x2, y2)))
        return tespitler

    # ─────────────── KATMAN 3: N-of-M zamansal filtre ───────────────
    def _izleri_guncelle(self, tespitler):
        """Her izin kayan penceresine bu karenin sonucunu işler.

        Dönen liste: bu karede İLK KEZ onaylanan izler (kanıt kaydı bunlar
        için alınacak).
        """
        gorulen_idler = set()
        for iz_id, sinif_adi, guven, kutu in tespitler:
            if iz_id is None:
                continue  # id'siz tespit zamansal sayıma giremez
            gorulen_idler.add(iz_id)
            if iz_id not in self.izler:
                _, pencere_boyu = self.nofm.get(sinif_adi, (4, 8))
                self.izler[iz_id] = IzDurumu(iz_id, sinif_adi, pencere_boyu)
            iz = self.izler[iz_id]
            iz.pencere.append(1)      # bu karede isabet var
            iz.kayip_sayaci = 0
            iz.son_kutu = kutu
            iz.son_guven = guven

        # Bu karede görünmeyen izlere "ıska" (0) yaz; uzun süre kayıplara
        # unutma uygula. list(...) kopyası: döngü içinde sözlükten
        # eleman sileceğimiz için gerekli.
        yeni_onaylananlar = []
        for iz_id, iz in list(self.izler.items()):
            if iz_id not in gorulen_idler:
                iz.pencere.append(0)
                iz.kayip_sayaci += 1
                if iz.kayip_sayaci > IZ_UNUTMA_SINIRI:
                    del self.izler[iz_id]
                    continue

            # N-of-M kontrolü: penceredeki isabet toplamı N'e ulaştı mı?
            gereken_n, _ = self.nofm.get(iz.sinif_adi, (4, 8))
            if not iz.onaylandi and sum(iz.pencere) >= gereken_n:
                iz.onaylandi = True
                yeni_onaylananlar.append(iz)
        return yeni_onaylananlar

    # ─────────────── KATMAN 4: kanıt kaydı (fotoğraf + CSV) ───────────────
    def _kanit_kaydet(self, iz: IzDurumu, cizili_kare, video_sn):
        """Onaylanan iz için anotasyonlu kareyi diske, olayı CSV'ye yazar.

        Alarm/bildirim bilinçli olarak YOK; sistemin görevi kanıt
        biriktirmek. Aynı iz için bu fonksiyon bir kez çağrılır
        (kanit_alindi bayrağı) — yoksa aynı yangın her karede yeni
        fotoğraf üretip diski doldururdu.
        """
        zaman = datetime.now()
        dosya_adi = f"{iz.sinif_adi}_iz{iz.iz_id}_{zaman.strftime('%Y%m%d_%H%M%S')}.jpg"
        dosya_yolu = self.kayit_klasoru / dosya_adi
        cv2.imwrite(str(dosya_yolu), cizili_kare)

        yeni_dosya = not self.csv_yolu.exists()
        with open(self.csv_yolu, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if yeni_dosya:
                w.writerow(["Tarih_Saat", "Video_sn", "Sinif", "Guven_%", "Iz_ID", "Kanit_Dosyasi"])
            w.writerow([
                zaman.strftime("%Y-%m-%d %H:%M:%S"),
                f"{video_sn:.1f}",
                iz.sinif_adi,
                f"{iz.son_guven * 100:.1f}",
                iz.iz_id,
                dosya_adi,
            ])
        iz.kanit_alindi = True
        return dosya_adi

    # ─────────────── ÇİZİM: aday/onaylı ayrımını görselleştir ───────────────
    def _kareyi_cizdir(self, kare, tespitler):
        """Kutuları duruma göre çizer: aday ince sarı, onaylı kalın renkli.

        result.plot() KULLANMIYORUZ çünkü o her kutuyu aynı stille çizer;
        aday/onaylı ayrımı bu sistemin kalbi olduğundan kendimiz çiziyoruz.
        """
        cizim = kare.copy()
        for iz_id, sinif_adi, guven, (x1, y1, x2, y2) in tespitler:
            iz = self.izler.get(iz_id) if iz_id is not None else None
            onayli = iz is not None and iz.onaylandi
            renk = RENK_ONAYLI.get(sinif_adi, (0, 0, 255)) if onayli else RENK_ADAY
            kalinlik = 3 if onayli else 1
            durum = "ONAYLI" if onayli else "aday"
            etiket = f"{sinif_adi} {guven:.2f} [{durum}]"

            cv2.rectangle(cizim, (x1, y1), (x2, y2), renk, kalinlik)
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(cizim, (x1, y1 - th - 8), (x1 + tw + 4, y1), renk, -1)
            cv2.putText(cizim, etiket, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        return cizim

    # ─────────────── DIŞ DÜNYAYA AÇILAN TEK KAPI ───────────────
    def kare_isle(self, kare, video_sn: float):
        """Bir kareyi uçtan uca işler; arayüzlerin çağırdığı tek fonksiyon.

        Dönenler:
          cizili_kare  : kutuları çizilmiş kare (ekranda göstermek için)
          yeni_olaylar : bu karede İLK KEZ onaylanan olayların listesi;
                         her öğe {"sinif", "guven", "iz_id", "dosya"} sözlüğü.
                         Arayüz bunları tabloya ekler/kullanıcıya gösterir.
        """
        tespitler = self._tespit_et(kare)              # katman 1+2
        yeni_onaylananlar = self._izleri_guncelle(tespitler)  # katman 3
        cizili_kare = self._kareyi_cizdir(kare, tespitler)

        yeni_olaylar = []
        for iz in yeni_onaylananlar:
            if not iz.kanit_alindi:
                dosya = self._kanit_kaydet(iz, cizili_kare, video_sn)  # katman 4
                yeni_olaylar.append({
                    "sinif": iz.sinif_adi,
                    "guven": iz.son_guven,
                    "iz_id": iz.iz_id,
                    "dosya": dosya,
                })
        return cizili_kare, yeni_olaylar

    def durum_ozeti(self):
        """Arayüzün durum satırı için kısa özet: aktif iz ve onaylı sayısı."""
        onayli = sum(1 for iz in self.izler.values() if iz.onaylandi)
        return {"aktif_iz": len(self.izler), "onayli": onayli}


# ─────────────────────────── TEK FOTOĞRAF MODU ───────────────────────────
def fotograf_isle(model, kare, esikler=None):
    """Tek bir fotoğrafta tespit yapar (Streamlit'in 'Görsel' sekmesi için).

    Zamansal filtre (N-of-M) burada YOK ve olamaz: tek karede "kalıcılık"
    diye bir kavram yok. Bu yüzden fotoğraf modunda sadece sınıf bazlı
    eşik uygulanır ve her tespit doğrudan çizilir.

    Dönenler: (cizili_kare, sayimlar)  — sayimlar örn. {"fire": 2, "smoke": 1}
    """
    esikler = esikler or dict(VARSAYILAN_ESIKLER)
    sonuc = model.predict(kare, conf=min(esikler.values()), imgsz=640, verbose=False)[0]

    cizim = kare.copy()
    sayimlar = {}
    if sonuc.boxes is not None:
        for i in range(len(sonuc.boxes)):
            sinif_adi = sonuc.names[int(sonuc.boxes.cls[i])]
            guven = float(sonuc.boxes.conf[i])
            if guven < esikler.get(sinif_adi, 0.5):
                continue
            x1, y1, x2, y2 = (int(v) for v in sonuc.boxes.xyxy[i].tolist())
            renk = RENK_ONAYLI.get(sinif_adi, (0, 0, 255))
            etiket = f"{sinif_adi} {guven:.2f}"
            cv2.rectangle(cizim, (x1, y1), (x2, y2), renk, 2)
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(cizim, (x1, y1 - th - 8), (x1 + tw + 4, y1), renk, -1)
            cv2.putText(cizim, etiket, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
            sayimlar[sinif_adi] = sayimlar.get(sinif_adi, 0) + 1
    return cizim, sayimlar


# ─────────────────────────── KOMUT SATIRI TESTİ ───────────────────────────
# Streamlit olmadan hızlı deneme: python FireAndSmokeVideo.py video.mp4
# (0 verirsen bilgisayarın webcam'ini açar)
if __name__ == "__main__":
    import sys

    kaynak = sys.argv[1] if len(sys.argv) > 1 else "0"
    kaynak = int(kaynak) if kaynak.isdigit() else kaynak

    izleyici = YanginDumanIzleyici()
    cap = cv2.VideoCapture(kaynak)
    if not cap.isOpened():
        raise SystemExit(f"Kaynak açılamadı: {kaynak}")

    KARE_ATLA = 5  # her 5 kareden 1'ini işle (app.py dosya moduyla aynı seyreltme)
    kare_no = -1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        kare_no += 1
        if kare_no % KARE_ATLA != 0:
            continue
        cizili, olaylar = izleyici.kare_isle(kare, kare_no / fps)
        for o in olaylar:
            print(f"[ONAY] {o['sinif']} (guven {o['guven']:.2f}) -> {o['dosya']}")
        cv2.imshow("Yangin/Duman Izleme - cikis: Q", cizili)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
