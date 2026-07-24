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
buradaki FireSmokeMonitor sınıfını kullanır. Böylece izleme mantığı
tek yerde yaşar, her arayüz aynı davranışı gösterir.
"""

import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent


def find_latest_best_weights():
    # Sadece FireAndSmoke/runs/detect altindaki egitimlere bakar (baska
    # projelerdeki agirlik dosyalarina karismaz). Her deneme klasorundeki
    # "best" agirligi kendi deney adiyla yeniden adlandirildi (ornek:
    # fire_smoke_yolo11s_dfire.pt) — "best.pt" gibi jenerik bir isim
    # istenmedigi icin. "last.pt" (her epoch sonu kaydedilen, henuz en
    # iyi olmayan agirlik) HARIC tutulur; kalanlardan en son degistirilen
    # (en guncel egitimden kalan) otomatik secilir. Bu desen, ileride
    # yeniden adlandirilmamis TAZE bir Ultralytics ciktisini (hala
    # "best.pt" adinda) da sorunsuz yakalar.
    runs_dir = BASE_DIR / "runs" / "detect"

    candidates = [p for p in runs_dir.glob("*/weights/*.pt") if p.name != "last.pt"]
    if not candidates:
        raise FileNotFoundError(
            f"{runs_dir} altinda hicbir 'best.pt' bulunamadi. Once train.py ile "
            f"bir egitim tamamlanmis olmali."
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)

# ─────────────────────────── AYARLAR ───────────────────────────
# Sınıf bazlı güven eşikleri. Değerler sezgiyle değil, evaluate.py'nin
# test setinde ürettiği F1-Confidence eğrisinden (metrics.box.f1_curve,
# metrics.box.px) her sınıf için F1'in zirve yaptığı nokta okunarak
# belirlendi: fire 0.45->0.366'da (F1 0.587->0.614), smoke 0.30->0.229'da
# (F1 0.623->0.634) daha iyi sonuç veriyordu. Yeni bir eğitimden sonra bu
# değerler değişebilir; evaluate.py çalıştırıp aynı yöntemle tekrar
# hesaplanmalı (bkz. FireAndSmoke/README.md).
DEFAULT_THRESHOLDS = {"fire": 0.37, "smoke": 0.23}

# N-of-M pencere ayarı: {sınıf: (N=en az kaç isabet, M=pencere boyu)}.
# Örnekleme hızımız saniyede ~4 kare olduğundan fire 4/8 ≈ "2 saniyelik
# pencerede 1 saniyelik kanıt", smoke 5/10 ≈ "2.5 saniyede 1.25 sn kanıt".
DEFAULT_NOFM = {"fire": (4, 8), "smoke": (5, 10)}

# Bir iz bu kadar örnekleme karesi boyunca hiç görünmezse unutulur
# (nesne sahneden çıktı ya da yanlış alarmdı demektir).
TRACK_FORGET_LIMIT = 12

# Çizim renkleri (BGR): aday sarı, onaylı fire kırmızı, onaylı smoke turuncu.
COLOR_CANDIDATE = (0, 220, 220)
COLOR_CONFIRMED = {"fire": (0, 0, 255), "smoke": (0, 140, 255)}


def _draw_box_with_label(canvas, box, color, thickness, label):
    """Kutuyu ve üstündeki etiketi (sınıf adı + güven skoru) çizer.

    NEDEN GEREKLİ (asıl mesele burada): duman yükselen bir şey olduğu için
    kutular sıkça karenin EN ÜST kısmında çıkıyor. Etiketi eskiden hep
    kutunun ÜSTÜNE (y1'in üzerine, negatif y koordinatına) çiziyorduk;
    kare üstte y1 küçükse (0'a yakınsa) bu, kare SINIRLARININ DIŞINA
    taşıyordu ve OpenCV ekran dışına çizileni göstermiyordu — yani etiket
    (sınıf adı + confidence) görünmez oluyordu.
    Çözüm: kutunun üstünde etiket için yeterli boşluk (>=0) var mı diye
    kontrol ediyoruz. Yoksa etiketi kutunun İÇİNE, üst kenarın hemen
    ALTINA çiziyoruz — böylece hiçbir zaman kare dışına taşmıyor.
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_h = th + 8  # etiket kutusunun toplam yüksekliği (metin + dolgu)

    if y1 - label_h >= 0:
        # Kutunun üstünde yeterli yer var: etiketi ÜSTÜNE çiz (eski davranış).
        top = y1 - label_h
        text_y = y1 - 5
    else:
        # Yer yok (kutu ekranın en üstüne yakın): etiketi kutunun İÇİNE,
        # üst kenarın hemen altına çiz ki ekran dışına taşmasın.
        top = y1
        text_y = y1 + th + 3

    cv2.rectangle(canvas, (x1, top), (x1 + tw + 4, top + label_h), color, -1)
    cv2.putText(canvas, label, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


class TrackState:
    """Tek bir track id'nin zamansal geçmişini tutar.

    window: son M örnekleme karesinin isabet kaydı (1=bu karede eşiği
    geçen tespit vardı, 0=yoktu). deque(maxlen=M) sayesinde eski kayıtlar
    kendiliğinden düşer — kayan pencere (sliding window) budur.
    """

    def __init__(self, track_id: int, class_name: str, window_size: int):
        self.track_id = track_id
        self.class_name = class_name
        self.window = deque(maxlen=window_size)
        self.confirmed = False        # N-of-M sağlandı mı?
        self.evidence_saved = False   # fotoğraf+CSV bir kez yazıldı mı?
        self.miss_count = 0           # üst üste kaç karedir görünmüyor
        self.last_box = None          # en son görüldüğü (x1,y1,x2,y2)
        self.last_conf = 0.0


class FireSmokeMonitor:
    """Video karelerini tek tek alır, mimarideki 1-4 katmanlarını uygular.

    Kullanım (arayüzden bağımsız):
        monitor = FireSmokeMonitor(model_path)
        for frame in frames:
            drawn_frame, new_events = monitor.process_frame(frame, video_sec)
    """

    def __init__(self, model_path=None, thresholds=None, nofm=None,
                 record_dir=None, csv_path=None, model=None):
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
            if model_path is None:
                model_path = find_latest_best_weights()
            self.model = YOLO(model_path)
        self.thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self.nofm = nofm or dict(DEFAULT_NOFM)

        # Kanıt fotoğraflarının klasörü ve olay CSV'si
        self.record_dir = Path(record_dir or (BASE_DIR / "YanginKayitlari"))
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = Path(csv_path or (BASE_DIR / "yangin_olay_log.csv"))

        # Aktif izler: {track_id: TrackState}
        self.tracks: dict[int, TrackState] = {}

    # ─────────────── KATMAN 1+2: track + sınıf bazlı eşik ───────────────
    def _detect(self, frame):
        """Bir karede takipli tespit yapar, eşik altını eler.

        Dönen liste: (track_id, class_name, conf, (x1,y1,x2,y2)) dörtlüleri.
        track_id None olabilir (tracker yeni nesneye henüz id atamadıysa);
        bunları zamansal filtreye sokamayacağımız için sadece ekranda
        aday olarak gösterir, sayıma katmayız.
        """
        result = self.model.track(
            frame,
            persist=True,                 # iz durumunu kareler arasında koru
            conf=min(self.thresholds.values()),  # ön eleme: en düşük eşikten azını hiç getirme
            imgsz=640,                    # eğitimle aynı çözünürlük → tutarlı skorlar
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections
        for i in range(len(result.boxes)):
            class_name = result.names[int(result.boxes.cls[i])]
            conf = float(result.boxes.conf[i])
            # Katman 2: sınıfın KENDİ eşiğini uygula
            if conf < self.thresholds.get(class_name, 0.5):
                continue
            track_id = None
            if result.boxes.id is not None:
                track_id = int(result.boxes.id[i])
            x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
            detections.append((track_id, class_name, conf, (x1, y1, x2, y2)))
        return detections

    # ─────────────── KATMAN 3: N-of-M zamansal filtre ───────────────
    def _update_tracks(self, detections):
        """Her izin kayan penceresine bu karenin sonucunu işler.

        Dönen liste: bu karede İLK KEZ onaylanan izler (kanıt kaydı bunlar
        için alınacak).

        BU FONKSİYON NE YAPAR? (genel bakış)
        Bu fonksiyon her kare için 2 aşamada çalışır:
          AŞAMA A: Bu karede GÖRÜNEN tespitleri ilgili izlere işler (isabet=1 yaz).
          AŞAMA B: Bu karede GÖRÜNMEYEN (yani bu karede eşleşmemiş) izlere
                    ıska=0 yazar, çok uzun süredir kayıp olanları siler, ve
                    her izin "onaylandı mı" durumunu kontrol eder.
        İki aşamanın ayrı olmasının sebebi: A aşamasında sadece bu karede
        GÖRÜLEN izleri biliyoruz; hangi izlerin GÖRÜLMEDİĞİNİ anlamak için
        self.tracks sözlüğünün TAMAMINI (yani geçmişten kalan tüm izleri)
        gezmemiz gerekiyor — bu da ancak A bittikten sonra, B aşamasında
        yapılabilir.
        """
        # ─── AŞAMA A: bu karede görülen tespitleri izlere işle ───
        # seen_ids: bu karede en az bir tespitle eşleşen track_id'lerin
        # kümesi. Aşama B'de "bu iz bu karede görüldü mü görülmedi mi?"
        # sorusunu cevaplamak için kullanılacak.
        seen_ids = set()
        for track_id, class_name, conf, box in detections:
            if track_id is None:
                # ByteTrack bazen (özellikle nesnenin ilk göründüğü anda)
                # henüz kalıcı bir id atamamış olabilir. id'siz bir tespiti
                # hangi ize ait olduğunu bilemeyeceğimiz için zamansal
                # pencereye ekleyemeyiz; sadece ekranda "aday" olarak
                # görünür (bkz. _draw_frame), N-of-M sayımına girmez.
                continue
            seen_ids.add(track_id)

            # Bu track_id'yi ilk defa görüyorsak (yeni bir nesne ortaya
            # çıktı), önce onun için boş bir TrackState (iz kaydı)
            # oluşturuyoruz. Pencere boyutu (window_size = M) sınıfa göre
            # değişir: fire için 8, smoke için 10 (DEFAULT_NOFM'e bak).
            if track_id not in self.tracks:
                _, window_size = self.nofm.get(class_name, (4, 8))
                self.tracks[track_id] = TrackState(track_id, class_name, window_size)

            track = self.tracks[track_id]
            # Bu karede bu iz için bir tespit VARDI -> pencereye 1 ekle.
            # window bir deque(maxlen=M) olduğu için, pencere zaten M
            # elemanla doluysa en eski eleman otomatik olarak düşer ve
            # yerine bu yeni 1 eklenir (kayan pencere / sliding window).
            # Örnek (M=8): [1,1,0,1,1,0,1,1] + yeni 1 -> ilk 1 düşer,
            #              [1,0,1,1,0,1,1,1] olur.
            track.window.append(1)
            track.miss_count = 0        # art arda kayıp sayacı sıfırlanır (yeniden görüldü)
            track.last_box = box        # ekranda çizim yapabilmek için son konumu güncelle
            track.last_conf = conf      # kanıt kaydında kullanılacak son güven skoru

        # ─── AŞAMA B: görünmeyenlere ıska yaz + unutma + N-of-M kontrolü ───
        # list(self.tracks.items()) ile bir KOPYA üzerinde geziyoruz çünkü
        # döngü içinde self.tracks sözlüğünden eleman silebiliyoruz
        # (del self.tracks[track_id]); orijinal sözlük üzerinde gezerken
        # onu değiştirmeye çalışmak Python'da RuntimeError fırlatır.
        newly_confirmed = []
        for track_id, track in list(self.tracks.items()):
            if track_id not in seen_ids:
                # Bu iz BU KAREDE görünmedi (nesne geçici olarak kayboldu,
                # başka bir şeyin arkasına girdi, ya da tamamen bitti).
                # Pencereye 0 (ıska) yazıyoruz -> bu, N-of-M oranını düşürür;
                # yani ara sıra kaybolan gerçek bir yangın bile toparlanabilir,
                # ama sürekli kaybolan bir "tek seferlik yanlış alarm" asla
                # N-of-M şartını sağlayamaz.
                track.window.append(0)
                track.miss_count += 1

                # Bir iz TRACK_FORGET_LIMIT (12) kareden fazla üst üste hiç
                # görünmezse, artık o nesne sahneden çıkmış demektir; izi
                # tamamen siliyoruz ki self.tracks sözlüğü sonsuza kadar
                # büyümesin (bellek sızıntısı olmasın) ve ByteTrack ileride
                # aynı track_id'yi başka bir nesneye verirse eski, alakasız
                # geçmişle karışmasın. `continue` ile bu ize N-of-M kontrolü
                # dahi uygulamıyoruz çünkü artık yok.
                if track.miss_count > TRACK_FORGET_LIMIT:
                    del self.tracks[track_id]
                    continue

            # N-of-M KONTROLÜ — bu fonksiyonun asıl amacı burası:
            # Bu izin sınıfına göre gereken minimum isabet sayısını (N) al
            # (fire için N=4, smoke için N=5 varsayılan).
            required_n, _ = self.nofm.get(track.class_name, (4, 8))
            # sum(track.window): pencerede kaç tane 1 (isabet) var, saymanın
            # en kısa yolu (deque içindeki tüm elemanları toplar; 1'ler 1,
            # 0'lar 0 katkı verdiği için toplam = isabet sayısı).
            # not track.confirmed: bir iz sadece BİR KEZ onaylanabilir;
            # zaten onaylanmış bir izi tekrar "yeni onaylandı" listesine
            # eklememek için bu kontrol var (yoksa her karede tekrar tekrar
            # kanıt kaydı tetiklenmeye çalışılırdı).
            if not track.confirmed and sum(track.window) >= required_n:
                track.confirmed = True
                # Bu izi "yeni onaylananlar" listesine ekliyoruz; bunu
                # process_frame() çağıran taraf kullanıp _save_evidence()
                # ile kanıt fotoğrafı/CSV satırı üretecek (katman 4).
                newly_confirmed.append(track)
        return newly_confirmed

    # ─────────────── KATMAN 4: kanıt kaydı (fotoğraf + CSV) ───────────────
    def _save_evidence(self, track: TrackState, drawn_frame, video_sec):
        """Onaylanan iz için anotasyonlu kareyi diske, olayı CSV'ye yazar.

        Alarm/bildirim bilinçli olarak YOK; sistemin görevi kanıt
        biriktirmek. Aynı iz için bu fonksiyon bir kez çağrılır
        (evidence_saved bayrağı) — yoksa aynı yangın her karede yeni
        fotoğraf üretip diski doldururdu.
        """
        timestamp = datetime.now()
        file_name = f"{track.class_name}_iz{track.track_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        file_path = self.record_dir / file_name
        cv2.imwrite(str(file_path), drawn_frame)

        is_new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(["Tarih_Saat", "Video_sn", "Sinif", "Guven_%", "Iz_ID", "Kanit_Dosyasi"])
            writer.writerow([
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                f"{video_sec:.1f}",
                track.class_name,
                f"{track.last_conf * 100:.1f}",
                track.track_id,
                file_name,
            ])
        track.evidence_saved = True
        return file_name

    # ─────────────── ÇİZİM: aday/onaylı ayrımını görselleştir ───────────────
    def _draw_frame(self, frame, detections):
        """Kutuları duruma göre çizer: aday ince sarı, onaylı kalın renkli.

        result.plot() KULLANMIYORUZ çünkü o her kutuyu aynı stille çizer;
        aday/onaylı ayrımı bu sistemin kalbi olduğundan kendimiz çiziyoruz.
        """
        canvas = frame.copy()
        for track_id, class_name, conf, (x1, y1, x2, y2) in detections:
            track = self.tracks.get(track_id) if track_id is not None else None
            confirmed = track is not None and track.confirmed
            color = COLOR_CONFIRMED.get(class_name, (0, 0, 255)) if confirmed else COLOR_CANDIDATE
            thickness = 3 if confirmed else 1
            status = "ONAYLI" if confirmed else "aday"
            label = f"{class_name} {conf:.2f} [{status}]"

            _draw_box_with_label(canvas, (x1, y1, x2, y2), color, thickness, label)
        return canvas

    # ─────────────── DIŞ DÜNYAYA AÇILAN TEK KAPI ───────────────
    def process_frame(self, frame, video_sec: float):
        """Bir kareyi uçtan uca işler; arayüzlerin çağırdığı tek fonksiyon.

        Dönenler:
          drawn_frame : kutuları çizilmiş kare (ekranda göstermek için)
          new_events  : bu karede İLK KEZ onaylanan olayların listesi;
                        her öğe {"class_name", "conf", "track_id", "file_name"} sözlüğü.
                        Arayüz bunları tabloya ekler/kullanıcıya gösterir.
        """
        detections = self._detect(frame)                     # katman 1+2
        newly_confirmed = self._update_tracks(detections)     # katman 3
        drawn_frame = self._draw_frame(frame, detections)

        new_events = []
        for track in newly_confirmed:
            if not track.evidence_saved:
                file_name = self._save_evidence(track, drawn_frame, video_sec)  # katman 4
                new_events.append({
                    "class_name": track.class_name,
                    "conf": track.last_conf,
                    "track_id": track.track_id,
                    "file_name": file_name,
                })
        return drawn_frame, new_events

    def get_status(self):
        """Arayüzün durum satırı için kısa özet: aktif iz ve onaylı sayısı."""
        confirmed = sum(1 for track in self.tracks.values() if track.confirmed)
        return {"active_tracks": len(self.tracks), "confirmed": confirmed}


# ─────────────────────────── TEK FOTOĞRAF MODU ───────────────────────────
def process_photo(model, frame, thresholds=None):
    """Tek bir fotoğrafta tespit yapar (Streamlit'in 'Görsel' sekmesi için).

    Zamansal filtre (N-of-M) burada YOK ve olamaz: tek karede "kalıcılık"
    diye bir kavram yok. Bu yüzden fotoğraf modunda sadece sınıf bazlı
    eşik uygulanır ve her tespit doğrudan çizilir.

    TTA (Test Zamanında Veri Artırma) burada HER ZAMAN açık: görseli
    birkaç varyasyonda (orijinal + aynalı + ölçekli) modele sokup
    sonuçları birleştirir, ~2-3 kat yavaş ama daha isabetli. Video gibi
    sürekli/gerçek zamanlı bir akış olmadığı için bu yavaşlamanın
    maliyeti yok — parametre olarak dışarı açmaya gerek görmedik.

    Dönenler: (drawn_frame, counts)  — counts örn. {"fire": 2, "smoke": 1}
    """
    thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
    result = model.predict(frame, conf=min(thresholds.values()), imgsz=640, augment=True, verbose=False)[0]

    canvas = frame.copy()
    counts = {}
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            class_name = result.names[int(result.boxes.cls[i])]
            conf = float(result.boxes.conf[i])
            if conf < thresholds.get(class_name, 0.5):
                continue
            x1, y1, x2, y2 = (int(v) for v in result.boxes.xyxy[i].tolist())
            color = COLOR_CONFIRMED.get(class_name, (0, 0, 255))
            label = f"{class_name} {conf:.2f}"
            _draw_box_with_label(canvas, (x1, y1, x2, y2), color, 2, label)
            counts[class_name] = counts.get(class_name, 0) + 1
    return canvas, counts


# ─────────────────────────── KOMUT SATIRI TESTİ ───────────────────────────
# Streamlit olmadan hızlı deneme: python FireAndSmokeVideo.py video.mp4
# (0 verirsen bilgisayarın webcam'ini açar)
if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    source = int(source) if source.isdigit() else source

    monitor = FireSmokeMonitor()
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
            print(f"[ONAY] {event['class_name']} (guven {event['conf']:.2f}) -> {event['file_name']}")
        cv2.imshow("Yangin/Duman Izleme - cikis: Q", drawn)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
