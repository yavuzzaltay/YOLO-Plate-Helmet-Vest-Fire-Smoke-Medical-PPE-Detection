# VestAndPlateDetection — Teknik Rapor

Baret (`hardhat`) ve yelek (`safety-vest`) takılıp takılmadığını tespit
eden modül. Diğer üç modülün aksine bu modülün **kendi Python dosyası
yok** — mantığı doğrudan kök dizindeki `app.py` içinde yaşıyor
(`run_vest_image`, `run_vest_video`, `log_vest_result`,
`compute_vest_metrics`, `load_vest_model` fonksiyonları). Klasörde
yalnızca eğitilmiş ağırlık (`best.pt`) ve olay kaydı (`ihlal_log.csv`)
bulunuyor; ayrı bir eğitim scripti veya yerel veri seti yok — model
muhtemelen başka bir ortamda eğitilip buraya bırakıldı.

## 1. Model

- 5 sınıf: `person`, `hardhat`, `no-hardhat`, `safety-vest`,
  `no-safety-vest` — **pozitif+negatif çift şema** (MedicalPPE'nin
  "direct" modunun ilham aldığı düzenin aynısı, burada zaten önceden
  mevcuttu).
- `load_vest_model()` doğrudan `VestAndPlateDetection/best.pt`'yi yükler
  (bulma mantığı yok — tek dosya, sabit yol).
- Mimari/eğitim parametreleri (epoch, batch, veri kaynağı) bu depoda
  belgelenmiyor; yalnızca eğitim-sonu checkpoint mevcut.

## 2. Çıkarım — düz `model.predict`, ek katman yok

`run_vest_image()` FireAndSmoke/MedicalPPE'deki gibi ByteTrack, N-of-M
zamansal onay veya kanıt-fotoğrafı katmanı **kullanmıyor** — bu
modülün diğer ikisinden temel farkı budur:

```
kare ──▶ model.predict(conf=slider_degeri) ──▶ r.plot() ile çizim
                                            ──▶ sınıf sayımı (counts)
```

- Sonuç çizimi Ultralytics'in kendi `result.plot()`'u ile yapılır
  (özel renk/durum ayrımı yok — diğer iki modülde "aday/onaylı" ayrımı
  için elle çizim var, burada değil).
- Güven eşiği tek bir sayı (Streamlit slider'ı, 0.1-0.9 arası); sınıf
  bazlı ayrı eşik veya F1-zirve kalibrasyonu yok.

## 3. Video hattı (`run_vest_video`)

Kare akışı da sade: `run_vest_image` her örneklenen karede tekrar
çağrılır, iz/kimlik takibi yok — her kare bağımsız değerlendirilir.

```
video ──▶ [dosya modu] her frame_skip=5 karede bir örnekle
       ──▶ [canlı mod]  SAMPLE_PERIOD=0.25s'de bir örnekle (buffer=1,
                         cap.grab() ile decode etmeden eski kareyi at)
       ──▶ run_vest_image(kare)
       ──▶ VEST_VIOLATION_CLASSES = {no-hardhat, no-safety-vest} sayısı
           >0 ise → log_vest_result() ile CSV'ye satır ekle
```

Loglama **kare bazlı ve zamansal filtresiz**: ihlal içeren her
örneklenen karede doğrudan bir CSV satırı yazılır (tarih, video sn,
kişi/baret-var/baret-yok/yelek-var/yelek-yok sayıları). FireAndSmoke ve
MedicalPPE'deki "aynı iz için tek seferlik kanıt" mantığı burada yok —
bu yüzden aynı ihlal birden çok karede tekrar tekrar loglanabilir.
Ayrıca kanıt fotoğrafı diske yazılmıyor, yalnızca CSV satırı.

Canlı (IP kamera) modda tampon küçültme (`CAP_PROP_BUFFERSIZE=1`) ve
zaman-tabanlı örnekleme diğer modüllerle aynı desenle uygulanıyor
(kaynak `app.py`, plaka video hattıyla paylaşılan teknik).

## 4. Metrik ölçümü — checkpoint okuma, canlı ölçüm yok

`compute_vest_metrics()` diğer üç modülden farklı çalışır: yerelde
bağımsız bir test seti **olmadığı için** `model.val()` çalıştırılamıyor.
Bunun yerine `best.pt`'nin PyTorch checkpoint'i doğrudan yüklenip
(`torch.load(..., weights_only=False)`) Ultralytics'in eğitim sonunda
otomatik gömdüğü `train_metrics` sözlüğü okunuyor:

```
Precision 92.6%   Recall 91.0%   mAP50 94.6%   mAP50-95 70.5%
```

Arayüzde bu açıkça "eğitim sırasındaki son doğrulama — yerelde ayrı
test seti yok" diye belirtiliyor; bu sayılar **bağımsız bir doğrulama
değil**, eğitimin kendi train/val ayrımından geliyor.

## 5. Arayüz entegrasyonu (`app.py`)

- `🦺 Baret & Yelek Tespiti` sekmesi: Görsel (yükle → tespit et →
  isteğe bağlı `ihlal_log.csv`'ye manuel kayıt) ve Video (kendi
  videon / IP Webcam telefon kamerası) alt sekmeleri.
- Güven eşiği slider'ı (0.1-0.9) kullanıcı tarafından ayarlanabilir —
  diğer modüllerdeki sabit/kalibre eşiklerin aksine.

## 6. Diğer modüllerle farkı — özet

| Özellik | VestAndPlateDetection | FireAndSmoke / MedicalPPE |
|---|---|---|
| Kimlik takibi (ByteTrack) | ❌ Yok | ✅ Var |
| Zamansal onay (N-of-M) | ❌ Yok | ✅ Var |
| Kanıt fotoğrafı | ❌ Yok | ✅ Var (tek seferlik) |
| Eşik kalibrasyonu | ❌ Tek sabit slider | ✅ Sınıf bazlı, F1-zirve kalibreli |
| Bağımsız test seti | ❌ Yok | ✅ Var |
| Eğitim scripti bu depoda | ❌ Yok | ✅ Var |

## Bilinen sınırlamalar / bekleyen iş

- Modelin nereden/nasıl eğitildiği bu depoda belgelenmiyor; yeniden
  eğitim veya iyileştirme gerekirse önce bir veri kaynağı ve eğitim
  scripti kurulmalı (MedicalPPE/FireAndSmoke'taki `prepare_dataset.py`
  + `train.py` deseni örnek alınabilir).
- Bağımsız test seti yokluğu, gösterilen metriklerin gerçek genelleme
  performansını değil eğitim-sonu doğrulamasını yansıttığı anlamına
  gelir.
- Zamansal filtre ve kanıt kaydı eksikliği, tek karelik yanlış
  alarmların doğrudan CSV'ye yazılabileceği anlamına gelir (diğer
  modüllerdeki N-of-M güvencesi burada yok).

## Dosya haritası

| Dosya | Görev |
|---|---|
| `best.pt` | Eğitilmiş model + `train_metrics` gömülü checkpoint. |
| `ihlal_log.csv` | Manuel/otomatik loglanan ihlal kayıtları. |
| *(mantık)* `app.py` içinde | `run_vest_image`, `run_vest_video`, `log_vest_result`, `load_vest_model`, `compute_vest_metrics`. |
