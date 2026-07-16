# FireAndSmoke — Yangın ve Duman Tespiti

## Dosyalar
| Dosya | Görev |
|---|---|
| `prepare_dataset.py` | İki Roboflow veri setini (HUO/YAN + fire/smoke) tek `merged_dataset/`e birleştirir, sınıf id'lerini isme göre normalize eder (0=fire, 1=smoke) |
| `train.py` | `merged_dataset/` üzerinde YOLO11s eğitimi başlatır |
| `evaluate.py` | En son eğitimin `best.pt`'sini **test** setinde ölçer (mAP/precision/recall) — eğitim öncesi/sonrası karşılaştırma bununla yapılır |
| `FireAndSmokeVideo.py` | Tespit çekirdeği — hem fotoğraf (`process_photo`, sınıf bazlı eşik + opsiyonel TTA) hem gerçek zamanlı video (`FireSmokeMonitor`: ByteTrack takip + N-of-M zamansal onay + kanıt fotoğrafı `YanginKayitlari/` + olay CSV'si `yangin_olay_log.csv`) burada. `find_latest_best_weights()` de burada tanımlı, `evaluate.py` ve `app.py` buradan import ediyor (tek doğru kaynak). Kökteki Streamlit arayüzü (`app.py`, 🔥 sekmesi) bu modülü kullanır; `python FireAndSmokeVideo.py video.mp4` ile arayüzsüz de test edilebilir |

## Çalıştırma sırası
```
python prepare_dataset.py   # sadece veri değişince
python train.py             # eğitim (GPU'lu ortamda)
python evaluate.py          # sayısal ölçüm
```
Görsel/toplu test ve klasöre foto ekleme artık Streamlit arayüzünde
(`streamlit run app.py`, 🔥 Yangın & Duman Tespiti sekmesi, Görsel alt
sekmesi) — ayrı bir CLI scripti yok, tek yer burası.

## Veri iyileştirme durumu

### 1. Hard-negative (tuzak) görseller — ÇÖZÜLDÜ (D-Fire ile)
İlk eğitimin konfüzyon matrisi, yanlış alarmların çoğunun parlak/turuncu
bölgelere "fire" denmesinden geldiğini gösterdi. Sebep: eğitim setindeki
1964 görselden sadece 12'si yangınsız (negatif) idi.

D-Fire veri seti (`smoke-fire-detection-yolo`, Kaggle) bunu çözüyor:
21.527 görsel, bunun 9.838'i ("None") yangınsız/negatif. `train_kaggle.py`
Hücre 2b bu veri setini Kaggle Input olarak alıp kendi `merged_dataset`
ile birleştiriyor; sınıf isimleri ters sırada olsa da (`smoke=0, fire=1`)
isme göre eşleştirip canonical id'ye (`fire=0, smoke=1`) çeviriyor,
etiketsiz (negatif) görselleri boş `.txt` ile işaretliyor.

Bu birleştirme sadece Kaggle tarafında yapılıyor (lokal `prepare_dataset.py`
D-Fire'ı içermiyor) çünkü 21k görseli lokale indirip tekrar zip'leyip
yüklemek gereksiz bant genişliği harcar.

### 2. Daha büyük açık veri seti — TAMAMLANDI
Yukarıdaki D-Fire entegrasyonu bu maddeyi kapsıyor.
