# FireAndSmoke — Teknik Rapor

Gerçek zamanlı yangın (`fire`) ve duman (`smoke`) tespiti + zamansal
doğrulama modülü. Bu belge modülün mevcut teknik yapısını anlatır;
tarihçe için kök dizindeki `PROJE_RAPORU.txt` (4. bölüm) ve
`README.md`'ye bakılabilir.

## 1. Model

- Mimari: **YOLO11s** (Ultralytics), 2 sınıf (`fire`, `smoke`).
- Aktif ağırlık: `FireAndSmoke/fire_smoke_yolo11s_dfire.pt` (KÖK dizinde,
  diğer üç modülün "elle bırakılan hazır model" deseniyle aynı).
  `find_latest_best_weights()` önce kök dizindeki .pt dosyalarına bakar
  (ön-eğitimli COCO ağırlıkları `yolo11n.pt`/`yolo11s.pt` hariç tutularak),
  bulamazsa `runs/detect/*/weights/*.pt` altındaki en son değiştirilen
  ağırlığa düşer (`last.pt` hariç) — birden fazla eski eğitim denemesi
  (`fire_smoke_yolo11-4` nano taban çizgisi, `fire_smoke_yolo11s`
  D-Fire'sız ilk small deneme) hâlâ o klasörde tutulur, ama artık
  kullanılmıyorlar.
- **nano → small geçişi**: ilk deneme (yolo11n, 50 epoch) recall'ü
  0.52'de kaldı — nano'nun ~2.6M parametresi, duman gibi şekilsiz/yarı
  saydam nesnelerde kapasite sınırına takılıyordu. yolo11s (~9.4M
  parametre), 6GB VRAM'li RTX 3060 Laptop'a `batch=4` + otomatik
  karışık hassasiyetle sığıyor ve zor sınıflarda belirgin mAP artışı
  sağlıyor (batch=8 denendi, epoch ortasında CUDA OOM verdi).

## 2. Veri hattı (`prepare_dataset.py` + `train_kaggle.py`)

**Yerel birleştirme** — iki Roboflow veri seti tek `merged_dataset/`e
normalize edilir:
- `fire-smoke-1` (Çince pinyin etiketli: `HUO`=ateş, `YAN`=duman)
- `fire-and-smoke-1` (İngilizce: `fire`, `smoke`)

Class id'lere körlemesine güvenilmez: her veri setinin kendi
`data.yaml`'ındaki isim listesi okunur, `NAME_SYNONYMS` sözlüğüyle
("huo"→fire, "yan"→smoke) kanonik isme çevrilir, etiket dosyalarındaki
class id'ler `CANONICAL_CLASSES` sırasına (`0=fire, 1=smoke`) göre
yeniden yazılır. Segmentasyon/poligon formatındaki etiketler de
desteklenir (yalnızca ilk sütun/class id değiştirilir).

**Hard-negative sorunu ve D-Fire çözümü** — ilk eğitimin konfüzyon
matrisi, yanlış alarmların çoğunluğunun parlak/turuncu bölgelere
"fire" denmesinden geldiğini gösterdi: eğitim setindeki 1964
görselden yalnızca 12'si negatifti (yangınsız). **D-Fire** veri seti
(Kaggle, 21.527 görsel, 9.838'i negatif/"None") bunu çözüyor;
`train_kaggle.py` bu veri setini D-Fire'ın ters sıralı sınıflarını
(`smoke=0, fire=1`) isme göre eşleştirip canonical id'ye çevirerek ve
etiketsiz (negatif) görselleri boş `.txt` ile işaretleyerek
`merged_dataset` ile birleştirir. Bu birleştirme yalnızca Kaggle
tarafında yapılır (21k görseli yerelde indirip tekrar paketlemek
gereksiz bant genişliği harcar); aktif model (`_dfire` son eki) bu
genişletilmiş veriyle eğitilmiş modeldir.

## 3. Eğitim parametreleri (`train.py`)

```
epochs=150, patience=30 (erken durdurma), imgsz=640, batch=4,
optimizer=AdamW, workers=4
```

`epochs` 150'ye çıkarıldı çünkü ilk 50 epoch'luk denemede loss/mAP
eğrileri hâlâ yükseliyordu (erken kesilmişti). Küçük batch'in etkisini
Ultralytics kendisi kısmen telafi eder: gradyanları biriktirip efektif
batch'i `nbs=64`'e tamamlar.

## 4. Video/canlı izleme hattı (`FireSmokeMonitor`)

Bu modülün asıl katma değeri — kare akışı 4 katmandan geçer:

```
kare ──▶ [1] YOLO track (ByteTrack, persist=True)
     ──▶ [2] sınıf bazlı güven eşiği (fire 0.37, smoke 0.23)
     ──▶ [3] N-of-M zamansal onay (kayan pencere, sınıfa göre farklı boyut)
     ──▶ [4] onay anında kanıt kaydı (fotoğraf + CSV, tek seferlik)
```

**Sınıf bazlı eşik farkı**: `fire` için daha yüksek eşik (parlak/
turuncu nesnelere yanlış alarm vermesin), `smoke` için daha düşük
eşik (duman doğası gereği düşük skor alır, kaçmasın). Değerler
sezgiyle değil, `evaluate.py`'nin test setinde ürettiği
F1-Confidence eğrisinden (her sınıf için F1'in zirve yaptığı nokta)
belirlendi: `fire 0.45→0.37` (F1 0.587→0.614), `smoke 0.30→0.23`
(F1 0.623→0.634).

**N-of-M zamansal filtre** (bu modülün temel tasarım fikri, diğer
modüllere de buradan taşındı): tek karelik tespit güvenilmez (far
yansıması, turuncu kıyafet, parıltı tek karede "fire" çıkabilir).
Her track id için `deque(maxlen=M)` bir kayan pencere tutulur (1=bu
karede eşiği geçen tespit vardı, 0=yoktu); pencerede en az `N` isabet
varsa iz **onaylanır**. Varsayılan pencereler sınıfa göre farklı:
`fire (4,8)` ≈ "2 saniyelik pencerede 1 saniyelik kanıt" (hızlı tepki),
`smoke (5,10)` ≈ "2.5 saniyede 1.25 sn kanıt" (daha fazla kanıt bekler,
~4 Hz örnekleme hızında). Bir iz `TRACK_FORGET_LIMIT=12` kareden fazla
üst üste görünmezse silinir (bellek sızıntısı olmasın, ByteTrack aynı
id'yi başka nesneye verirse eski geçmişle karışmasın).

Görünmeyen izlere pencerede **0 (ıska) yazılır** — ara sıra kaybolan
gerçek bir yangın bile toparlanabilir, ama sürekli kaybolan tek seferlik
yanlış alarm N-of-M şartını asla sağlayamaz.

**Kanıt kaydı**: onaylanan her iz için bir kez anotasyonlu JPEG
(`YanginKayitlari/`) ve CSV satırı (`yangin_olay_log.csv`: tarih,
video sn, sınıf, güven, iz id, dosya adı) yazılır. Alarm/bildirim
bilinçli olarak yok — sistemin görevi kanıt biriktirmek, aynı yangın
her karede yeni fotoğraf üretip diski doldurmasın diye tek seferlik.

**Görsel dil**: ince sarı kutu = aday (tespit var, henüz N-of-M onayı
yok), kalın kırmızı = onaylı fire, kalın turuncu = onaylı smoke.
Etiket konumlandırması özel: duman genelde karenin en üst kısmında
çıktığı için, etiketin üstte yeterli yer olup olmadığı kontrol edilir;
yoksa etiket kutunun içine (üst kenarın hemen altına) çizilir ki
ekran dışına taşıp görünmez olmasın.

## 5. Fotoğraf modu (`process_photo`)

Tek karede zamansal onay uygulanamaz; bu yüzden yalnızca sınıf bazlı
eşik uygulanır, her tespit doğrudan çizilir. TTA (`augment=True`) her
zaman açık — görsel birkaç varyasyonda (orijinal + aynalı + ölçekli)
modele sokulup sonuçlar birleştirilir, ~2-3 kat yavaş ama daha
isabetli; video gibi sürekli bir akış olmadığı için bu maliyetin
önemi yok.

## 6. Arayüz entegrasyonu (`app.py`)

- `🔥 Yangın & Duman Tespiti` sekmesi, `process_photo` (Görsel) ve
  `FireSmokeMonitor` (Video, dosya/IP kamera) üzerinden çalışır.
- "📊 Metrikleri Hesapla" butonu `model.val()` ile `merged_dataset/test`
  üzerinde ölçüm yapar (Precision/Recall/F1/mAP50/mAP50-95 + sınıf
  bazlı açılır tablo).
- `python FireAndSmokeVideo.py video.mp4` ile Streamlit'siz komut
  satırından da (webcam dahil, `0` argümanıyla) test edilebilir.

## 7. Bilinen sınırlamalar

- Yerel `prepare_dataset.py` D-Fire'ı içermiyor (yalnızca Kaggle
  tarafında birleştiriliyor); yerelde yeniden eğitim yapılırsa D-Fire
  ayrıca indirilip eklenmeli.
- Diskte üç farklı eğitim denemesinin çıktısı (`runs/detect/fire_smoke_yolo11-4/`,
  `fire_smoke_yolo11s/`, `fire_smoke_yolo11s_dfire/`) birikmiş durumda;
  aktif model (`fire_smoke_yolo11s_dfire.pt`) artık kök dizine taşındı,
  `runs/` altındakiler sadece arşiv — temizlenebilir.

## Dosya haritası

| Dosya | Görev |
|---|---|
| `FireAndSmokeVideo.py` | Tespit çekirdeği: `FireSmokeMonitor` (video), `process_photo` (fotoğraf), `find_latest_best_weights()`. Arayüz içermez. |
| `prepare_dataset.py` | İki Roboflow veri setini `merged_dataset/`de birleştirir, sınıf id'lerini normalize eder. |
| `train.py` | `merged_dataset/` üzerinde YOLO11s eğitimi (yerel). |
| `train_kaggle.py` | Kaggle'da D-Fire dahil genişletilmiş eğitim. |
| `evaluate.py` | En son ağırlığı test setinde ölçer, eşik önerisi üretir. |
| `fire_smoke_yolo11s_dfire.pt` | **Aktif model** (kök dizinde). |
| `runs/detect/*/weights/*.pt` | Eski eğitim denemelerinin arşivi (artık kullanılmıyor). |
| `YanginKayitlari/`, `yangin_olay_log.csv` | Video hattının kanıt fotoğrafları ve olay kaydı. |
