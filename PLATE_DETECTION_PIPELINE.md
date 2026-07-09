# Türk Plaka Tespit Sistemi — Sıfırdan Mimari Anlatımı

Bu belge [colab_local.py](colab_local.py) (fotoğraf hattı) ve [video_plaka.py](video_plaka.py) (video hattı) dosyalarında yapılan HER ŞEYİ, en baştan başlayarak, adım adım açıklar. Her bölümde üç soru cevaplanır:

1. **Sorun neydi?** — hangi ihtiyaç/hata bu kararı doğurdu
2. **Alternatifler nelerdi?** — başka ne yapılabilirdi, neden onlar seçilmedi
3. **Ne seçildi ve nasıl çalışıyor?** — mekanizmanın kendisi

Amaç: bu belgeyi okuyunca projeyi savunabiliyor olman — sadece "böyle yaptık" değil, "bunu şu yüzden, şu alternatiflere karşı seçtik" diyebiliyor olman.

---

# BÖLÜM A — PROBLEM TANIMI VE GENEL YAKLAŞIM

## A.1. Elimizde ne vardı, ne istiyorduk

Başlangıçta elimizde önceden eğitilmiş bir YOLO nesne tespit modeli (`best.pt`, Türk plakalarını "kutulamak" için eğitilmiş) ve EasyOCR adlı hazır bir metin okuma kütüphanesi vardı. İstenen: bir araç fotoğrafı/videosu verildiğinde plaka numarasını doğru metne çevirip kaydetmek. Başlangıç doğruluğu **6 test fotoğrafından 3'ü** (%50) — kabul edilemez derecede düşük.

## A.2. Neden tek bir "uçtan uca" model değil de çok aşamalı bir hat (pipeline)?

Burada ilk büyük mimari karar şuydu: plaka tanımayı **tek bir dev modele** (görüntüyü ver, plaka metnini al) mi yaptırmalıydık, yoksa **ayrı, uzmanlaşmış aşamalara** mı bölmeliydik?

- **Uçtan uca tek model** (örn. bir görüntüden direkt "34 ABC 123" üreten bir transformer) — bu yaklaşım çok büyük, çeşitli, etiketli bir veri seti ister (binlerce plaka görüntüsü + doğru metin eşleşmesi). Elimizde böyle bir veri seti yoktu ve sıfırdan böyle bir model eğitmek hem zaman hem hesaplama gücü açısından bu projenin kapsamı dışındaydı.
- **Çok aşamalı hat** (bizim seçtiğimiz): "önce NEREDE olduğunu bul (tespit), sonra O BÖLGEYİ oku (OCR)" şeklinde ayrıştırma. Bunun avantajı: her aşama görece basit, iyi bilinen bir problem haline geliyor (nesne tespiti + genel amaçlı metin tanıma), ve her aşama ayrı ayrı iyileştirilip hatalar izole edilebiliyor ("bu bir tespit hatası mı, yoksa bir okuma hatası mı?" sorusu ayrıştırılabiliyor).

Bu ayrım proje boyunca defalarca işimize yaradı: örneğin başlangıçtaki %50 doğruluğun **asıl nedeninin OCR değil tespit olduğunu** ancak bu ayrım sayesinde teşhis edebildik (bkz. Bölüm B).

---

# BÖLÜM B — PLAKA TESPİTİ (DETECTION)

## B.1. Neden YOLO?

Nesne tespiti için birkaç aile var: **iki aşamalı dedektörler** (Faster R-CNN gibi — önce aday bölgeler önerilir, sonra sınıflandırılır; yüksek doğruluk ama yavaş), **tek aşamalı dedektörler** (YOLO, SSD — görüntüyü tek geçişte tarayıp hem konum hem sınıf tahmini yapar; daha hızlı, gerçek zamanlı kullanıma uygun), ve **klasik (öğrenmesiz) görüntü işleme yöntemleri** (kenar/renk/şekil tabanlı, öğrenme gerektirmez ama kırılgan).

YOLO zaten önceden eğitilmiş olarak elimizdeydi, gerçek zamanlı hıza (video işlemek için kritik) sahip ve nesne tespitinde endüstri standardı. Faster R-CNN gibi bir alternatife geçmek hem modeli sıfırdan eğitmeyi hem de video hattında kabul edilemez bir yavaşlığı gerektirirdi. Bu yüzden YOLO ile devam edildi — asıl iyileştirme alanı modelin **kendisini değiştirmek değil, nasıl kullanıldığını** değiştirmekti.

## B.2. Sabit çözünürlük sorunu ve çözüm alternatifleri

**Sorun**: Model varsayılan olarak `imgsz=640` ile çalışıyordu — yani her görüntü 640 piksele küçültülüp modele veriliyordu. Yüksek çözünürlüklü bir fotoğrafta (örn. 2048px) küçük bir plaka bu küçültme sonrası 15-25 piksele düşüyor, model onu **hiç göremiyordu**. Bu, test setindeki başarısızlıkların çoğunun **asıl kök nedeniydi** — OCR'ı iyileştirmeden önce bunu bulmak proje için dönüm noktasıydı.

**Değerlendirilen alternatifler:**

| Yaklaşım | Ne yapar | Neden seçilmedi/seçildi |
|---|---|---|
| **Modeli yeniden eğitmek** (yüksek çözünürlüklü veriyle) | Kalıcı, en temiz çözüm | Zaman/veri seti kısıtı vardı; kısa vadede uygulanamazdı. Uzun vadede önerilen çözüm budur (bkz. Bölüm F) |
| **Görüntüyü kutulara bölüp (tiling) her parçayı ayrı taramak** | Her parça kendi çözünürlüğünde kalır, küçük nesneler kaybolmaz | Video için çok pahalı (kare başına N kat YOLO çağrısı); karmaşıklığı fayda/maliyet dengesini bozuyordu |
| **Çok geçişli (multi-pass) tespit** — SEÇİLEN | Aynı görüntüyü farklı `imgsz` değerleriyle birden çok kez tara | Ek eğitim gerektirmez, uygulaması basit, video'da da makul maliyetli (genelde ilk geçişte bulunur) |

**Nasıl çalışıyor**: `detect_plate_boxes()` üç geçiş dener: `imgsz=1280/conf=0.25` → `1920/0.10` → `640/0.03`. Üçüncü geçiş özellikle ilginç: modelin **kendi eğitildiği çözünürlükte** (640px) çok düşük bir güven eşiğiyle taranması. Deneyler şunu gösterdi: model 640px'te bazı zor plakaları düşük güvenle de olsa görebiliyordu, ama görüntü yüksek çözünürlükte bırakıldığında hiç göremiyordu — yani modelin "ölçek genellemesi" (farklı boyutlardaki nesneleri tanıma yeteneği) zayıftı. Bu üçüncü geçiş bunu telafi ediyor.

## B.3. Neden ilk bulan geçişte durmuyoruz (erken çıkışın kaldırılması)

**Sorun**: İlk tasarımda bir geçiş kutu bulduğunda döngü hemen duruyordu ("bulduysan yeter, devam etme" mantığı — sezgisel olarak makul görünür ama yanlış çıktı). Yakın çekim bir fotoğrafta (`license-plate-frames...jpg`, gerçek plaka `SNIP3R`) bu ciddi bir hataya yol açtı: plaka görüntünün %73'ünü kaplayan dev bir nesneydi, ilk geçiş (1280px) onu tanıyamadı ama çerçevenin üzerindeki reklam yazı şeridini ("...FeelTheMPower") plaka sandı (en/boy oranı benzer). Erken çıkış yüzünden gerçek plakayı (yalnızca 320px gibi küçük bir taramada bulunabilen) hiç arama fırsatı olmadı.

**Neden bu tuzağa düşüldü**: Sezgi "büyük/net bir kutu bulundum, aramaya devam etmenin maliyeti var" idi. Ama gerçek dünya verisi bunu çürüttü: **yakın ve uzak plakalar farklı ölçeklerde görünür bulunabiliyor**, "ilk bulunan" her zaman "doğru olan" değil.

**Çözüm**: Artık üç geçişin **tamamı** çalışıyor, bulunan tüm kutular toplanıp **IoU (kesişim/birleşim oranı) tabanlı tekilleştirmeden** geçiriliyor (aynı fiziksel plakanın farklı geçişlerde bulunan kutuları birleştiriliyor, en yüksek güvenlisi tutuluyor), sonra güvene göre sıralanıp en fazla 5 aday olarak OCR'a gönderiliyor. Yanlış adaylar (yazı şeridi gibi) zaten sonraki OCR + yapısal doğrulama aşamasında elenir — yani "fazladan aday tarama" riski düşük, faydası yüksek.

## B.4. Klasik CV yedek dedektörü — neden ve nasıl

**Sorun**: Bazı fotoğraflarda YOLO, `conf=0.01` gibi saçma derecede düşük bir eşikte bile **hiçbir kutu üretmiyordu** — model o açı/durumla hiç eğitilmemiş demekti. Model yeniden eğitilene kadar bir yedek gerekiyordu.

**Değerlendirilen alternatifler:**

| Yaklaşım | Neden seçilmedi/seçildi |
|---|---|
| **Şablon eşleştirme (template matching)** | Plaka boyutu/açısı/aydınlatması çok değişken; şablonlar bu varyasyona dayanamaz |
| **Yalnızca renk tabanlı tespit** (Türk plakasının beyaz zemin + mavi TR bandı) | Aydınlatma değişimlerine aşırı duyarlı, gölgeli/gece görüntülerde işe yaramaz |
| **Blackhat morfolojisi + Sobel gradyanı** — SEÇİLEN | Klasik ANPR (otomatik plaka tanıma) literatüründeki en yerleşik, aydınlatmaya nispeten dayanıklı yöntem; doğrudan "plaka = parlak zemin üzerinde yoğun koyu metin" fiziksel tanımına dayanır |

**Nasıl çalışıyor** (`detect_plate_classical()`), adım adım:
1. Görüntü ~1200px çalışma genişliğine ölçeklenir (hız/detay dengesi).
2. **Blackhat morfolojisi**: `kapanış(görüntü) - görüntü` işlemi, parlak zemin üzerindeki koyu detayları (harfleri!) beyaz olarak öne çıkarır — bu, plakanın "imzasıdır". Yatay-uzun (25×7) bir çekirdek kullanılır çünkü plaka metin bloğu da yatay-uzundur.
3. **Parlaklık maskesi**: plaka zemini açık renkli olduğu için OTSU eşiklemesiyle görüntüdeki parlak bölgeler ayrıca maskelenir.
4. **X-yönlü Sobel gradyanı**: yan yana dizili harfler yoğun bir dikey-kenar (yatay gradyan) deseni oluşturur; bu adım "burada sık aralıklı dikey çizgiler var" diyen bölgeleri bulur.
5. **Bulanıklaştırma + yatay kapama**: ayrık duran harf kenarlarını tek bitişik blok haline getirir.
6. **Kesişim + temizlik**: gradyan maskesiyle parlaklık maskesinin kesişimi alınır (hem harf-yoğun hem zemini parlak olan bölgeler kalır — koyu zeminli yanlış pozitifler elenir).
7. **Kontur analizi ve skorlama**: kalan konturlar plaka geometrisine (en/boy oranı 1.3-8.0, min. boyut) göre filtrelenip `skor = parlaklık × metin_yoğunluğu × alan` formülüyle sıralanır, en iyi 3 aday döndürülür.

**İlginç bir ayrıntı**: en/boy oranı alt sınırı 2.0 değil **1.3** olarak bilerek gevşek tutuldu — çünkü morfolojik kapama bazen plaka blobunu üstündeki/altındaki tampon çizgisiyle birleştirip daha kareye yakın gösteriyor (ölçülen gerçek değer AR≈1.5). Sınırı gevşek tutmanın riski düşük çünkü kesin karar burada verilmiyor; bu fonksiyon yalnızca "aday üretiyor", nihai hüküm OCR + yapısal plaka kurallarında veriliyor.

---

# BÖLÜM C — KIRPIM TEMİZLİĞİ (plaka üzerindeki "kirlilik" giderme)

## C.1. Neden temizlik gerekiyor, hangi kirlilikler var

Türk plakaları üzerinde OCR'ı yanıltan dört tip unsur var: **mavi TR bandı** (sol kenarda), **turuncu muayene sticker'ı**, **siyah yapıştırmalar/etiketler**, ve **plaka çerçevesi/galeri reklamı altlığı**. Bunlar harf olmadıkları halde OCR motoruna "karakter gibi" görünüp yanlış okumalara yol açabiliyor.

## C.2. Neden kırpmak değil, boyamak (inpainting)

**İlk yaklaşım kırpmaydı**: mavi bandı/sticker'ı basitçe görüntüden kesip atmak. Sorun: bu unsurlar bazen gerçek karakterlere **çok yakın** duruyor — örneğin il kodu (`34`, `06`) mavi banda bitişik olabiliyor, kırpma payını hesaplarken küçük bir hata gerçek rakamları da götürebiliyor.

**Değerlendirilen alternatifler:**

| Yaklaşım | Sorun |
|---|---|
| **Doğrudan kırpma** | Yakın karakterleri riske atıyor (yukarıda anlatıldı) |
| **Bulanıklaştırma (blur)** | Kirliliği "gizler" ama tamamen kaldırmaz, OCR yine bir şeyler okumaya çalışıp gürültü üretebilir |
| **Inpainting (boyama)** — SEÇİLEN | Kirli bölgeyi çevresindeki dokudan **yeniden inşa eder** (görüntü tamamlama algoritması), böylece o bölge "hiç var olmamış gibi" beyaza döner, komşu karakterlere zarar vermez |

`cv2.inpaint` fonksiyonu **TELEA algoritmasını** kullanıyor — bu, hasarlı/maskelenmiş bir bölgeyi, sınırındaki piksellerden içe doğru "büyüterek" dolduran klasik bir görüntü tamamlama yöntemi (orijinal kullanım amacı eski fotoğraf restorasyonu/çizik giderme). Plaka bağlamında: mavi bandı maskeleyip inpaint'e verdiğinizde, algoritma çevresindeki beyaz zeminden yola çıkarak o bölgeyi düzgün beyaza boyuyor — kırpma gibi geometriyi bozmuyor.

## C.3. Her kirlilik türü için özel mantık

- **Mavi bant** (`detect_blue_band`): Sadece plakanın **sol %25'lik** kısmında aranır (Türk plakasında bandın konumu sabit; tüm görüntüde aramak yanlış pozitif riskini artırır). **HSV renk uzayında** (RGB değil!) mavi aralık taranır (H:100-130, S:80-255, V:50-255). *Neden HSV*: RGB'de "mavilik" ışık şiddetiyle karışır (parlak mavi ile karanlık mavi RGB'de çok farklı sayısal değerler alır); HSV'de Hue (renk tonu) kanalı ışık şiddetinden (Value) ayrıştığı için "bu piksel mavi mi" sorusu ışık koşulundan bağımsız cevaplanabilir.
- **Turuncu sticker** (`detect_and_remove_orange_sticker`): Varlığı önce **doğrulanır** (turuncu piksel oranı ≥%0.5), yoksa inpaint hiç çalıştırılmaz. *Neden bu koşullu kontrol var*: gereksiz yere her plakada inpaint çalıştırmak, sticker olmayan temiz bir plakada bile hafif bulanıklık/gürültü üretiyordu (deneyle görüldü) — "sorun yoksa dokunma" ilkesi.
- **Siyah sticker** (`detect_and_remove_black_sticker`): En zor ayrım çünkü karakterler de siyah. Üç filtre birden kullanılıyor: **en/boy oranı** (harfler dikey dikdörtgen, ~0.3-0.9 oranında; sticker'lar daha kare/yuvarlak), **konum** (sticker'lar genelde kenarlara/köşelere yapıştırılır), **dairesellik** (yuvarlak sticker'ları ayırt etmek için).
- **Çerçeve/altlık** (`remove_plate_frame_and_holder`): Canny kenar tespiti + `approxPolyDP` (kontur poligon yaklaştırması) ile en büyük dörtgen (plaka çerçevesi) bulunup içi kırpılır; alt %25'teki küçük konturlar (galeri ismi, telefon numarası gibi reklam yazıları) boyut eşiğine göre silinir.

---

# BÖLÜM D — GÖRÜNTÜ İYİLEŞTİRME VE 5 VARYANT

## D.1. Neden tek bir "en iyi" iyileştirme yerine 5 farklı varyant

**Sorun**: Deneyler gösterdi ki **hiçbir tek strateji her plakada kazanmıyor** — bir plakada mükemmel çalışan bir iyileştirme (örn. agresif kontrast artırma), başka bir plakada karakterleri bozup okumayı kötüleştirebiliyor. Bu, klasik bir **"tek çözüm yok, en iyi kombinasyon aranmalı"** durumu.

**Alternatif olarak düşünülebilecek yaklaşımlar:**

| Yaklaşım | Neden tercih edilmedi |
|---|---|
| **Tek "evrensel" en iyi filtre bulmaya çalışmak** | Deneyler gösterdi ki böyle bir filtre yok; her görüntü koşulu farklı işlem gerektiriyor |
| **Öğrenilmiş süper-çözünürlük modeli** (ESRGAN vb.) | Ek bir derin öğrenme modeli, ek GPU yükü ve karmaşıklık getirir; basit bicubic büyütme + klasik CV çoğu durumda yeterli sonuç veriyor |
| **5 varyant üretip hepsini dene, en iyisini seç** — SEÇİLEN | Hesaplama maliyeti kabul edilebilir (video'da bile), her varyant farklı bir başarısızlık modunu hedefliyor, "ensemble" mantığıyla toplam doğruluk tek stratejiden daha yüksek |

## D.2. Beş varyantın her biri neyi hedefliyor

| Varyant | Mekanizma | Hangi somut sorunu çözüyor |
|---|---|---|
| **temiz** | Bölüm C'deki tüm inpaint temizliği + CLAHE kontrast | Kirli/sticker'lı plakalar |
| **ham** | Yalnızca eğim düzeltme, temizlik YOK | Temizlik algoritmasının yanlışlıkla harfe zarar verme ihtimaline karşı "sigorta" — bazı görüntülerde temizlik gereksiz/zararlı çıkabiliyor |
| **sade** | Dar kırpım + 4x bicubic büyütme, başka hiçbir işlem yok | Bazen "ham" işleme bile zarar veriyor; en az müdahale bazı görüntülerde en isabetli sonucu veriyor |
| **ince** | Gri tonlama + büyütme + **dilasyon** (karakter inceltme) | A↔H karışıklığı (aşağıda detaylı) |
| **gölge** | Morfolojik aydınlatma düzleştirme + büyütme + inceltme | Benekli/dengesiz gölge (ağaç gölgesi vb.) |

## D.3. Bicubic büyütme — neden bu, öğrenilmiş süper-çözünürlük değil

Küçük plaka kırpımları (örn. 100×24 piksel) OCR için fiziksel olarak yetersiz — karakterler birkaç piksele sığıyor. Büyütmek gerekiyor. **Bicubic interpolasyon**, komşu piksellerin ağırlıklı ortalamasını alarak ara değerleri "tahmin eden" klasik bir yeniden örnekleme yöntemi — hızlı, ek model gerektirmez, GPU'da anlık çalışır. Öğrenilmiş süper-çözünürlük modelleri (yapay zekâ ile "eksik detayı halüsinasyon eder gibi tamamlayan" modeller) teorik olarak daha keskin sonuç verebilir ama bu projede ek karmaşıklığa değecek bir kazanım göstermedi — bicubic + doğru kontrast/keskinlik ayarları yeterli oldu.

## D.4. CLAHE — neden global histogram eşitleme değil

Kontrast artırma için en basit yöntem **global histogram eşitleme**dir (tüm görüntünün parlaklık dağılımını yeniden dağıtır). Sorun: bu, görüntünün bir kısmı zaten iyi aydınlatılmışken diğer kısmı gölgedeyse, ya gölgeyi yetersiz açar ya da aydınlık kısmı aşırı parlatır. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** görüntüyü küçük karolara (tile) bölüp her karoyu **yerel olarak** eşitler — böylece hem gölgeli hem aydınlık bölgeler kendi bağlamında iyileştirilir. "Contrast Limited" kısmı, aşırı gürültü büyütmesini önlemek için bir tavan koyuyor (limitsiz yerel eşitleme gürültüyü de agresifçe büyütür).

## D.5. A↔H karışıklığı ve "ince" varyantının fiziksel mantığı

**Gözlenen somut hata**: `20 AFB 280` plakası sürekli `20 HFB 280` olarak okunuyordu. Neden? Düşük çözünürlükte bulanıklık, karakter çizgilerini kalınlaştırıp birbirine yapıştırıyor — A harfinin içindeki küçük üçgen boşluk bulanıklıktan dolayı doluyor, görsel olarak H'ye benziyor.

**Çözüm mantığı — tek yönlü bir fiziksel gerçeğe dayanıyor**: Gri tonlamalı bir görüntüde **dilasyon** işlemi parlak (beyaz) bölgeleri genişletir — yani siyah karakter çizgilerini **inceltir**. İnceltme uygulanınca yapışan çizgiler ayrışır, A'nın iç boşluğu yeniden ortaya çıkar. Kritik nokta: bu mekanizma **tek yönlüdür** — inceltme, H'nin zaten ayrı duran iki dikey çizgisini asla A'nın kapalı üçgenine "birleştiremez". Yani eğer "ince" varyant yüksek güvenle A okuyorsa ve diğer (kalın) varyantlar aynı pozisyonda H okuyorsa, fiziksel olarak **A'nın doğru olma ihtimali çok daha yüksektir** — bu yüzden skorlama sisteminde bu duruma özel +0.35 bonus tanımlandı.

**Neden erozyon değil dilasyon**: Erozyon tam tersini yapardı (siyahı kalınlaştırır) — bu, zaten var olan yapışma sorununu şiddetlendirirdi. Dilasyon doğru yönde çalışıyor çünkü hedef "ayrıştırmak", "kalınlaştırmak" değil.

## D.6. Gölge varyantı — morfolojik aydınlatma düzleştirme

**Sorun**: Ağaç gölgesi gibi benekli, dengesiz aydınlatmada CLAHE bile yetmiyordu — bir gerçek test görüntüsünde (`arabalar.jpeg`) gölgedeki plaka hiç okunamıyordu.

**Değerlendirilen alternatifler:**

| Yaklaşım | Neden tercih edilmedi/edildi |
|---|---|
| **Retinex tabanlı aydınlatma düzeltmesi** | Daha karmaşık, parametre ayarı hassas, bu ölçekte gereksiz |
| **Homomorfik filtreleme** (frekans uzayında aydınlatma/yansıma ayrıştırma) | Uygulaması ve ayarı daha karmaşık, gerçek zamanlı video için ek yük |
| **Morfolojik arka plan kestirimi + bölme** — SEÇİLEN | Basit, hızlı, iyi anlaşılan bir teknik; bu spesifik sorun (yavaş değişen gölge deseni) için yeterli |

**Nasıl çalışıyor**: Görüntünün arka plan aydınlatma haritası **morfolojik kapama** (31×31 kernel) ile kestirilir — bu büyüklükteki bir kernel karakterleri tamamen "siler", geriye yalnızca yavaş değişen (gölge/aydınlık geçişi gibi) düşük frekanslı bileşen kalır. Orijinal görüntü bu haritaya **bölünür** (`cv2.divide`) — bu işlem, "bu piksel yerel ortalamasına göre ne kadar açık/koyu" bilgisini normalize eder, yani gölge deseni sadeleşir ve tüm karakterler eşit kontrasta gelir.

---

# BÖLÜM E — OCR MOTORU SEÇİMİ VE ÇOK STRATEJİLİ OYLAMA

## E.1. Neden EasyOCR — alternatiflerle karşılaştırma

| Motor | Artı | Eksi | Karar |
|---|---|---|---|
| **Tesseract** | Klasik, hafif, Türkçe dil paketi var | Düşük çözünürlükte (<30px karakter) zayıf performans | Kullanılmadı — plaka kırpımları genelde bu eşiğin altında |
| **PaddleOCR** | EasyOCR'a benzer, bazı senaryolarda daha hızlı | Kalite genelde benzer, ek bağımlılık | Değerlendirilmedi ama gelecekte ensemble'a eklenebilir (bkz. Bölüm F) |
| **Ticari bulut API'leri** (Azure Read, Google Vision, AWS Textract) | Genelde en yüksek ham doğruluk | Ücretli, internet bağlantısı gerektirir, video'da her kare/kırpım için maliyet katlanır, gecikme (latency) video akışını yavaşlatır | Kullanılmadı — yerel/ücretsiz/hızlı çalışma önceliği vardı |
| **Claude Vision gibi büyük dil modelleri** | Bağlamsal okuma yapabilir ("bu Türk plakası" gibi) | Yüksek gecikme, token maliyeti, video ölçeğinde pratik değil | Kullanılmadı — aynı sebep |
| **EasyOCR** — SEÇİLEN | Yerel çalışır (internet gerekmez), ücretsiz, GPU destekli (hızlı), **allowlist** özelliği (yalnızca belirli karakterleri tanımaya zorlanabilir) var | Ham doğruluğu ticari API'lerden düşük olabilir | Bu projenin gereksinimlerine (yerel, hızlı, ücretsiz, video ölçeğinde çalışabilir) en uygun seçenek |

**`allowlist` özelliğinin önemi**: EasyOCR'a "yalnızca şu karakterleri okuyabilirsin" (örn. Türk plakası için `ABCDEFGHIJKLMNOPRSTUVYZ0123456789` — dikkat: W, Q, X yok) denilebiliyor. Bu, motorun bağlam dışı karakterler üretmesini büyük ölçüde engelliyor ve doğruluğu ciddi artırıyor — ama aynı zamanda **yabancı plaka sorununun da kaynağı** oldu (Bölüm G'de anlatılıyor), çünkü W/Q/X içeren bir plaka bu allowlist ile hiç doğru okunamıyor.

## E.2. Neden tek OCR çağrısı yetmiyor — çoklu strateji + oylama

Bir görüntü varyantı üzerinde bile EasyOCR'ı farklı **ön-işleme stratejileriyle** (örn. renkli görüntü vs. OTSU ile ikili/binary hale getirilmiş görüntü) birden fazla kez çalıştırmak, farklı sonuçlar üretebiliyor. `read_plate_ocr()` bu yüzden her 5 varyantı 2 stratejiyle (yaklaşık 10 OCR çağrısı) deniyor ve sonuçları **karakter bazlı oylamaya** sokuyor.

**Karakter oylaması nasıl çalışıyor**: Örneğin üç farklı okuma `34ABC12`, `34A8C12`, `34ABC12` şeklinde geldiyse, her pozisyondaki en sık görülen karakter seçilerek nihai metin (`34ABC12`) sentezleniyor. Bu, istatistikte **çoğunluk oylaması (majority voting)** olarak bilinen, ensemble öğrenmenin temel bir tekniği — tek bir "hakem" yerine birden fazla bağımsız görüşün ortak paydası alınıyor.

## E.3. Varyant AİLESİ kavramı — neden bağımsızlık kritik

**Sorun keşfedildi**: İlk tasarımda her varyant-strateji kombinasyonu (temiz/renkli, temiz/otsu, ham/renkli...) **eşit ve bağımsız** kanıt sayılıyordu. Ama temiz/ham/sade hepsi **aynı "kalın karakterli" görüntü ailesinden** geliyor — düşük çözünürlükte üçü de **aynı kök nedenle** aynı hatayı yapabiliyor (örn. üçü de A'yı H okuyabiliyor, çünkü hepsi aynı bulanıklık sorununu paylaşıyor). Bunları "3 ayrı kanıt" gibi saymak istatistiksel olarak yanlıştı — gerçekte bu tek bir korelasyonlu hata kaynağıydı, 3 tane değil.

**Bu neden önemli — istatistik mantığı**: Eğer 3 bağımsız kaynak aynı yanlış cevaba varırsa, bu tesadüf olma ihtimali düşüktür (güvenilir bir sinyaldir). Ama eğer 3 kaynak **aynı kök nedenden** dolayı aynı hatayı yapıyorsa, bu "3 bağımsız kanıt" değil, "1 kanıtın 3 kopyası"dır — güvenilirliği yanlışlıkla şişirir. Bu, gerçek yaşamda "3 kişi aynı yanlış haberi okudu, öyleyse doğru olmalı" mantık hatasına benzer.

**Çözüm**: Konsensüs artık yalnızca **farklı ailelerin** (kalın: temiz/ham/sade vs. ince/gölge) aynı sonuca varmasıyla sayılıyor — bu, gerçekten bağımsız doğrulamayı garantiliyor.

## E.4. Skorlama formülünün bileşenleri

Her okuma adayına şu bileşenlerden oluşan bir skor veriliyor:

1. **Geçerlilik bonusu** — yapısal olarak geçerli bir Türk plakasıysa, "onarım maliyetiyle" (kaç karakter Bölüm F'deki tablo kullanılarak değiştirildi) ters orantılı bir bonus alır — az onarımla geçerli olan, çok onarımla zorla geçerli yapılandan daha güvenilir sayılır.
2. **Konsensüs bonusu** — farklı aileler aynı formata varmışsa.
3. **A/H çözümleme bonusu** — Bölüm D.5'te anlatılan tek yönlü mekanizma.
4. **Altdizi bonusu** — bir aday, başka geçerli bir adayı **altdizi** olarak kapsıyorsa (yani karakter kaybı olmuş kısa okumayı tam olarak içeriyorsa), bu karakter düşmesini telafi eden bir bonus alır. **Ek koşul**: uzun adayın aile desteği kısa adayınkinden az olamaz — bu koşul sonradan eklendi çünkü bir gerçek hatada, gölge varyantının TR bandı kalıntısından ürettiği **sahte bir karakter** (`34SN5953` — fazladan 'S'), doğru kısa okumayı (`34N5953`) yanlışlıkla yeniyordu.

## E.5. Tek-kaynak güvencesi

Eğer kazanan okuma **yalnızca tek bir varyant ailesinden** geliyorsa ve güveni orta seviyedeyse (<0.60), sonuç zorla "şüpheli" bandına indiriliyor (güven 0.30'a düşürülür) — bu, CSV kalite kapısından geçemez hale gelir. **Gerekçe**: gölgeli/bozuk plakalarda tek bir varyant yapısal olarak geçerli görünen ama **yanlış** bir okuma üretebiliyor (gerçek örnek: gölgedeki bir plaka tek kaynaktan `34 JLB 191` okundu, gerçeği `34 JA 8191`'di). Bağımsız doğrulaması olmayan orta-güvenli bir okumayı kesin diye sunmak yanlıştı — "emin değilim" demek, yanlış "eminim" demekten iyidir.

---

# BÖLÜM F — TÜRK PLAKA FORMAT DOĞRULAMA VE ONARIM

## F.1. Neden salt OCR çıktısına güvenilmiyor — yapısal kurallar neden gerekli

OCR, görüntüyü karaktere çevirir ama **bağlam bilmez** — "bu üçüncü karakterin rakam mı harf mi olması gerektiğini" bilmiyor, sadece görsel olarak ne gördüğünü söylüyor. Türk plakaları katı bir yapıya sahip (`99 X 9999` / `99 XX 999-9999` / `99 XXX 99-999` — il kodu + 1-3 harf + 2-4 rakam, il kodu 01-81 arası). Bu yapısal bilgi, OCR'ın belirsiz kaldığı durumlarda **doğru kararı vermesine yardım edebilir**.

## F.2. Harf↔Rakam onarım tablosu

Bazı karakterler görsel olarak neredeyse ayırt edilemez ve OCR sık sık karıştırır: `O`/`Q`↔`0`, `I`/`L`↔`1`, `Z`↔`2`, `S`↔`5`, `G`↔`6`, `T`↔`7`, `B`↔`8`. `format_turkish_plate_ex()` fonksiyonu, plakanın **beklenen pozisyonuna** (harf mi rakam mı olması gerektiği) bakarak bu karışıklıkları düzeltmeye çalışır — örneğin "rakam beklenen yerde S görüldüyse, muhtemelen 5'tir" mantığıyla onarım yapar. Her onarım bir "maliyet" olarak sayılır (Bölüm E.4'teki skorlamada kullanılıyor) — az onarımla geçerli hale gelen okuma, çok onarımla "zorla" geçerli yapılandan daha güvenilir kabul edilir.

---

# BÖLÜM G — YABANCI PLAKA AYRIMI

## G.1. Sorunun kökeni

Bölüm E.1'de bahsedilen `allowlist` (yalnızca Türk plakasında kullanılan karakterler: W, Q, X hariç) hem faydalı hem de bir yan etkiye sahip: **yabancı bir plaka** (örn. Alman `WI TJ 473`) bu allowlist ile okunmaya çalışıldığında W/Q/X harflerini asla üretemiyor, dolayısıyla bozuk bir metin çıkıyor. Daha da kötüsü, Bölüm F'deki onarım mekanizması bu bozuk metni **zorla** geçerli bir Türk plakasına (`16 TJ 473` gibi) çevirebiliyordu — yani sistem kendinden emin ama **tamamen sahte** bir kayıt üretiyordu. Bu, "yanlış cevap vermek, cevap vermemekten kötüdür" ilkesinin en net ihlaliydi ve acilen çözülmesi gerekiyordu.

## G.2. Çözüm — iki adımlı doğrulama

1. **Tam alfabe çapraz kontrolü**: Eğer sonuç "geçerli Türk plakası" olarak çıkarsa, **tüm 5 varyantta ayrıca** tam alfabeyle (W/Q/X dahil) bir doğrulama okuması yapılır. Eğer W/Q/X **en az 2 farklı varyantta** görülürse, plaka yabancı sayılır ve Türk formatına zorlanmaz. *(Neden 2 varyant, 1 değil: bir denemede tek varyantlık bir "hayalet" W okuması yüzünden gerçek bir Türk plakası — `06 AB 8655` — yanlışlıkla yabancı işaretlenmişti. İki varyant şartı, rastgele/tekil hayalet okumaları eleyip gerçek yabancı kanıtını (genelde çoğu varyantta tutarlı görünür) ayırt ediyor.)*
2. **Standardizasyon**: Türk formatına uymayan ama W/Q/X kanıtı taşımayan plakalar da (örn. bir Litvanya süs plakası, `SNIP3R`) tutarlı biçimde `[YABANCI]` etiketi alır. Kural: geçersiz format + güven ≥0.40 → net okunmuş ama Türk formatında değil → YABANCI. Güven <0.40 ise etiketlenmiyor çünkü bu muhtemelen kötü okunmuş bir Türk plakasıdır — ona da "yabancı" demek başka bir yanlış iddia olurdu.

---

# BÖLÜM H — ÇOKLU PLAKA DESTEĞİ VE CSV KALİTE KAPISI

Bir fotoğrafta birden fazla araç/plaka olabilir. En fazla 5 aday kutu ayrı ayrı işlenir ve **geçerli olan hepsi** sonuç listesine eklenir (ilk tasarımda yalnızca ilk geçerli adayda erken çıkılıyordu — bu, aynı fotoğraftaki ikinci bir plakayı sistematik olarak gözden kaçırıyordu).

**CSV'ye yazma şartı**: yapısal olarak geçerli **ve** güven ≥%35. Bu eşiğin altındaki veya geçersiz okumalar **sessizce atlanır** — hata olarak raporlanmaz, sadece loglanmaz. Felsefe: "yanlış bir kayıt yazmak, hiç yazmamaktan kötüdür" — bir CSV'de %100 doğru ama eksik kayıtlar, %90 doğru ama içinde sahte kayıtlar olan bir CSV'den her zaman daha değerlidir (özellikle bu veriler bir karar/rapor için kullanılacaksa).

---

# BÖLÜM I — VİDEO HATTI: NEDEN VE NASIL FARKLI BİR MİMARİ GEREKTİ

## I.1. Video, fotoğraftan neden temelde farklı bir problem

Fotoğrafta tek bir kare vardır, tek bir karar anı vardır. Videoda ise **zaman** bir boyut olarak eklenir — aynı plaka onlarca farklı karede, farklı netlik/açı/aydınlatmayla görünür. Bu hem bir **fırsat** (birden fazla bağımsız kanıt, tek karenin çözemediği hataları çözebilir) hem bir **zorluk** (60 kare/saniyelik bir videoda her kareyi tam OCR hattından geçirmek hesaplama olarak imkânsız — kare başına ~1-2 saniye süren bir işlemi saniyede 60 kez çalıştıramazsınız).

## I.2. Video, fotoğraf hattının kodunu tekrar mı yazdı?

**Hayır — doğrudan onu kullanıyor.** `video_plaka.py` dosyasının başında `colab_local.py`'den beş fonksiyon aynen içe aktarılır:

```python
from colab_local import (
    initialize_model,          # YOLO ağırlığını yükler
    initialize_ocr,            # EasyOCR okuyucusunu başlatır
    process_plate_candidate,   # TEK KIRPIMIN uçtan uca işlenmesi
    _character_vote,           # karakter bazlı oylama
    format_turkish_plate_ex,   # Türk format onarımı + doğrulama
)
```

Kritik olan `process_plate_candidate` — bu, Bölüm B-G'de anlatılan **her şeyi** (eğim düzeltme, kirlilik temizliği, 5 varyant, çoklu strateji OCR + oylama + skorlama, yabancı plaka kontrolü) tek bir çağrıda çalıştıran fonksiyon. Video hattı bir plaka kırpımını OCR'a soktuğunda, fotoğraf hattıyla **birebir aynı işlemden** geçiriyor — aynı mavi bant boyama, aynı 5 varyant, aynı A/H çözümleme, aynı yabancı plaka kontrolü.

**Video'ya özgü olup fotoğrafta hiç olmayan kısımlar:**
- İz (track) kavramı ve IoU eşleştirme — fotoğrafta "aynı plaka bir sonraki karede de var mı" sorusu yok
- Kırpım kalitesi ölçümü + zamansal çeşitlilik filtresi — tek karede gerek yok
- Artımlı okuma / erken karar mimarisi (Bölüm I.6) — tamamen video'ya özgü zamanlama sorunu
- Kareler arası konsensüs ve tekrar önleme — "bu plaka daha önce görüldü mü" sorusu yalnızca video'da anlamlı
- Canlı izleme penceresi

Sonuç: **"plakayı nasıl okuruz" sorusunun cevabı tamamen `colab_local.py`'de yaşıyor; "hangi kırpımı ne zaman okumaya değer" sorusunun cevabı `video_plaka.py`'de.** Fotoğraf hattında yapılan bir iyileştirme (örn. yeni bir varyant eklemek) otomatik olarak video hattına da yansır — hiçbir yerde kod kopyalanmadı.

## I.3. Örnekleme — neden her kare değil

**Sorun**: 60fps bir videoda her kareyi işlemek hem gereksiz (ardışık kareler neredeyse birebir aynı görüntüdür, yeni bilgi taşımaz) hem de hesaplama olarak imkânsız.

**Değerlendirilen alternatifler:**

| Yaklaşım | Neden tercih edilmedi/edildi |
|---|---|
| **Sahne değişikliği tespiti** (yalnızca görüntü önemli ölçüde değiştiğinde işle) | Karmaşık, plaka hareketi genelde kamera hareketinden ayırt edilmesi zor bir "küçük değişiklik" |
| **Hareket tespiti tabanlı örnekleme** | Ek bir hesaplama katmanı, dashcam videosunda neredeyse her şey hareket halinde (kamera da hareket ediyor) |
| **Sabit aralıklı örnekleme** — SEÇİLEN | Basit, öngörülebilir, yeterli: plaka en az ~1 saniye kadrajda kaldığı sürece kaçırılmaz |

**Nasıl çalışıyor**: Her `FRAME_SKIP=6` karede bir YOLO tespiti çalıştırılıyor (60fps'te saniyede 10 örnek). Aradaki kareler `cap.grab()` ile **decode edilmeden** atlanıyor — bu önemli bir performans detayı: video dosyasından bir kareyi okumak iki adımdır, `grab()` (kareyi kaynaktan al ama açma/çözme yapma) ve `retrieve()` (asıl piksel verisini çöz). Atlanacak kareler için yalnızca `grab()` çağrılıyor, pahalı çözme (decode) işlemi hiç yapılmıyor — zaman çizgisinde hiçbir kare atlanmıyor, sadece gereksiz çözme maliyetinden kaçınılıyor.

**Çözünürlüğe göre uyarlama**: 4K karede YOLO'nun giriş boyutu `imgsz=1920`, 1080p ve altında `1280` kullanılıyor. Neden: 4K'yı direkt 1280'e küçültmek, uzak/küçük plakaları orantılı olarak 3'te 1'e küçültüp kaybettiriyordu (Bölüm B.2'deki aynı sorunun video versiyonu).

## I.4. İz takibi — neden basit IoU eşleştirme, gelişmiş takip algoritmaları değil

**Sorun**: Ardışık örneklerde tespit edilen kutuların **aynı fiziksel plakaya mı yoksa farklı bir plakaya mı** ait olduğunu bilmemiz gerekiyor — aksi halde her örnekte "yeni bir plaka gördüm" sanılır ve hiçbir konsensüs/oylama kurulamaz.

**Değerlendirilen alternatifler:**

| Yaklaşım | Ne yapar | Neden tercih edilmedi/edildi |
|---|---|---|
| **Kalman filtresi** | Nesnenin hız/konumunu tahmin edip bir sonraki karede nerede olacağını öngörür | Video'daki plakalar (araçlar) çoğunlukla öngörülebilir, yavaş/düzenli hareket ediyor; bu karmaşıklığın getirisi bu ölçekte sınırlı |
| **Optical flow (optik akış)** | Piksel hareketini takip ederek nesneyi kare boyunca izler | Hesaplama maliyeti IoU'dan yüksek, 10Hz örneklemede kareler zaten birbirine yakın, gerek görülmedi |
| **DeepSORT / ByteTrack gibi öğrenilmiş çoklu nesne takipçileri** | Görünüm (appearance) özelliklerini de kullanarak takip yapar, kalabalık sahnelerde daha güçlü | Ek model + ek hesaplama yükü; bu projenin trafik yoğunluğu ve 10Hz örnekleme sıklığında gerekmedi |
| **Basit IoU (kesişim/birleşim oranı) eşleştirme** — SEÇİLEN | Ardışık örnekteki kutuları, örtüşme oranına göre eşleştirir | Basit, hızlı, 10Hz örneklemede ardışık kutular zaten büyük ölçüde örtüşüyor — bu sıklıkta IoU tek başına yeterli sonuç veriyor |

**Nasıl çalışıyor**: Her yeni tespit kutusu, mevcut aktif izlerin (`PlateTrack` nesneleri) kutularıyla IoU hesaplanarak karşılaştırılır; en yüksek örtüşmeye sahip iz varsa ona eşlenir (güncellenir), yoksa yeni bir iz açılır. Bir iz belirli bir süre (`MISS_LIMIT=15` örnek, ~1.5 saniye) hiç güncellenmezse "kapandı" sayılır.

## I.5. Kırpım kalitesi ve zamansal çeşitlilik

Bir iz boyunca biriken tüm kırpımları OCR'a sokmak hem yavaş hem gereksiz — en kaliteli birkaçı yeterli. **Kalite ölçütü**: genişlik × netlik (Laplace varyansı — görüntüdeki keskin kenar miktarını ölçer; bulanık görüntülerde bu değer düşüktür çünkü kenarlar yumuşamıştır).

**Sorun (sonradan keşfedildi)**: "en iyi 5 kırpım"ı seçerken, kalite sıralaması genelde **ardışık kareleri** (neredeyse birebir aynı anı) seçiyordu — yani "5 bağımsız kanıt" sanılan şey aslında "1 anın 5 kopyası"ydı, hataları da ortaktı (Bölüm E.3'teki aile sorununun bir başka versiyonu). Çözüm: `CROP_GAP=0.4` saniyeden daha yakın zamanlı kırpımlar birbirini eler (daha kalitelisi kalır) — böylece tutulan kırpımlar izin ömrüne **yayılır** ve gerçekten bağımsız kanıt taşır.

## I.6. Artımlı okuma mimarisi — projenin en son ve en kritik kararı

### Sorun nasıl keşfedildi

İlk video mimarisi "izi bitir, sonra topluca oku" (**batch**) mantığındaydı: OCR yalnızca bir iz kapandığında (araç kadrajdan çıkıp ~1.5 saniye güncellenmediğinde) çalışıyordu. Bu, canlı izleme penceresinde test edilirken çok net bir kusur ortaya çıkardı: **kamerayla benzer hızda giden, kadrajdan hiç çıkmayan bir aracın** (örneğin tam önümüzdeki araç) plakası **sonsuza dek işlenmeden kalıyordu** — ekranda yeşil bir takip kutusu görünüyordu ama hiçbir zaman bir okuma sonucu üretilmiyordu, çünkü izin "kapanması" hiç gerçekleşmiyordu.

### Alternatifler ve araştırma

Bu noktada durup gerçek dünyadaki ANPR (otomatik plaka tanıma) sistemlerinin — özellikle bariyer kontrol/geçiş sistemleri gibi düşük gecikme gerektiren uygulamaların — bu sorunu nasıl çözdüğü araştırıldı. Bulgular:

| Yaklaşım | Nasıl çalışır | Bizim durumumuza uygunluk |
|---|---|---|
| **Batch (izi bitir, sonra oku)** — eski mimarimiz | Basit ama gecikmeli, sürekli görünen nesnede hiç tetiklenmeyebilir | Bariyer açma gibi kullanımlarda kabul edilemez — araç geçmeden karar gerekir |
| **Her karede bağımsız OCR + en son sonucu göster** | Her an bir cevap var ama tutarsız/gürültülü (her karede farklı okuma çıkabilir) | Hesaplama maliyeti çok yüksek, tek kare güvenilirliği düşük |
| **Artımlı (incremental) konsensüs** — GERÇEK ANPR SİSTEMLERİNİN STANDARDI, BİZİM DE SEÇTİĞİMİZ | İz açıkken belirli aralıklarla OCR tekrar denenir; yeterli bağımsız kanıt (≥2 kırpım aynı sonuca varır VEYA tek okuma çok yüksek güvenle gelir) birikir birikmez sonuç **hemen** kesinleşir — izin kapanması beklenmez | Hem düşük gecikme (gerçek sistemler gibi) hem de çoklu-kanıt güvenilirliği (batch'in avantajını kaybetmeden) |

### Nasıl uygulandı

Eski tek parça `finalize_track()` fonksiyonu ikiye ayrıldı:
- **`attempt_read()`** — salt okuma denemesi (OCR + oylama + çapraz doğrulama kontrolü yapar, CSV'ye YAZMAZ). Hem iz açıkken artımlı kontrolde hem iz kapanışında ortak kullanılıyor — kod tekrarı yok.
- **`log_result()`** — çapraz doğrulanmış bir sonucu kalite kapısından geçirip CSV'ye yazar (tekrar önlemeyle birlikte).

Her `PlateTrack` nesnesi artık bir **`resolved`** (çözüldü) bayrağı taşıyor. Ana döngüde, bir ize her yeni kırpım eklendiğinde şu kontrol yapılıyor: iz henüz çözülmemişse VE yeterli yeni kanıt birikmişse (ilk denemede en az 3 kırpım, sonraki denemelerde en az 2 YENİ kırpım — bu, `CROP_GAP` sayesinde doğal olarak ~0.8 saniyelik bir "yavaşlatma" sağlıyor, her örnekte pahalı OCR çalıştırılmasını önlüyor), `attempt_read()` çağrılıyor. Çapraz doğrulama sağlanırsa iz hemen `resolved=True` işaretlenip sonuç loglanıyor (ya da zaten bilinen bir plakayla eşleşiyorsa "tekrar" olarak elenip yine de iz çözülmüş sayılıyor — cevap zaten bilindiği için tekrar tekrar OCR çalıştırmaya gerek yok).

**Bir ince ayar hatası ve düzeltmesi**: İlk uygulamada `resolved` bayrağı yalnızca sonuç **gerçekten CSV'ye yazıldığında** işaretleniyordu. Bu, zaten bilinen (dedup'a düşen) bir plakanın izini **50 kırpıma kadar tekrar tekrar** OCR'a sokuyordu — çünkü "tekrar" sayılıp yazılmadığı için sistem hâlâ "çözülmedi" sanıyordu. Düzeltme: çapraz doğrulama sağlanır sağlanmaz (CSV'ye yazılsın ya da tekrar olarak elensin fark etmeksizin) iz çözülmüş sayılıyor — cevap zaten biliniyor, gereksiz tekrar hesaplama önleniyor.

İz kapanması artık yalnızca bu eşiğe **hiç ulaşamamış** zayıf izler için bir son çare denemesi — eskisi gibi ana/tek karar noktası değil.

### Kareler arası konsensüs — kademeli güven

≥3 bağımsız kırpım birebir aynı okursa güven tabanı %60'a yükseltiliyor (üç ayrı anın aynı 7-8 karakteri **tesadüfen** üretmesi istatistiksel olarak ihmal edilebilir). Ama yalnızca 2 kırpım aynıysa **taban yok** — yalnızca gerçek güvenlerin en yükseği alınıyor. *Neden bu ayrım var*: gerçek bir hatada, `35 FR 056` plakası iki farklı karede de **aynı sistematik hatayla** `35 G 0056` okunmuştu (aynı font/bozulma yüzünden hata korele oluyordu) — 2 kırpımlık taban bu yanlışı CSV'ye taşımıştı. İki kare, sistematik/korele bir hatayı dışlamak için istatistiksel olarak yetersiz; üç kare yeterli görüldü.

### Tekrar önleme (dedup)

Aynı plaka `DEDUP_WINDOW=120` saniye içinde tekrar loglanmıyor — dashcam'de öndeki bir araç dakikalarca takip edilebilir, her görünüş yeni bir kayıt anlamına gelmemeli. Eşleşme kontrolü **yakın eşleşme** mantığıyla yapılıyor (birebir aynı metin, ya da eş uzunlukta tek karakter farkı, ya da bir karakter eksik/fazla altdizi ilişkisi) — çünkü hayalet karakter ekleme/düşürme (örn. çerçeve kenarından gelen sahte bir '1') aynı plakanın iki farklı metin olarak iki kez loglanmasına yol açabiliyordu; bu yakın-eşleşme kontrolü bunu önlüyor.

## I.7. Canlı izleme penceresi ve bilinen bir ödünleşim

Video oynatılırken aktif izler ekranda kutuyla gösteriliyor: **yeşil** = hâlâ okunmaya çalışılıyor, **camgöbeği** = çapraz doğrulama sağlandı (plaka metni kutunun üzerinde görünüyor). Alt siyah şeritte video zamanı, işleme hızı ve son loglanan plakalar listeleniyor.

**Bilinen bir ödünleşim**: sistem tek iş parçacıklı (single-threaded) çalışıyor — bir izin OCR denemesi (özellikle iz kapanışındaki "son çare" denemesi) çalışırken video karesi okuma/gösterme de aynı iş parçacığında bloke oluyor. Bu, canlı izlemede her araç geçişi civarında kısa bir duraksama olarak hissediliyor. Donanım hızı (GPU) bu duraksamanın **süresini** belirliyor ama duraksamanın kendisi donanımdan bağımsız, mimari bir sonuç. Çözümü OCR'ı ayrı bir iş parçacığında (thread) çalıştırıp video oynatmayı hiç kesintiye uğratmamak olurdu — henüz uygulanmadı, bir sonraki adım olarak duruyor.

---

# BÖLÜM J — TEST EDİLEN GERÇEK GÖRÜNTÜLER VE BULGULAR

## J.1. Fotoğraf testleri (`arabalar/` klasörü)

| Görüntü | Zorluk | Sonuç ve hangi karar bunu çözdü |
|---|---|---|
| `145874607...jpg` | Standart | `42 AEH 738` — %99+ güvenle doğru |
| `727919538.jpg` / `728951572.jpg` | Aynı araç, biri eğik+bulanık 83×35px kırpım | `06 ABY 325` — biri kesin doğru, diğeri düşük güvenle dürüstçe "belirsiz" işaretlendi (uydurulmadı) |
| `740271856.jpg` | YOLO conf=0.01'de bile hiç bulamıyordu | Çok geçişli tespit + klasik CV yedeğiyle çözüldü (Bölüm B) → `42 ALD 013` |
| `IMG-20220712...jpg` | A harfi sürekli H okunuyordu (`20 HFB 280`) | "ince" varyantı + A/H fiziksel çözümleme eklendi (Bölüm D.5) → `20 AFB 280` |
| `araba1.jpg` | Gölge varyantı TR bandından sahte 'S' üretip yanlış kazanıyordu | Altdizi bonusuna aile-desteği koşulu eklendi (Bölüm E.4) → doğru `34 N 5953` |
| `arabalar.jpeg` | Aynı karede 3 plaka, biri ağaç gölgesinde | Çoklu plaka desteği (Bölüm H) + "gölge" varyantı (Bölüm D.6); gölgedeki plaka tam okunamayınca dürüstçe "şüpheli" işaretlenip CSV dışı bırakıldı |
| `europ2.jpg` | Alman plakası (`WI TJ 473`) Türk formatına zorla oturtuluyordu | Tam alfabe çapraz kontrolü eklendi (Bölüm G) → `[YABANCI]` etiketiyle CSV dışı |
| `license-plate-frames...jpg` | Yakın çekimde YOLO çerçevenin reklam yazısını plaka sanıyordu | Erken çıkış kaldırıldı (Bölüm B.3) → gerçek plaka `SNIP3R` bulunup doğru şekilde `[YABANCI]` işaretlendi |
| `images.jpeg` | Açılı + kısmen gölgeli, doğru cevabın parçaları farklı varyantlara dağılmış | Kısmen çözüldü, bilinen sınır olarak bırakıldı (Bölüm K) |

## J.2. Video testleri

- **10 dakikalık 1080p dashcam videosu** (ilk video mimarisiyle): 3 plaka bulundu, hepsi doğru.
- **4 dakikalık 4K trafik videosu** (4K + çapraz doğrulama + sonradan artımlı mimariyle): 24-25 plaka bulundu, ~%92 doğruluk. 2 bilinen hata tespit edildi (Bölüm K'de detaylı).

---

# BÖLÜM K — BİLİNEN SINIRLAR (dürüstçe, çözülmemiş olanlar dahil)

- **OCR duraksaması (canlı izlemede)**: Bölüm I.7'de anlatıldı — thread'e taşınarak çözülebilir, henüz yapılmadı.
- **İki satırlı plaka karakter sırası karışması**: motosikletlerde yaygın iki satırlı format (üstte il kodu+harf, altta rakamlar) OCR fragment birleştirme mantığı tarafından yanlış sıralanabiliyor — gerçek `34 ZM 1487` sistem tarafından `34 I 4872` yazıldı. **Kök neden**: parça birleştirme mantığı yalnızca x-koordinatına (yatay konum) göre sıralama yapıyor, satır (y-koordinatı) ayrımı yapmıyor. Bilinçli olarak şimdilik düzeltilmedi — çözümü parçaları önce y-kümelemesine (satır tespiti) sonra x'e göre sıralamaktan geçer.
- **Sistematik optik hata**: plaka çok küçük/uzak/bulanıksa TÜM OCR varyantları aynı yanlış karakteri tutarlı biçimde üretebiliyor (`34 TE 2469` → `13 IE 2469`, tüm varyantlarda tutarlı). Bölüm E.3'teki çapraz doğrulama mekanizması yalnızca **bağımsız** hatalara karşı korur; ortak/sistematik bir hata mutabakat şartını yanlışlıkla da sağlayabiliyor. Algoritmik bir çözümü yok — daha yüksek çözünürlük veya kameraya yakınlık gerekir.
- **Yabancı plaka formatlaması**: `[YABANCI]` etiketi doğru ayrımı yapıyor ama metni ülke-özel gruplama kurallarına (örn. Alman şehir kodu + harf + rakam) oturtmuyor — ham OCR parçalarını gösteriyor.
- **Tek kare fotoğraflarda tekil karakter hatası**: bazı zor fotoğraflarda doğru cevabın parçaları farklı varyantlara dağılmış olabiliyor ama farklı yapıdaki okumaları güvenle birleştirecek bir mekanizma yok. Video/çoklu-kare senaryosunda bu sorun pratikte çözülüyor (birden fazla bağımsız kare şansı var).
- **4K işleme hızı**: VP9 4K decode maliyeti yüzünden 1080p'ye göre ~4 kat yavaş (bu kare çözme hızı, YOLO/OCR hızı değil). Canlı kamera akışında (RTSP/webcam, zaten çözülmüş kare gelir) bu sorun yok.

---

# BÖLÜM L — GELECEK İÇİN DÜŞÜNÜLEBİLECEK YÖNLER

Bunlar uygulanmadı ama proje ilerledikçe değerlendirilebilir:

- **Modelin yeniden eğitilmesi**: `detect_plate_classical()` yedek dedektörü var olduğu sürece işe yarıyor ama kalıcı çözüm modelin daha çeşitli veriyle (farklı açı, ışık, ülke) yeniden eğitilmesi olurdu.
- **OCR'ı thread'e taşımak**: canlı izlemedeki duraksamayı ortadan kaldırır (Bölüm I.7).
- **İki satırlı plaka desteği**: y-koordinatına göre satır kümeleme eklemek (Bölüm K).
- **Plaka-özel bir tanıma modeli eğitmek** (CRNN/PARSeq gibi): genel amaçlı EasyOCR yerine, biriken kendi plaka kırpımı verisiyle fine-tune edilmiş bir model, sistematik optik hataları (Bölüm K) ve dashcam'deki uzak/bulanık plakaları önemli ölçüde iyileştirebilir.
- **Ensemble OCR** (EasyOCR + PaddleOCR birlikte): iki bağımsız motorun oylaması, tek motorun sistematik zayıflıklarını (Bölüm E.1'deki tablo) telafi edebilir.

---

# BÖLÜM M — PRATİK ALTYAPI NOTU: YouTube'dan belirli bir zaman aralığını 4K indirmek

Proje sırasında karşılaşılan ve çözülen bir altyapı sorunu, ileride tekrar gerekirse diye:

`yt-dlp --download-sections "*20:00-24:00"` gibi bir komutla videonun **ortasından** bir segment istendiğinde, ffmpeg CDN'e (googlevideo) uzaktan bir "seek" (aralık atlama) isteği gönderir. Bu istek 4K/VP9 formatlarında **HTTP 400** ile reddedilebiliyor (YouTube'un bot koruması + PO token gereksinimleri + istemci tipine göre değişen URL yapısı yüzünden). Denenen dört farklı istemci (`android_vr`, `tv`, `ios`, `android`) hepsi farklı sebeplerle başarısız oldu (seek reddi, DRM, PO token, SABR).

**Çözüm**: videonun **başından** istenen zamana kadar sıralı indirmek (`--download-sections "*0-1500"` gibi — bu seek gerektirmez, ffmpeg baştan okur) ve ardından istenen aralığı **yerel** dosyadan `ffmpeg -ss 1200 -t 240 -c copy` ile anında/kayıpsız kesmek. Yerel dosyada seek network'e bağlı olmadığı için sorunsuz çalışır.

---

# BÖLÜM N — SAYILARLA ÖZET (kanıtlanmış sonuçlar)

| Aşama | Sonuç |
|---|---|
| Başlangıç (tek varyant, tek geçişli tespit) | 6 fotoğraftan 3'ü doğru (%50) |
| Çok geçişli tespit + 5 varyant + oylama | 6/6 (5 kesin doğru + 1 doğru işaretlenmiş şüpheli) |
| Genişletilmiş test seti (17 görüntü, Avrupa+gölgeli+süs plakalar dahil) | 12 güvenilir + 3 şüpheli (doğru işaretli) + 2 doğru YABANCI etiketi |
| Video — ilk versiyon (10dk 1080p dashcam) | 3 plaka, hepsi doğru |
| Video — 4K + çapraz doğrulama (4dk trafik videosu) | 24-25 plaka, ~%92 doğruluk (2 bilinen hata: iki satır karışması + sistematik optik hata) |
| Video — artımlı okuma mimarisi sonrası | Aynı doğruluk, + "kadrajdan hiç çıkmayan araç" kör noktası kapatıldı |

---

# BÖLÜM O — ADIM ADIM YÜRÜTME İZİ (ne çalıştırınca gerçekte ne oluyor)

Yukarıdaki bölümler "neden böyle tasarlandı"yı anlatıyor. Burada tam tersi: **bir komut çalıştırdığında satır satır, fonksiyon fonksiyon gerçekte ne oluyor**, hiç atlamadan.

## O.1. Fotoğraf hattı — `python colab_local.py` çalıştırınca

```
1.  initialize_model() → best.pt diskten yüklenir, YOLO nesnesi hazırlanır
2.  initialize_ocr() → EasyOCR Reader başlatılır (GPU varsa GPU'da)
3.  process_all_images(klasör, model, reader) çağrılır
    │
    └─ Klasördeki her görüntü için process_single_image(yol, model, reader):
       │
       ├─ 4.  Görüntü diskten okunur (cv2.imread)
       │
       ├─ 5.  detect_plate_boxes(model, frame, device) → TESPİT AŞAMASI
       │      │
       │      ├─ 5a. GEÇİŞ 1: model.predict(imgsz=1280, conf=0.25)
       │      │       → YOLO görüntüyü 1280'e ölçekleyip tarar
       │      │       → dönen kutular en/boy oranı + min. boyut filtresinden geçirilir
       │      │       → geçerli kutu bulunduysa bu geçiş listeye eklenir
       │      │
       │      ├─ 5b. GEÇİŞ 2: model.predict(imgsz=1920, conf=0.10) → aynı filtre
       │      ├─ 5c. GEÇİŞ 3: model.predict(imgsz=640,  conf=0.03) → aynı filtre
       │      │
       │      ├─ 5d. Hiçbir geçiş kutu bulamadıysa:
       │      │       detect_plate_classical(frame) çağrılır
       │      │       → blackhat + Sobel + kontur analiziyle "klasik" aday üretilir
       │      │
       │      └─ 5e. TÜM geçişlerden toplanan kutular birleştirilir,
       │             IoU > 0.5 olanlar tekilleştirilir (aynı fiziksel plakanın
       │             farklı geçişlerdeki kopyaları birleşir),
       │             güvene göre sıralanır → en fazla 5 aday kutu kalır
       │
       ├─ 6.  Her aday kutu için process_plate_candidate(frame, box, reader, ...):
       │      │
       │      ├─ 6a. Kutu paylı (padded_crop) ve paysız (tight_crop) kırpılır
       │      │
       │      ├─ 6b. rectify_plate(crop) → OTSU ile plaka gövdesi maskelenir,
       │      │       minAreaRect ile eğim açısı ölçülür, açı 1°-30° arasındaysa
       │      │       cv2.warpAffine ile döndürülüp düzeltilir (deskew)
       │      │
       │      ├─ 6c. detect_blue_band(crop) → sol %25'te HSV mavi taranır,
       │      │       bulunursa cv2.inpaint (TELEA) ile beyaza boyanır
       │      │
       │      ├─ 6d. detect_and_remove_orange_sticker(crop) → turuncu piksel
       │      │       oranı ≥%0.5 ise inpaint ile temizlenir, değilse dokunulmaz
       │      │
       │      ├─ 6e. detect_and_remove_black_sticker(crop) → en/boy oranı +
       │      │       konum + dairesellik filtresiyle sticker'lar bulunup
       │      │       inpaint ile temizlenir
       │      │
       │      ├─ 6f. remove_plate_frame_and_holder(crop) → Canny + approxPolyDP
       │      │       ile çerçeve bulunup içi kırpılır, alt kısımdaki reklam
       │      │       yazıları silinir
       │      │
       │      ├─ 6g. BEŞ VARYANT ÜRETİLİR:
       │      │       • enhance_plate_for_ocr(temizlenmiş_crop) → "temiz"
       │      │       • enhance_plate_for_ocr(rectified_ham_crop) → "ham"
       │      │       • cv2.resize(tight_crop, 4x, bicubic) → "sade"
       │      │       • gri + büyüt + cv2.dilate(iki kez) → "ince"
       │      │       • morfolojik kapama ile arka plan kestir + böl + büyüt
       │      │         + dilate → "gölge"
       │      │
       │      ├─ 6h. read_plate_ocr([5 varyant], reader) çağrılır:
       │      │       │
       │      │       ├─ Her varyant için reader.readtext() birkaç farklı
       │      │       │   stratejiyle çalıştırılır (renkli hâliyle, OTSU ile
       │      │       │   ikilileştirilmiş hâliyle, +yüksek-güven-filtreli,
       │      │       │   +tüm-parça-birleşimli) → ~10 ham OCR sonucu
       │      │       │
       │      │       ├─ merge_ocr_fragments() → her sonuçtaki metin
       │      │       │   parçaları (bounding box'larına göre) birleştirilir
       │      │       │
       │      │       ├─ Her birleşik metin format_turkish_plate_ex() ile
       │      │       │   Türk formatına onarılmaya çalışılır (harf↔rakam
       │      │       │   tablosu), geçerli/geçersiz + onarım maliyeti
       │      │       │   hesaplanır
       │      │       │
       │      │       ├─ _character_vote() → geçerli okumalar arasında
       │      │       │   karakter bazlı oylama yapılıp ek bir "oylama"
       │      │       │   adayı üretilir
       │      │       │
       │      │       ├─ Her aday skorlanır: geçerlilik bonusu + varyant
       │      │       │   AİLESİ konsensüs bonusu + A/H çözümleme bonusu +
       │      │       │   altdizi bonusu → en yüksek skorlu aday seçilir
       │      │       │   (tam formül ve her bonusun sayısal değeri için
       │      │       │   → Bölüm Q.5)
       │      │       │
       │      │       ├─ YABANCI KONTROLÜ: sonuç geçerli Türk plakasıysa,
       │      │       │   tüm 5 varyantta tam alfabeyle (W/Q/X dahil) tekrar
       │      │       │   okunur; ≥2 varyantta W/Q/X görülürse sonuç
       │      │       │   '[YABANCI]' etiketiyle geçersiz kılınır
       │      │       │
       │      │       └─ TEK-KAYNAK GÜVENCESİ: kazanan tek aileden geliyor
       │      │           ve güven <0.60 ise, güven zorla 0.30'a düşürülür
       │      │           (CSV kapısından geçemez hale gelir)
       │      │
       │      └─ 6i. Sonuç (metin, güven, formatlanmış, geçerli mi, skor)
       │             process_single_image'a döner
       │
       ├─ 7.  Geçerli olan TÜM aday sonuçlar (erken çıkış yok) final listeye eklenir
       │
       └─ 8.  Görüntünün sonuç listesi process_all_images'a döner
    │
    ├─ 9.  Tüm görüntülerin sonuçları tek bir listede toplanır
    ├─ 10. log_results_to_csv() → geçerli VE güven≥%35 olanlar CSV'ye yazılır
    └─ 11. print_summary_table() → konsola ✓/?/✗ özet tablosu basılır
```

## O.2. Video hattı — `python video_plaka.py video.mp4` çalıştırınca

```
1.  process_video(video_path) başlar
2.  initialize_model() + initialize_ocr() → aynı model/OCR fotoğraf hattıyla PAYLAŞILIR
3.  cv2.VideoCapture(video_path) ile video açılır; fps, kare sayısı, genişlik okunur
4.  pick_imgsz(genişlik) → 4K ise 1920, değilse 1280 seçilir
5.  show=True ise cv2.namedWindow ile izleme penceresi açılır
6.  ANA DÖNGÜ başlar:
    │
    ├─ 7.  cap.grab() → sıradaki kare kaynaktan "çekilir" ama ÇÖZÜLMEZ (ucuz işlem)
    │
    ├─ 8.  frame_no % FRAME_SKIP != 0 ise → döngü başa döner (bu kare atlandı)
    │
    ├─ 9.  FRAME_SKIP'e denk gelen karede: cap.retrieve() → kare gerçekten
    │       çözülüp piksel verisi (frame) elde edilir
    │
    ├─ 10. detect_boxes_fast(model, frame, device, imgsz) → YOLO TEK geçişte
    │       çalışır (video'da 3 geçiş yerine 1 — izleme zaten zamansal
    │       fazlalık sağladığı için tek geçiş yeterli), geometri filtresinden
    │       geçen kutular döner
    │
    ├─ 11. HER TESPİT KUTUSU İÇİN:
    │      │
    │      ├─ 11a. Mevcut aktif izlerle IoU hesaplanır
    │      │
    │      ├─ 11b. IoU > 0.30 olan bir iz varsa: track.update() → izin son
    │      │        konumu ve son görülme zamanı güncellenir
    │      │
    │      ├─ 11c. Eşleşen iz yoksa: PlateTrack(box, ...) ile YENİ bir iz
    │      │        açılır, aktif izler listesine eklenir
    │      │
    │      ├─ 11d. track.add_crop(frame, box, conf, video_sec) çağrılır:
    │      │        kutu çevresinde paylı bir alt-kare kesilir, kalite
    │      │        (genişlik × Laplace netliği) hesaplanır, CROP_GAP=0.4sn
    │      │        içinde zaten bir kırpım varsa yalnızca daha kalitelisi
    │      │        tutulur (zamansal çeşitlilik), en iyi 8 kırpım saklanır
    │      │
    │      └─ 11e. [ARTIMLI KONTROL] iz HENÜZ "çözülmemiş"se VE yeterli
    │              yeni kırpım biriktiyse (ilk denemede ≥3, sonrasında
    │              ≥2 YENİ kırpım):
    │              │
    │              ├─ attempt_read(track, reader, ...) çağrılır:
    │              │   │
    │              │   ├─ En iyi 5 kırpımın HER BİRİ için
    │              │   │   process_plate_candidate() çağrılır — BU,
    │              │   │   FOTOĞRAF HATTINDAKİ AYNI FONKSİYON (yukarıdaki
    │              │   │   6a-6i adımlarının TAMAMI her kırpım için tekrar
    │              │   │   çalışır: deskew, temizlik, 5 varyant, OCR, skorlama,
    │              │   │   yabancı kontrolü, tek-kaynak güvencesi)
    │              │   │
    │              │   ├─ 5 kırpımın okumaları _character_vote() ile
    │              │   │   birleştirilir, format_turkish_plate_ex() ile
    │              │   │   yeniden formatlanır
    │              │   │
    │              │   ├─ KARELER ARASI KONSENSÜS: ≥3 kırpım birebir aynı
    │              │   │   okuduysa güven tabanı %60'a çekilir
    │              │   │
    │              │   └─ ÇAPRAZ DOĞRULAMA: ≥2 kırpım mutabık VEYA tek
    │              │       okuma güveni ≥0.80 ise "doğrulandı" sayılır
    │              │
    │              └─ Çapraz doğrulandıysa: log_result() çağrılır
    │                  │
    │                  ├─ Yakın-eşleşme (dedup) kontrolü: aynı/benzer plaka
    │                  │   son 120 saniyede zaten loglandıysa → atlanır
    │                  │
    │                  └─ Değilse: CSV'ye satır yazılır, en iyi kırpım
    │                      cikti_video/ klasörüne kanıt olarak kaydedilir
    │
    │              → track.resolved = True işaretlenir (CSV'ye yazılsın ya da
    │                tekrar olarak elensin fark etmez, cevap artık bilinir —
    │                bu iz bir daha OCR'a sokulmaz)
    │
    ├─ 12. SÜRESİ DOLAN İZLER kontrol edilir (son görülmeden bu yana geçen
    │       örnek sayısı > MISS_LIMIT=15, yani ~1.5 saniye):
    │       │
    │       └─ HENÜZ çözülmemiş olanlara SON ÇARE denemesi yapılır
    │           (11e'deki aynı attempt_read + log_result akışı bir kez daha,
    │           mevcut kırpımlarla)
    │
    ├─ 13. Pencere açıksa: _draw_overlay() ile kutular (yeşil=açık,
    │       camgöbeği=çözüldü) ve alt bilgi şeridi çizilir, cv2.imshow
    │       ile gösterilir; Q/ESC'e basıldıysa döngüden çıkılır
    │
    └─ 14. Video bitene/kesilene kadar 7-13 arası tekrarlanır
15. Video bitince: hâlâ açık ve çözülmemiş izler için son bir deneme daha yapılır
16. cap.release(), csv_file.close(), pencere kapatılır
17. Özet konsola yazdırılır (kaç kare, kaç örnek, kaç plaka, ne kadar sürdü)
```

**Dikkat edilmesi gereken en önemli nokta**: video hattındaki 11e adımı (artımlı OCR denemesi), fotoğraf hattındaki **tüm 6a-6i akışını** her denemede yeniden çalıştırıyor. Yani bir video izlerken arka planda aslında sürekli sürekli "mini fotoğraf işleme" çağrıları yapılıyor — video kendi başına bir OCR mantığına sahip değil, sadece **ne zaman ve hangi kırpımla** bu fotoğraf mantığını tetikleyeceğine karar veriyor.

---

# BÖLÜM P — BU PROJEDE NELERLE ÇALIŞTIK, NELER ÖĞRENDİK

## P.1. Kullanılan teknolojiler ve kütüphaneler

| Araç | Bu projede ne için kullanıldı |
|---|---|
| **Ultralytics YOLO** | Plaka tespiti (nesne tespiti modeli); `YOLO("best.pt")`, `model.predict(source, conf, imgsz, device, verbose)` |
| **EasyOCR** | Plaka üzerindeki karakterleri metne çevirmek; `easyocr.Reader(['en'], gpu=...)`, `reader.readtext(img, allowlist=, paragraph=)` |
| **OpenCV (cv2)** | Görüntü işlemenin neredeyse tamamı — aşağıda P.2'de fonksiyon fonksiyon listelendi |
| **NumPy** | Piksel dizileri üzerinde matematiksel işlemler (normalize etme, mutlak değer, min/max ölçekleme) |
| **PyTorch (torch)** | `torch.cuda.is_available()` ile GPU varlığını kontrol edip YOLO/EasyOCR'ı GPU'ya yönlendirmek |
| **yt-dlp** | YouTube videolarını (belirli format/çözünürlükte, gerekirse belirli zaman aralığında) indirmek |
| **ffmpeg** | Video segmentlerini kesmek (`-ss`, `-t`, `-c copy`), format dönüştürmek |
| **csv / datetime (Python standart kütüphane)** | Sonuçları `plaka_log.csv` / `plaka_video_log.csv` dosyalarına Excel-uyumlu (`utf-8-sig`) biçimde yazmak |
| **PowerShell** | Komutları çalıştırmak, arka plan işlemlerini yönetmek, dosya/süreç durumunu kontrol etmek |

## P.2. Öğrenilen/uygulanan OpenCV görüntü işleme teknikleri

Bu proje aslında küçük bir "klasik görüntü işleme" turu oldu. Kullanılan teknikler ve OpenCV karşılıkları:

- **Renk uzayı dönüşümü**: `cv2.cvtColor` ile BGR→HSV (mavi bant/turuncu sticker tespiti için — Hue kanalı ışık şiddetinden bağımsız olduğu için renk tabanlı tespitte RGB'den daha güvenilir) ve BGR→GRAY (OCR öncesi gri tonlama).
- **Morfolojik işlemler**: `cv2.morphologyEx` ile `MORPH_BLACKHAT` (parlak zemin üzerindeki koyu detayları — harfleri — öne çıkarmak), `MORPH_CLOSE` (küçük boşlukları kapatıp bitişik bloklar oluşturmak; hem klasik tespitte hem gölge/aydınlatma düzleştirmede kullanıldı), `cv2.dilate`/`cv2.erode` (karakter kalınlaştırma/incelteme — A/H çözümlemesinin temeli).
- **Kenar/gradyan tespiti**: `cv2.Sobel` (x-yönlü gradyan — harflerin yoğun dikey kenar deseninden plaka bölgesini bulmak), `cv2.Canny` (çerçeve/dörtgen kenarlarını bulmak).
- **Eşikleme**: `cv2.threshold` ile `THRESH_OTSU` (görüntüyü otomatik olarak en uygun eşik değerinde siyah/beyaza ayırmak — plaka gövdesini maskelemek ve klasik tespitte parlaklık haritası çıkarmak için).
- **Kontur analizi**: `cv2.findContours`, `cv2.boundingRect` (bir bölgenin sınırlayıcı dikdörtgenini bulmak), `cv2.approxPolyDP` (bir konturu basit bir poligona indirgemek — plaka çerçevesini dörtgen olarak yakalamak), `cv2.minAreaRect` (bir konturu saran EN KÜÇÜK döndürülmüş dikdörtgeni bulmak — eğim açısını ölçmek için).
- **Geometrik dönüşüm**: `cv2.warpAffine` (döndürme/deskew), `cv2.resize` (büyütme — `INTER_CUBIC` bicubic enterpolasyonla, küçültme — `INTER_AREA` ile, her ikisi de farklı enterpolasyon kalitesi/hız dengesi sunuyor).
- **Görüntü onarımı**: `cv2.inpaint` (`INPAINT_TELEA` algoritmasıyla maskelenmiş bölgeyi çevresinden "yeniden inşa etmek" — kırpma yerine boyama).
- **Kontrast iyileştirme**: `cv2.createCLAHE` (yerel/adaptif histogram eşitleme — global eşitlemenin aksine görüntüyü karolara bölüp her birini kendi bağlamında iyileştirir).
- **Netlik ölçümü**: `cv2.Laplacian` ile varyans hesaplamak (bir görüntünün ne kadar bulanık/keskin olduğunu sayısallaştırmak — video hattında "en iyi kırpımı" seçmek için kullanıldı).
- **Aritmetik görüntü işlemleri**: `cv2.divide` (bir görüntüyü kestirilen arka plan aydınlatma haritasına bölmek — gölge düzleştirme), `cv2.normalize` (piksel değerlerini 0-255 aralığına yeniden ölçeklemek).
- **Video G/Ç**: `cv2.VideoCapture`, `.grab()` (kareyi kaynaktan al ama ÇÖZME — ucuz), `.retrieve()` (asıl piksel verisini çöz — pahalı), `.get(cv2.CAP_PROP_*)` (fps, çözünürlük, kare sayısı okuma).
- **Çizim/görselleştirme**: `cv2.rectangle`, `cv2.putText`, `cv2.imshow`, `cv2.namedWindow`, `cv2.waitKey` (canlı izleme penceresi için).

## P.3. Öğrenilen nesne tespiti / makine öğrenmesi kavramları

- **Güven eşiği (confidence threshold)**: bir tespitin "gerçek" sayılması için gereken minimum olasılık skoru; düşürmek recall'u artırır ama yanlış pozitif riskini de artırır.
- **`imgsz` ve mutlak/göreli piksel boyutu ayrımı**: bir modelin giriş çözünürlüğünü değiştirmenin, nesnelerin **oranını değil mutlak piksel boyutunu** değiştirdiği ve modelin eğitim sırasında öğrendiği "tipik nesne boyutu" aralığının dışına çıkan nesneleri (çok büyük ya da çok küçük) tanımakta zorlandığı (dağılım dışı / out-of-distribution problemi).
- **IoU (Intersection over Union — kesişim/birleşim oranı)**: iki sınırlayıcı kutunun ne kadar örtüştüğünü ölçen standart metrik; hem tekrarlayan tespitleri birleştirmek (tekilleştirme) hem video izlerinde ardışık kareler arası eşleştirme için kullanıldı.
- **Çok geçişli çıkarım (multi-pass inference)**: aynı modeli aynı görüntü üzerinde farklı parametrelerle (çözünürlük, eşik) birden çok kez çalıştırıp sonuçları birleştirmek — tek bir "en iyi" parametre setinin her durumda yeterli olmadığı durumlarda kullanılan bir teknik.
- **Ensemble / çoğunluk oylaması (majority voting)**: birden fazla bağımsız tahmin kaynağının ortak paydasını almak; bunun **istatistiksel bağımsızlık** gerektirdiği ve korelasyonlu (aynı kök nedenden etkilenen) kaynakların yanlışlıkla güven şişirebileceği.
- **Klasik (öğrenmesiz) bilgisayarlı görü ile derin öğrenmenin yedekli (redundant) kullanımı**: bir derin öğrenme modelinin başarısız olduğu durumlar için klasik, kural-tabanlı bir yöntemi (blackhat+Sobel dedektörü) "son çare" olarak devreye sokmak.

## P.4. Öğrenilen video işleme kavramları

- **Kare örnekleme (frame sampling) ve `grab`/`retrieve` ayrımı**: bir video dosyasından kareleri okumanın iki aşamalı olabildiği (getir vs çöz) ve gereksiz çözme işleminden kaçınmanın büyük performans kazancı sağladığı.
- **Nesne takibi (object tracking) — basit IoU eşleştirme**: karmaşık takip algoritmalarına (Kalman filtresi, optical flow, DeepSORT/ByteTrack) her zaman gerek olmadığı, örnekleme sıklığı yeterince yüksekse basit kutu-örtüşme eşleştirmesinin de işe yaradığı.
- **Video codec/container farkları**: `.mp4` (H.264/AVC1, AV1), `.webm` (VP9) gibi formatların OpenCV/ffmpeg ile okunabilirliği, 4K çözünürlükte VP9 çözmenin H.264'e göre çok daha yavaş olabildiği.
- **DASH akış (streaming) ve CDN'in uzaktan seek davranışı**: video-only/audio-only ayrı akışlar (YouTube'un DASH formatı), bir CDN URL'sinde ortadan bir zaman noktasına atlamanın (`ffmpeg -ss` uzak dosyada) her zaman desteklenmeyebileceği, bunun yerine baştan sıralı okuyup yerelde kesmenin daha güvenilir olduğu.
- **Artımlı (incremental) vs toplu (batch) işleme**: bir sonucu "tüm veriler toplanana kadar" beklemek yerine, yeterli kanıt biriktikçe erken karar vermenin (gerçek zamanlı sistemlerde) neden tercih edildiği.

## P.5. Öğrenilen yazılım mimarisi ilkeleri

- **Tek kaynak ilkesi (single source of truth)**: video hattının fotoğraf hattının OCR mantığını **tekrar yazmak yerine doğrudan import edip çağırması** — bir yerde yapılan iyileştirmenin otomatik olarak her yere yayılması.
- **Kalite kapıları (quality gates)**: bir sonucu kaydetmeden önce açık, ölçülebilir eşikler (güven ≥%35, yapısal geçerlilik, çapraz doğrulama) tanımlamak; "belirsizse hiç yazma" ilkesinin kod düzeyinde nasıl uygulandığı.
- **Sorumlulukların ayrılması (separation of concerns)**: `attempt_read()` (salt okuma denemesi) ile `log_result()` (kayıt+yan etkiler) fonksiyonlarının bilinçli olarak ayrılması — bu sayede aynı okuma mantığı hem artımlı kontrolde hem son-çare denemesinde kod tekrarı olmadan kullanılabildi.
- **Durum bayrakları ile gereksiz tekrar hesaplamayı önlemek**: `track.resolved` bayrağının doğru yerde (çapraz doğrulama anında, CSV yazımından bağımsız) işaretlenmesinin, aynı işi onlarca kez tekrar çalıştırmaktan nasıl kaçındığı.
- **Erken çıkış (early exit) optimizasyonlarının gizli riski**: performans için eklenen kısayolların (ilk bulunan tespiti kabul etmek, izi bitirmeden okumamak) nadir ama gerçek senaryolarda tamamen yanlış sonuca yol açabildiği.

## P.6. Öğrenilen altyapı/pratik beceriler

- **yt-dlp ile format seçimi ve segment indirme**: `-f <format_id>`, `--download-sections`, `--force-keyframes-at-cuts`, `--ffmpeg-location`, `--extractor-args "youtube:player_client=..."` gibi bayrakların ne işe yaradığı; YouTube'un bot koruması/PO token/DRM kısıtlarının pratikte nasıl karşılaşıldığı ve etrafından dolaşıldığı.
- **ffmpeg ile hassas video kesme**: `-ss` (başlangıç), `-t` (süre), `-c copy` (yeniden kodlamadan, kayıpsız ve anında kesim) parametrelerinin anlamı.
- **Windows'ta paket/derleme çakışmaları**: `opencv-python` vs `opencv-python-headless` vs `opencv-contrib-python` paketlerinin aynı ortamda çakışıp GUI penceresinin (`cv2.imshow`) neden sessizce çalışmayabildiği, doğru paketi seçip temiz kurulumla nasıl çözüldüğü.
- **Arka planda uzun süren işlemleri yönetmek**: uzun bir video işleme/indirme komutunu arka planda başlatıp ilerlemesini periyodik kontrol etmek, yarıda kesilen bir işlemi (indirme koptuğunda) nasıl güvenle yeniden başlatılacağını değerlendirmek (resume destekleyip desteklemediğini anlamak).

## P.7. Genel dersler

**1. "Neresi bozuk" sorusunu doğru katmanda sormak zaman kazandırıyor.** Başta OCR'ı iyileştirmeye çalışıyorduk ama asıl sorun tespit katmanındaydı (model sabit 640px'te küçük plakaları hiç göremiyordu). Eğer bu ayrımı (tespit vs okuma) yapmadan direkt "OCR'ı güçlendirelim" deseydik, haftalarca yanlış yerde debelenebilirdik. Bir sistemi katmanlara ayırmanın en büyük faydası, hatayı izole edip **doğru katmanda** düzeltebilmek.

**2. Sezgi bazen istatistiği yanıltıyor — "3 kaynak aynı şeyi söylüyor" her zaman "güvenilir" demek değil.** Varyant ailesi sorunu (Bölüm E.3) bunu çok net gösterdi: 3 farklı görüntü varyantı aynı yanlış cevaba varabiliyordu çünkü hepsi **aynı kök nedenden** (aynı bulanıklık) etkileniyordu. Bağımsızlık varsayımını kontrol etmeden "çoğunluk kazanır" mantığını uygulamak, sistematik hataları yanlışlıkla güçlendiriyor. Bu, makine öğrenmesi ensemble yöntemlerinde de bilinen ama kolayca gözden kaçan bir tuzak.

**3. Kullanıcının "bu tuhaf" demesi, loglardan çok daha değerli bir hata ayıklama sinyali olabiliyor.** Artımlı okuma mimarisine geçiş kararı, herhangi bir log satırından değil, senin "tam önümdeki araba neden hiç okunmuyor" gözleminden çıktı. Sistem "çalışıyor" görünüyordu (hata vermiyordu, CSV doluyordu) ama bir kullanım senaryosunda tamamen kör bir noktası vardı — bunu ancak gerçek kullanımı izleyen bir insan fark edebilirdi.

**4. Parametrelerin (imgsz, conf gibi) etkisi göründüğünden çok daha incelikli olabiliyor.** "Daha büyük imgsz = daha iyi tespit" gibi basit bir sezgiyle başladık ama gerçek deney bunun tam tersini gösterebiliyordu (yakın çekim plakalar büyük imgsz'de kayboluyor, çünkü mutlak piksel boyutu modelin eğitim dağılımının dışına çıkıyor). Bir parametreyi "mantıklı geliyor" diye ayarlamak yerine, gerçek veriyle ölçüp doğrulamak şart.

**5. Dürüst belirsizlik, yanlış kesinlikten iyidir.** Yabancı plaka sorunu ("kendinden emin ama tamamen sahte" bir Türk plakası kaydı üretmek) projedeki en tehlikeli hata sınıfıydı — çünkü sistem hata vermiyordu, sadece **yanlış** ve **kendinden emin** görünüyordu. Bunun tersi (bir şeyi "şüpheli" ya da "okunamadı" olarak işaretlemek) çok daha az zararlı. "Sessizce yanlış olma" ihtimali olan her yerde, sistemin bilinçli olarak "emin değilim" diyebilmesi kritik bir tasarım hedefi olmalı.

**6. Altyapı sorunları (network, codec, CDN kısıtları) bazen algoritma sorunlarından daha çok zaman alabiliyor.** 4K video segmentini indirmek, tüm OCR/tespit mimarisini kurmaktan neredeyse daha uzun sürdü — çünkü sorun bizim kodumuzda değil, YouTube'un bot koruması ve ffmpeg'in uzak seek davranışındaydı. Bu tür sorunlarda "kök nedeni izole et" (uzaktan seek mi sorun, yoksa format mı) yaklaşımı yine işe yaradı: sorunu ortadan kaldırmak yerine, sorunu **gerektirmeyen** bir yol bulmak (baştan indirip yerelde kesmek) çoğu zaman savaşmaktan daha hızlı.

**7. Erken çıkış (early exit) optimizasyonları sessizce yanlışlık üretebiliyor.** Hem tespit katmanında ("ilk bulunanı kabul et") hem video izleme mimarisinde ("izi bitir sonra oku") aynı desen tekrarlandı: performans için eklenen bir kısayol, nadir ama gerçek senaryolarda tamamen yanlış/eksik sonuca yol açıyordu. "Daha hızlı çalışıyor" ile "her zaman doğru çalışıyor" arasındaki gerilimi her erken-çıkış kararında ayrı ayrı sorgulamak gerekiyor.

---

# BÖLÜM Q — ULTRA DETAYLI TEKNİK REFERANS: HER FONKSİYON, HER PARAMETRE, HER FORMÜL

Bu bölüm, önceki bölümlerde "özetle anlatılan" her mekanizmayı **hiçbir soyutlama bırakmadan**, kodun kendisiyle birebir örtüşecek şekilde açıklıyor. Hiçbir yerde sadece "X tespiti yapıldı" denip geçilmiyor — X'in **piksel seviyesinde tam olarak nasıl** yapıldığı anlatılıyor. Bilgisayarlı görü veya OCR hakkında hiçbir ön bilgisi olmayan biri bile bunu okuyunca her adımı zihninde canlandırabilmeli.

## Q.1. OTSU eşiklemesi — ne olduğu, nasıl çalıştığı, neden seçildiği

Bu proje boyunca defalarca (`rectify_plate`, `detect_and_remove_black_sticker`, `read_plate_ocr`'daki OTSU stratejisi, `detect_plate_classical`) kullanılan **OTSU eşiklemesi**'ni tam olarak anlamak önemli çünkü kod tabanında en sık tekrarlanan tekniklerden biri.

### Problem: gri tonlamalı bir görüntüyü siyah/beyaza (ikili/binary) nasıl ayırırsın?

Bir gri tonlamalı görüntüde her piksel 0 (tam siyah) ile 255 (tam beyaz) arasında bir değer taşır. "Eşikleme" (thresholding), bir eşik değeri seçip bu değerin altındaki pikselleri siyah, üstündeki pikselleri beyaz yapmaktır. Sorun: **bu eşik değerini nasıl seçeceğiz?**

- **Sabit/manuel eşik** (örn. her zaman 127 kullan): basit ama kırılgan. Parlak bir günde çekilen plaka ile gölgede çekilen plakanın piksel dağılımları tamamen farklıdır — sabit bir sayı birinde mükemmel çalışırken diğerinde tüm görüntüyü ya tamamen siyaha ya tamamen beyaza boyayabilir.
- **OTSU yöntemi** (Nobuyuki Otsu, 1979) — SEÇİLEN: eşik değerini görüntünün **kendi piksel dağılımına bakarak otomatik** hesaplar.

### OTSU algoritması adım adım

1. Görüntünün **histogramı** çıkarılır — yani her parlaklık değerinden (0-255) kaç piksel olduğu sayılır.
2. Algoritma, olası her eşik değerini (0'dan 255'e) tek tek dener. Her aday eşik için, pikseller iki sınıfa ayrılır: eşiğin altındakiler (örn. "arka plan") ve üstündekiler (örn. "ön plan/karakterler").
3. Her aday eşik için **sınıflar arası varyans** (between-class variance) hesaplanır — bu, iki sınıfın ortalama parlaklıklarının birbirinden ne kadar uzak olduğunu ve her sınıfın kendi içinde ne kadar tutarlı (düşük varyans) olduğunu ölçen istatistiksel bir değer.
4. Sınıflar arası varyansı **maksimize eden** eşik değeri seçilir — yani "arka planı ön plandan en net ayıran" nokta.

**Sezgisel anlamı**: Bir plaka görüntüsünde tipik olarak iki "yığın" piksel vardır — parlak beyaz zemin ve koyu siyah karakterler. Histogramda bu iki yığın genelde iki ayrı tepe (bimodal dağılım) olarak görünür. OTSU, bu iki tepenin arasındaki "vadiyi" bulup eşik olarak seçer — yani zemin ile karakteri en temiz ayıran noktayı otomatik keşfeder.

### Neden OTSU, manuel/sabit eşik değil

| Yaklaşım | Sorun/Avantaj |
|---|---|
| **Sabit eşik (örn. 127)** | Aydınlatma değiştiğinde (gölge, parlak gün ışığı) tamamen yanlış sonuç verir — proje boyunca farklı ışık koşullarındaki onlarca farklı fotoğrafla çalışıldığı için bu seçenek pratik değildi |
| **Manuel/deneysel ayarlanmış eşik** (her görüntü için elle ayarlamak) | Otomatik bir pipeline'da uygulanamaz — insan müdahalesi gerektirir |
| **Adaptif eşikleme** (`cv2.adaptiveThreshold`, aşağıda Q.2'de anlatılıyor) | Yerel (bölgesel) eşikleme yapar, OTSU ise global (tüm görüntü için tek eşik) — ikisi farklı problemler için uygun, kodda ikisi de yerinde kullanıldı |
| **OTSU** — SEÇİLEN (global eşikleme gereken yerlerde) | Otomatik, parametre ayarı gerektirmez, "iki net yığınlı" (bimodal) görüntülerde (plaka gövdesi vs karakterler, ya da parlak zemin vs koyu detaylar) çok güvenilir |

### Kod tabanında OTSU'nun kullanıldığı 4 farklı yer ve her birinde NEDEN

1. **`rectify_plate()`** — `cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`: plaka gövdesini (parlak/beyaz) arka plandan ayırıp en büyük konturu (plakanın kendisi) bulmak için. Burada `THRESH_BINARY` bayrağıyla birlikte kullanılıyor — OTSU sadece **eşik değerini** otomatik hesaplıyor, `THRESH_BINARY` ise bu eşiğe göre klasik (eşik-altı-siyah, eşik-üstü-beyaz) ikili dönüşümü uyguluyor.
2. **`detect_and_remove_black_sticker()`** — `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)`: burada `THRESH_BINARY_INV` (ters ikili) kullanılıyor çünkü hedef **koyu** nesneleri (siyah sticker'lar VE karakterler) beyaz olarak işaretlemek — normal `THRESH_BINARY`'nin tam tersi mantık.
3. **`read_plate_ocr()`'daki "otsu" stratejisi** — her görüntü varyantı hem "renkli" hem "otsu" (ikilileştirilmiş) haliyle OCR'a veriliyor. Bazı EasyOCR modelleri/durumlarda ikilileştirilmiş net siyah-beyaz görüntü, renkli/gri tonlamalı görüntüden daha iyi sonuç verebiliyor (gürültü ve renk varyasyonu tamamen elenmiş oluyor).
4. **`detect_plate_classical()`** — parlaklık maskesi oluşturmak için (`cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)`) — plakanın "parlak zeminli" olma özelliğinden yararlanarak aday bölgeleri filtrelemek için.

**Önemli teknik detay**: `cv2.threshold` çağrısında `0` olarak verilen ilk eşik parametresi **görmezden gelinir** — çünkü `THRESH_OTSU` bayrağı aktifken fonksiyon eşiği kendisi hesaplayıp kullanıyor, elle verilen `0` sadece bir yer tutucu.

## Q.2. Adaptif eşikleme (`cv2.adaptiveThreshold`) — OTSU'dan farkı ve nerede kullanıldığı

OTSU **tek bir global eşik** hesaplar — tüm görüntü için aynı sayı geçerlidir. Ama bir görüntünün bir kısmı gölgede bir kısmı aydınlıksa, hiçbir tek sayı her iki bölge için de doğru olamaz. **Adaptif eşikleme** bu sorunu çözer: her piksel için, **o pikselin yerel komşuluğundaki** (örneğin 15×15 piksellik bir blok) ortalama/ağırlıklı parlaklığa bakıp kendi yerel eşiğini hesaplar.

Kod tabanında kullanıldığı yerler:
- **`adaptive_binarize()`**: `ADAPTIVE_THRESH_GAUSSIAN_C`, blok boyutu 15, sabit çıkarma değeri (C) 4. Gaussian ağırlıklı ortalama, komşuluktaki merkeze yakın piksellere daha fazla ağırlık verir (kenardaki piksellere göre). *Not: bu fonksiyon `read_plate_ocr()`'da artık kullanılmıyor — deneyler bunun tutarlı biçimde çöp ürettiğini gösterdi (`'3448'`, `'B8BY5'` gibi anlamsız sonuçlar), muhtemelen plaka üzerindeki ince detayları/gürültüyü de "kenar" olarak yorumlayıp OCR'ı yanılttığı için. Kodda hâlâ duruyor ama aktif OCR stratejilerinde çağrılmıyor — bu, "denendi ama işe yaramadığı için terk edildi" kategorisine iyi bir örnek.*
- **`remove_plate_frame_and_holder()`**: plakanın alt %25'lik bölgesindeki (galeri/reklam yazısı) küçük yazıları bulmak için, blok boyutu 11, C=2.

## Q.3. Mavi TR bandı tespiti — HSV renk uzayının tam mekaniği

### Neden RGB/BGR değil HSV

Bir görüntüdeki "bu piksel mavi mi" sorusunu cevaplamanın en doğal yolu RGB (ya da OpenCV'nin varsayılanı BGR) kanallarına bakmak gibi görünür — ama bu yanıltıcıdır. RGB'de bir rengin "mavilik derecesi" ile "parlaklığı" **iç içe geçmiştir**: karanlık bir mavi (örn. RGB≈20,20,80) ile aydınlık bir mavi (örn. RGB≈100,100,255) sayısal olarak çok farklı görünür, oysa ikisi de "mavi". Bu yüzden RGB'de "mavi aralığı" tanımlamak, farklı aydınlatma koşullarında güvenilmezdir.

**HSV (Hue-Saturation-Value)** bu iki özelliği ayırır:
- **Hue (Renk Tonu)**: rengin "kimliği" — kırmızı mı, mavi mi, yeşil mi (0-179 arası bir değer olarak temsil edilir OpenCV'de, gerçek dünyada 0-360°'nin yarısı)
- **Saturation (Doygunluk)**: rengin ne kadar "canlı/saf" olduğu (0=gri/soluk, 255=en canlı hâli)
- **Value (Değer/Parlaklık)**: rengin ne kadar aydınlık/karanlık olduğu

Bu ayrım sayesinde "mavi" aramak yalnızca **Hue** kanalına odaklanarak yapılabilir — Value (aydınlatma) ne olursa olsun aynı Hue aralığı geçerli kalır.

### Kod tam olarak ne yapıyor (`detect_blue_band`)

```python
left_region = plate_crop[:, :left_w]  # sadece sol %25
hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 80, 50])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

1. **Neden sadece sol %25'e bakılıyor**: Türk plakasında mavi TR bandının konumu sabittir (her zaman sol kenarda). Tüm görüntüde mavi aramak, arka plandaki mavi bir araç gövdesi, gökyüzü yansıması, mavi bir tabela gibi unsurları da yanlışlıkla "bant" sayabilirdi (yanlış pozitif riski). Aramayı coğrafi olarak sınırlamak, arama uzayını daraltıp yanlış pozitifleri baştan eliyor.
2. **`cv2.inRange(hsv, lower_blue, upper_blue)`**: her piksel için, üç kanalın (H,S,V) **hepsinin aynı anda** belirtilen aralıkta olup olmadığını kontrol eder. Aralıktaysa o pikseli 255 (beyaz) yapar, değilse 0 (siyah) — sonuç, "mavi olan yerler beyaz, diğerleri siyah" ikili bir maske.
3. **Hue 100-130**: OpenCV'de Hue 0-179 arası temsil edilir (gerçek 0-360°'nin yarısı, 8-bit'e sığdırmak için). Mavi renk gerçek dünyada yaklaşık 210-260° arasındadır, bu da OpenCV ölçeğinde ~105-130'a denk gelir — yani bu aralık gerçekten "mavi" rengin bulunduğu bölge.
4. **Saturation ≥80**: düşük doygunluklu (gri/soluk/beyaza yakın) pikselleri dışlar — plaka zemininin kendisi (beyaz) ya da hafif gri gölgeler yanlışlıkla "mavi" sayılmasın diye.
5. **Value ≥50**: çok karanlık (neredeyse siyah) pikselleri dışlar — gölgedeki koyu bir bölge, düşük ışıkta "koyu mavi" gibi yanlış algılanabilir, bu alt sınır bunu önlüyor.
6. **`cv2.countNonZero(blue_mask)`**: maskede beyaz (mavi bulunan) piksel sayısını sayar; toplam piksele oranı (`blue_ratio`) hesaplanır. Bu oran **%8'i** (`BLUE_BAND_THRESHOLD = 0.08`) aşarsa "bant var" kararı verilir.

### Neden kırpma değil boyama (inpaint) — mekanizma detayı

Bant tespit edildiğinde, maske **tüm plaka boyutuna** genişletilir (`full_mask[:, :left_w] = blue_mask`), sonra `cv2.dilate` ile 1 iterasyon **3×3 kernelle genişletilir** — bu genişletmenin amacı, mavi-beyaz geçiş bölgesindeki "yarı mavi" (mavi ile beyazın karıştığı ara ton) pikselleri de maskeye dahil etmek, aksi halde bu geçiş pikselleri temizlenmeden kalıp hafif bir mavi hale olarak görünmeye devam edebilirdi.

`cv2.inpaint(result, full_mask, 5, cv2.INPAINT_TELEA)` çağrısı:
- **`5`**: inpaint yarıçapı — her doldurulacak piksel için, o pikselin **5 piksel yarıçapındaki** komşularına bakarak bir değer hesaplanır.
- **`INPAINT_TELEA`**: Alexandru Telea'nın **Fast Marching Method (Hızlı Yürüyüş Yöntemi)** algoritması. Bu algoritma, maskelenmiş bölgenin **kenarından başlayarak içe doğru** ilerler — sanki bir "dalga" maskenin sınırından merkeze doğru yayılıyormuş gibi, her adımda zaten doldurulmuş komşu piksellerin ağırlıklı ortalamasını alarak yeni pikselleri doldurur. Sonuç: maskelenmiş bölge, çevresindeki dokudan "doğal" bir şekilde yeniden inşa edilmiş olur (bu durumda: düz beyaz zemin).

**Alternatif algoritma**: `INPAINT_NS` (Navier-Stokes tabanlı, akışkanlar dinamiğinden esinlenen bir yöntem) — daha karmaşık dokular/desenler için TELEA'dan daha iyi sonuç verebilir ama daha yavaştır. Plaka zemini düz, tek renkli bir yüzey olduğu için bu karmaşıklığa gerek yok — TELEA hem yeterli hem daha hızlı.

## Q.4. A ↔ H karışıklığı — TAM MEKANİZMA (önce şişirme YOK, doğrudan tek adımlı inceltme)

Bu, üzerinde en çok yanlış anlaşılma olabilecek kısım olduğu için baştan netleştirelim: **süreç "önce karakterleri şişirip sonra inceltmek" DEĞİLDİR.** Kodda tek bir işlem var: gri tonlamalı görüntüye doğrudan `cv2.dilate` uygulanıyor. Bunu anlamak için önce dilasyonun gri tonlamalı (sadece siyah/beyaz değil, ara tonları da olan) bir görüntüde ne yaptığını tam olarak açıklamak gerekiyor.

### Dilasyon (genişletme) tam olarak ne yapar

`cv2.dilate`, görüntüdeki her pikseli, o pikselin çevresindeki (kernel/yapısal eleman boyutunca) komşularının **maksimum** değeriyle değiştiren bir işlemdir. İkili (siyah/beyaz, 0/255) bir görüntüde bu, "beyaz alanları genişletmek, siyah alanları küçültmek" anlamına gelir — sezgisel olarak bilinen hâli budur.

**Ama burada işlem gri tonlamalı bir görüntü üzerinde yapılıyor** (`gray_crop`, ikili değil!). Gri tonlamalı bir görüntüde dilasyon şu anlama gelir: her piksel, komşuluğundaki **en parlak** (en yüksek sayısal değerli) pikselin değerini alır. Sonuç olarak:
- Görüntüdeki **parlak (yüksek değerli) bölgeler büyür** — komşu koyu pikselleri "yutarak" genişlerler.
- Görüntüdeki **koyu (düşük değerli) bölgeler küçülür** — çünkü koyu bir pikselin komşuluğunda daha parlak bir piksel varsa, o piksel artık komşusunun parlak değerini alır.

### Bunun plaka karakterlerine etkisi

Bir plaka görüntüsünde zemin **parlak** (beyaz), karakterler **koyu** (siyah)'dır. Dilasyon uygulandığında:
- Parlak zemin genişler
- Koyu karakter çizgileri **incelir** (çünkü zemin onları her yönden "yiyor")

**Bu yüzden tek bir dilasyon adımı, doğrudan "karakter inceltme" etkisi yaratıyor — önce kalınlaştırıp sonra inceltmek gibi iki aşamalı bir süreç YOK.** Kod tam olarak şöyle çalışıyor:

```python
gray_crop = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY)
thin_scale = min(max(270.0 / max(gray_crop.shape[0], 1), 1.0), 8.0)
gray_up = cv2.resize(gray_crop, None, fx=thin_scale, fy=thin_scale, interpolation=cv2.INTER_CUBIC)
thinned = cv2.dilate(gray_up, np.ones((3, 3), np.uint8), iterations=2)
```

Adımlar: (1) gri tonlama, (2) hedef yüksekliğe (~270px) göre bicubic büyütme, (3) **doğrudan** `cv2.dilate` ile 2 iterasyon (yani işlem art arda 2 kez tekrarlanır — her tekrarda inceltme etkisi biraz daha güçlenir, 3×3'lük kernel ile). Erozyon (kalınlaştırma) hiçbir yerde **önce** uygulanmıyor.

### Neden A ve H spesifik olarak bu sorunu yaşıyor

A harfinin şekli: iki eğik çizgi + ortalarında yatay bir çizgi + **üstte kapalı bir üçgen boşluk**. H harfinin şekli: iki dikey çizgi + ortalarında yatay bir köprü — **kapalı boşluk yok**, açık bir "H" şekli.

Düşük çözünürlükte/bulanıklıkta, A'nın normalde **beyaz (zemin rengi)** olması gereken üst üçgen boşluğu, bulanıklık yüzünden komşu siyah çizgilerin "bulaşmasıyla" griye/koyuya döner ve dolar. Bu doluysa, görsel olarak A artık H'ye çok benzer (iki dikey/eğik çizgi + yatay köprü, boşluksuz).

Dilasyon (parlaklığı genişletme) uygulandığında, eğer o üçgen bölgede **hâlâ biraz** parlaklık kalıntısı varsa (tamamen simsiyah olmamışsa), dilasyon bu kalıntı parlaklığı büyütüp üçgeni yeniden "açığa çıkarır" — A'nın gerçek şekli geri ortaya çıkar.

### Neden bu mekanizma TEK YÖNLÜDÜR (kritik nokta)

Eğer harf gerçekten H ise (yani üstte hiç kapalı bir boşluk **hiç yoksa**, çünkü H'nin yapısında zaten böyle bir boşluk bulunmuyor), dilasyon bu H'yi A'ya çeviremez — çünkü ortada "açığa çıkarılacak" gizli bir üçgen zaten yok, dilasyon var olmayan bir şeyi yaratamaz, sadece var olan ama bulanıklıkla gizlenmiş bir şeyi ortaya çıkarabilir.

Bu yüzden mantık şudur: **eğer "ince" (dilasyon uygulanmış) varyant yüksek güvenle A okuyorsa VE kalın varyantlar (dilasyonsuz) aynı pozisyonda H okuyorsa, A'nın doğru olma ihtimali çok yüksektir** — çünkü bu senaryo (H'nin yanlışlıkla A görünmesi) fiziksel olarak mümkün değildir, ama A'nın bulanıklıkla H görünmesi mümkündür. Kod bu asimetriyi ödüllendiriyor:

```python
if p['valid'] and p['strategy'].startswith('ince') and p['conf'] >= 0.80:
    ...
    if diffs and all(a == 'A' and b == 'H' for a, b in diffs):
        score += 0.35
```

**Neden erozyon değil dilasyon seçildi**: Erozyon (komşuluktaki minimum değeri almak) tam ters etki yaratırdı — koyu bölgeleri genişletip parlak bölgeleri küçültürdü, yani karakterleri daha da **kalınlaştırıp** yapışmayı şiddetlendirirdi. Hedef "ayrıştırmak" olduğu için yön olarak dilasyon doğru seçimdi.

**Neden 2 iterasyon**: Deneyler tek iterasyonun bazı durumlarda yetersiz kaldığını, iki iterasyonun (aynı 3×3 kerneli iki kez art arda uygulamak, etkin olarak biraz daha büyük bir komşuluğa yayılmak) daha güvenilir ayrışma sağladığını gösterdi.

## Q.5. Skorlama formülünün TAM matematiği

`read_plate_ocr()` içinde her aday için hesaplanan nihai skor, şu bileşenlerin **toplamıdır**:

```
skor = güven
     + geçerlilik_bonusu
     + konsensüs_bonusu
     + A_H_çözümleme_bonusu   (koşullu, +0.35)
     + altdizi_bonusu          (koşullu, +0.25)
```

### Bileşen 1: `güven` (conf)
EasyOCR'ın kendi ürettiği ham güven skoru (0.0-1.0), `merge_ocr_fragments()` içinde birden fazla parçanın **uzunluk-ağırlıklı ortalaması** olarak hesaplanır: `conf = Σ(parça_güveni × parça_uzunluğu) / toplam_uzunluk`. *Neden uzunluk ağırlıklı*: eşit ağırlıklı basit ortalama, 1 karakterlik rastgele bir "gürültü" parçasına, 6 karakterlik gerçek plaka metniyle eşit söz hakkı veriyordu — bu, kısa ama yüksek güvenli sahte bir parçanın toplam güveni yanlışlıkla yükseltmesine yol açabiliyordu.

### Bileşen 2: `geçerlilik_bonusu`
```python
if p['valid']:
    score += max(0.3, 1.0 - 0.15 * p['cost'])
elif re.match(r'^\d{2} [A-Z]{1,3} \d{2,4}$', p['formatted']):
    score += 0.2
```
Eğer aday yapısal olarak **tam geçerli** bir Türk plakasıysa: `1.0 - 0.15 × onarım_maliyeti` bonusu verilir, ama en az `0.3` garanti edilir (taban). Yani **hiç onarım gerekmemiş** (maliyet=0) bir okuma tam `1.0` bonus alırken, 3 birimlik onarımla zorla geçerli yapılmış bir okuma yalnızca `1.0 - 0.45 = 0.55` bonus alır — "az düzeltmeyle doğal olarak geçerli olan" okuma sistematik olarak öne çıkarılıyor. Eğer aday yapısal olarak tam geçerli değilse ama **şekil olarak** plakaya benziyorsa (regex ile "2 rakam boşluk 1-3 harf boşluk 2-4 rakam" deseni), küçük bir teselli bonusu (`+0.2`) veriliyor — tamamen rastgele bir metinle karşılaştırıldığında yine de bir öncelik hakkı tanınıyor.

### Bileşen 3: `konsensüs_bonusu`
```python
agreement = len(format_families.get(p['formatted'], set())) - 1
score += min(0.30, 0.15 * max(0, agreement))
```
`format_families` sözlüğü, her benzersiz formatlanmış metnin **kaç farklı varyant ailesinden** (kalın veya ince/gölge) destek aldığını tutar (bir `set()` içinde — aynı aileden birden fazla strateji aynı sonucu verse bile set yalnızca bir kez sayar, bu da "aynı ailenin tekrarları bağımsız kanıt sayılmasın" ilkesinin matematiksel karşılığıdır). `agreement = aile_sayısı - 1` (yani yalnızca 1 aile varsa agreement=0, bonus yok; 2 aile varsa agreement=1, bonus=0.15; teorik olarak daha fazla aile olsaydı bonus artacaktı ama tavan `0.30`'da sınırlı).

### Bileşen 4: A/H çözümleme bonusu (Q.4'te detaylı anlatıldı)
Koşullu `+0.35` — yalnızca "ince" ailesinden gelen, güveni ≥0.80 olan, ve "kalın" bir adayla tam olarak yalnızca A↔H farkı taşıyan bir eşleşme bulunduğunda uygulanır.

### Bileşen 5: Altdizi (subsequence) bonusu
```python
if (len(other) < len(my_clean)
        and _is_subsequence(other, my_clean)
        and my_fams >= other_fams):
    score += 0.25
```
Eğer bu aday (`my_clean`), başka geçerli bir adayın (`other`, daha kısa) **tüm karakterlerini sırayla** içeriyorsa (`_is_subsequence` — `other`'daki her karakterin `my_clean` içinde aynı sırayla bulunup bulunmadığını kontrol eder, aralarda başka karakter olması sorun değil), bu, "kısa okumada bir karakter düşmüş, uzun okuma onu telafi ediyor" durumunu işaret eder ve `+0.25` bonus verilir. **Ek koşul** (`my_fams >= other_fams`): uzun adayın aile desteği kısa adayınkinden az olamaz — bu, gölge varyantının ürettiği sahte fazladan bir karakterin (aile desteği zayıf) doğru kısa okumayı (aile desteği güçlü) yenmesini önlemek için sonradan eklendi.

### Neden bu beş bileşen toplanıyor, çarpılmıyor veya başka bir formül kullanılmıyor

Basit toplama (additive scoring) tercih edildi çünkü her bileşen **bağımsız bir kanıt türünü** temsil ediyor ve yorumlanabilir kalması isteniyor (hangi bileşenin skoru ne kadar etkilediği kolayca izlenebiliyor, log çıktısında görülebiliyor). Çarpımsal bir formül (örn. güven × geçerlilik × konsensüs) bir bileşen sıfıra yakınsa tüm skoru sıfırlayabilir — bu, kısmi kanıtın da bir değeri olması gerektiği durumlarda çok sert bir ceza olurdu.

## Q.6. `merge_ocr_fragments` — çok hipotezli parça birleştirme mekaniği

EasyOCR bir plakayı genelde tek bir metin olarak değil, **birden fazla ayrı "parça"** (fragment) olarak okur — örneğin `"06"`, `"ABY"`, `"325"` gibi üç ayrı tespit. Her parçanın kendi konumu (bounding box), metni ve güveni vardır. Sorun: bu parçaların **hangisinin gerçek plaka metnine ait, hangisinin sahte** (çerçeve kenarı, vida, sticker kalıntısı) olduğu belli değildir.

### Neden "en olası tek yorumu bul" değil "birden fazla hipotez üret"

Fonksiyon **tek bir "doğru" birleşimi tahmin etmeye çalışmak yerine**, üç farklı filtreleme varsayımıyla üç farklı hipotez üretiyor ve hepsini aday havuzuna gönderiyor — hangisinin doğru olduğuna nihai skorlama sistemi (Q.5) karar veriyor:

1. **"yükseklik filtreli" (ana hipotez)**: yalnızca en yüksek parçanın yüksekliğinin **≥%45'i kadar yüksek** olan parçalar birleştirilir (`max_h * 0.45` eşiği). *Mantık*: plaka karakterleri hep aynı yüksekliktedir; bir parça bariz şekilde daha kısaysa (örn. çerçevenin kenarından yanlışlıkla okunan küçük bir '1'), bu muhtemelen gerçek karakter değil bir artefakttır.
2. **"yükseklik + güven filtreli"**: yukarıdaki filtreye ek olarak, güveni **<0.30** olan parçalar da atılır. *Gerçek örnek*: gevşek bir kırpımda plaka yanındaki galeri sticker'ından gelen `"EN"` parçası (güven 0.16) yükseklik filtresini geçebiliyordu (yeterince "uzun" göründüğü için) ama güven filtresiyle elenip gerçek plaka metni doğru ayrışıyordu.
3. **"tümü" (filtresiz)**: hiçbir filtre uygulanmadan tüm parçalar birleştirilir. *Neden gerekli*: yükseklik filtresi bazen **yanlışlıkla gerçek bir parçayı** da atabilir (örn. gerçek bir karakter grubu farklı bir netlik/perspektifte biraz daha kısa ölçülmüş olabilir) — filtresiz hipotez bu durumda telafi sağlıyor.

Her hipotez, `_join_fragments()` ile soldan sağa (x-koordinatına göre) sıralanıp birleştiriliyor ve uzunluk-ağırlıklı güven hesaplanıyor (Q.5, Bileşen 1'de anlatıldığı gibi). En az 4 karakter uzunluğundaki ve daha önce üretilmemiş (tekrarsız) hipotezler döndürülüyor.

## Q.7. `_character_vote` — karakter seviyesinde çoğunluk oylaması mekaniği

Bu fonksiyon, `read_plate_ocr` içinde toplanan **tüm geçerli** (yapısal olarak Türk plaka formatına uyan) adayları alıp, aralarında ek bir "sentezlenmiş" aday üretir.

### Neden yapı imzasına göre gruplama gerekiyor

Farklı stratejiler bazen tamamen farklı **uzunlukta** ya da harf/rakam **düzeninde** sonuçlar üretebiliyor (biri "34 N 5953" — 1 harf 4 rakam, diğeri "34 AH 738" — 2 harf 3 rakam gibi tamamen farklı plakalar okumuş olabilir, bunlar aynı fiziksel plakanın farklı okumaları değildir). Bunları karıştırıp pozisyon pozisyon oylamak anlamsız olurdu (1. pozisyondaki bir harfi, tamamen farklı yapıdaki bir adayın 1. pozisyonundaki rakamla oylamak gibi). Bu yüzden önce **"yapı imzası"** (`uzunluk, her_pozisyonun_harf_mi_rakam_mı_olduğu`) hesaplanıp, yalnızca **aynı imzaya sahip** adaylar aynı grupta toplanıyor.

### Oylama mekaniği

En kalabalık grup (eşitlik durumunda toplam güveni en yüksek olan) seçiliyor. Bu grubun içindeki her aday için, **her karakter pozisyonu ayrı ayrı** ele alınıyor: o pozisyondaki her farklı karakter için, o karakteri öneren tüm adayların güvenleri **toplanıyor** (`votes[karakter] += güven`), ve en yüksek toplam güveni alan karakter o pozisyon için kazanan seçiliyor.

**Somut örnek** (docstring'den): üç strateji aynı plakayı okuyor:
- Strateji A: `"06HBY325"` (güven 0.81) — 3. pozisyonda yanlışlıkla 'H'
- Strateji B: `"06ABY325"` (güven 0.51) — doğru
- Strateji C: `"06ABY325"` (güven 0.50) — doğru

Tek tek bakıldığında en yüksek güvenli aday A'dır (0.81) — ama **yanlış**. Karakter oylamasında 3. pozisyon için: 'H' toplam güveni 0.81, 'A' toplam güveni 0.51+0.50=1.01. 'A' kazanır çünkü **iki bağımsız strateji** onu destekliyor, tek bir yüksek-güvenli ama yalnız kalan strateji yerine. Sentezlenen metin: `"06ABY325"` — doğru sonuç, "en yüksek tek güven" mantığının vermeyeceği sonuç.

Sentezin nihai güveni, grubun içindeki **en güvenilir tekil üyenin güveni** olarak atanıyor (`max_conf = max(c for _, c in best_group)`) — mantık: "oylanmış bir sentez, en azından grubun en iyi üyesi kadar güvenilirdir" varsayımı.

## Q.8. Format onarımı (`format_turkish_plate_ex`) — tam onarım arama mekaniği

Bu fonksiyon, ham OCR metnini alıp Türk plaka formatına oturtmaya çalışırken, aslında **kapsamlı bir arama** yapıyor — tek bir düzeltme denemesi değil.

### Onarım arama uzayı

```python
for lead in range(0, 3):      # baştan 0, 1 veya 2 karakter kırp
    for trail in range(0, 3):  # sondan 0, 1 veya 2 karakter kırp
        candidate = text[lead:n-trail]
        fixed = _positional_fix(candidate)
        for variant in (candidate, fixed):  # kırpılmış ham hâli VE pozisyonel-düzeltilmiş hâli
            if is_valid_turkish_plate(variant):
                valid_options.append((maliyet, dönüşüm_sayısı, variant))
```

Bu, `3 × 3 × 2 = 18` farklı kombinasyonu (bazıları tekrarlanıp elense de) deniyor — her biri için yapısal geçerlilik kontrol ediliyor, geçerli olanlar arasından **en düşük maliyetli** (en az kırpma + en az karakter dönüşümü) seçiliyor.

### Neden kırpma (lead/trail) deneniyor

Plaka çerçevesinin kenarı, bir vida başı, ya da hafif bir gölge kalıntısı bazen OCR tarafından yanlışlıkla fazladan bir karakter (genelde '1' ya da 'I' gibi ince şekiller) olarak okunabiliyor. Örnek (docstring'den): `"06ABY3251"` — sonunda fazladan bir '1' var, bu OCR gerçek plakanın hemen yanındaki bir çerçeve kenarını karakter sanmış. Son karakteri kırpınca (`trail=1`) `"06ABY325"` kalıyor — bu yapısal olarak geçerli. Kırpma denemeleri bu tür kenar artefaktlarını otomatik tespit edip düzeltiyor.

### `_positional_fix` — pozisyona duyarlı karakter dönüşümü

Bu fonksiyon, plakanın **hangi bölgesinde hangi tip karakterin (harf mi rakam mı) beklendiğini** bilerek düzeltme yapıyor:
1. **İl kodu bölgesi** (ilk 2 karakter): her zaman rakam olmalı — eğer bir harf görünüyorsa (`LETTER_TO_DIGIT` tablosundan, örn. 'O'→'0'), dönüştürülür.
2. **Numara bölgesi** (sondan geriye doğru, en fazla 4 karakter): rakam olması beklenir — sondan başlayarak rakam sayıp, karşılaşılan harfleri (`LETTER_TO_DIGIT` ile) rakama çevirir. **Güvenlik kuralı**: eğer sondan zaten 2 veya daha fazla gerçek rakam bulunduysa, bir sonraki harfi artık rakama çevirmeyi durdurur — çünkü bu noktada karşılaşılan harf muhtemelen gerçekten bir harftir (harf serisinin son harfi), rakam sanıp yanlışlıkla dönüştürmek gerçek bir harfi bozar (örnekte verilen: `"34ABT738"`'deki gerçek T harfinin yanlışlıkla 7'ye çevrilmesini önlemek).
3. **Orta bölge** (harf serisi bölgesi): rakam görünüyorsa (`DIGIT_TO_LETTER` tablosundan, örn. '0'→'O') harfe çevrilir.

### Harf↔Rakam dönüşüm tablosunun kaynağı

Bu eşleştirmeler rastgele değil, **görsel benzerlik** temelinde seçilmiş: `O`/`Q`↔`0` (yuvarlak şekiller), `I`/`L`↔`1` (dikey tek çizgi), `Z`↔`2` (benzer eğik hat), `S`↔`5` (benzer kıvrım), `G`↔`6` (benzer yuvarlak+kanca), `T`↔`7` (benzer üst çizgi+dikey), `B`↔`8` (benzer çift yuvarlak). Bunlar, düşük çözünürlükte/bulanıklıkta OCR'ın gerçekten karıştırdığı, literatürde de bilinen klasik karakter çiftleridir.

### Neden "en ucuz onarım" seçiliyor, ilk bulunan değil

`valid_options.sort(key=lambda o: (o[0], o[1]))` — önce toplam maliyete (kırpma+dönüşüm sayısı), eşitlik durumunda dönüşüm sayısına göre sıralanıyor. **Neden dönüşüm sayısı ikincil kriter**: kırpma (bir karakteri tamamen atmak) ile dönüşüm (bir karakteri başka bir karaktere çevirmek) eşit maliyetli sayılsa da, kırpma genelde daha **güvenli** bir işlemdir (kenar artefaktı atmak, yaygın ve zararsız bir düzeltme), dönüşüm ise gerçek bir karakter kanıtını değiştirmek anlamına gelir (daha riskli bir varsayım). Eşit maliyette kırpmayı tercih etmek, daha az riskli onarım yoluna öncelik veriyor. **Maksimum onarım maliyeti 3** ile sınırlı — bundan daha fazla düzeltme gerektiren bir metin muhtemelen o kadar bozuk ki, "onarılmış" hali bile güvenilir sayılmamalı.

## Q.9. "Konsensüs" ve "varyant ailesi" tam olarak ne demek — kavramsal netleştirme

Bu iki terim rapor boyunca sürekli geçiyor ama ne oldukları net açıklanmadan kullanılmış olabilir. Baştan, gündelik dilde açıklayalım.

### Konsensüs (consensus) nedir

Gündelik anlamı: **"birden fazla bağımsız kaynağın aynı sonuca varması"**. Bu projede iki farklı yerde, iki farklı ölçekte kullanılıyor:

1. **Fotoğraf hattında (tek görüntü içinde)**: bir plaka kırpımından **5 farklı görüntü varyantı** (temiz/ham/sade/ince/gölge) üretiliyor ve her biri ayrı ayrı OCR'dan geçiyor. Eğer birden fazla varyant **aynı** plaka metnine varıyorsa, bu "konsensüs var" demektir — ve sistem bu duruma bir güven bonusu veriyor (Bölüm Q.5, Bileşen 3).
2. **Video hattında (kareler arası)**: aynı fiziksel plaka, videoda birden fazla farklı **karede** görünüyor. Eğer birden fazla kare bağımsız olarak aynı sonuca varıyorsa, bu da bir konsensüs biçimidir (Bölüm I.6'daki "kareler arası konsensüs").

**Neden önemli**: Tek bir kaynağın (tek varyant, tek kare) yanlış okuma yapma ihtimali her zaman vardır — bulanıklık, gölge, açı gibi nedenlerle. Ama **birden fazla bağımsız kaynağın aynı yanlışı yapması** çok daha düşük bir ihtimaldir (özellikle kaynaklar gerçekten bağımsızsa — aşağıdaki "aile" kavramı tam olarak bunu garanti etmeye çalışıyor). Bu yüzden "kaç kaynak aynı şeyi söylüyor" bilgisi, bir okumanın ne kadar güvenilir olduğuna dair güçlü bir sinyal.

### Varyant ailesi (variant family) nedir — ve neden konsensüsün ÖNKOŞULU

Burada kritik bir incelik var: **"3 kaynak aynı şeyi söylüyor" ifadesi, ancak o 3 kaynak gerçekten birbirinden BAĞIMSIZSA anlamlıdır.** Eğer 3 kaynak aslında aynı kök nedenden (örn. aynı bulanıklık sorunu) aynı şekilde etkileniyorsa, "3 bağımsız kanıt" değil, "1 kanıtın 3 kopyası" elde edilmiş olur — ve bunu "güçlü konsensüs" sanmak yanıltıcıdır.

**Somut örnek**: `temiz`, `ham`, `sade` varyantlarının üçü de aynı **"kalın karakterli"** temel görüntüden türetiliyor — hiçbiri karakter çizgilerini incelten bir işlem (dilasyon) içermiyor. Düşük çözünürlükte bu üç varyantın **üçü de** A harfini H olarak okuyabilir, çünkü hepsi aynı kök nedenden (bulanıklıkla dolan A boşluğu) aynı şekilde etkileniyor. Eğer bu 3 varyantı "3 bağımsız kanıt" sayıp konsensüs bonusu verseydik, sistematik bir hatayı yanlışlıkla "güçlü doğrulanmış" olarak işaretlemiş olurduk.

**Çözüm — varyant ailesi kavramı**: Kod, varyantları **iki aileye** ayırıyor:
```python
def variant_family(strategy):
    return "ince" if strategy.startswith(("ince", "gölge")) else "kalın"
```
- **"kalın" ailesi**: `temiz`, `ham`, `sade` — hiçbiri karakter inceltme (dilasyon) içermiyor, hepsi "orijinal kalınlıktaki" karakterlerle çalışıyor.
- **"ince" ailesi**: `ince`, `gölge` — ikisi de dilasyon tabanlı bir işlemden geçiyor (biri doğrudan karakter inceltme için, diğeri gölge/aydınlatma düzleştirmesinin bir parçası olarak — ama ikisi de aynı "inceltilmiş karakter" fiziksel özelliğini taşıyor, bu yüzden aynı aileye konuyorlar).

**Konsensüs bonusu artık yalnızca FARKLI ailelerin aynı sonuca varmasıyla sayılıyor** (`format_families` sözlüğü her sonucun hangi ailelerden destek aldığını bir `set()` ile tutuyor — aynı aileden gelen tekrarlar sette yalnızca bir kez sayılıyor). Yani 3 "kalın" varyantı aynı yanlış cevaba varsa bu **hâlâ tek bir aile** sayılır, bonus almaz. Ama 1 "kalın" + 1 "ince" varyant aynı cevaba varırsa, bu **2 farklı aile** demektir — gerçekten bağımsız iki farklı fiziksel mekanizmanın (biri kalın karakterle, biri inceltilmiş karakterle) aynı sonuca vardığı anlamına gelir, bu çok daha güçlü bir kanıttır.

### Video hattındaki kareler arası konsensüs neden farklı bir eşiğe sahip

Video hattında (Bölüm I.6) benzer bir mantık var ama sayılar farklı: ≥3 **kare** aynı okursa güven tabanı yükseltiliyor, ama yalnızca 2 kare aynıysa taban verilmiyor. Neden fotoğraftaki "2 aile yeterli" kuralından farklı: video kareleri, farklı varyant AİLELERİ kadar yapısal olarak "farklı" değil — ardışık kareler genelde birbirine çok benziyor (aynı araç, aynı ışık, aynı açı, sadece milisaniyeler farklı zaman). Bu yüzden video'da "bağımsızlık" güvencesi daha zayıf, dolayısıyla daha fazla tekrar (3 kare) isteniyor — tek bir "farklı aile" kavramı video karelerinde doğrudan uygulanamıyor çünkü kareler arasında fotoğraftaki gibi net bir "mekanizma farkı" yok, sadece zaman farkı var.

## Q.10. Video hattında tespit ve OCR — fotoğraftan farkı ve aynı kalan kısımlar

Bölüm I ve Bölüm O'da video mimarisi genel hatlarıyla anlatıldı; burada eklenmesi gereken tek teknik detay: video hattındaki `detect_boxes_fast()` fonksiyonu, fotoğraftaki `detect_plate_boxes()`'ın **basitleştirilmiş bir versiyonu** — 3 geçiş yerine **tek geçiş** kullanıyor (`model.predict(imgsz=uyarlanmış_değer, conf=0.15)`), aynı geometri filtresini (en/boy oranı 1.5-7.5, min. genişlik/yükseklik) uyguluyor. Tek geçiş yeterli çünkü video zaten **zamansal fazlalık** sağlıyor — bir karede kaçırılan bir plaka, saniyede 10 örnekten birinde tekrar denenmiş oluyor; fotoğrafta ise tek bir "şans" olduğu için 3 geçişe ihtiyaç var.
