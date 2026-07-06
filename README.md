# 🗺️ Ataşehir Dereyolu Gerçek Zamanlı Trafik ve Emisyon Simülatörü

Akıllı şehirler, mikro iklim analizleri ve kentsel karbon ayak izi takibi için geliştirilmiş, Bilgisayarlı Görü (Computer Vision) ve Web GIS (Coğrafi Bilgi Sistemleri) teknolojilerini harmanlayan yenivelikçi bir altyapı projesi. Sistem, canlı şehir kameralarını Derin Öğrenme ile işleyerek yoğun veya dur-kalk trafikteki araçları bile kaçırmadan algılar ve Leaflet haritası üzerindeki 100 metrelik etki alanını (Buffer) anlık emisyon verileriyle simüle eder.

---

## 🚀 Canlı Demo ve Arayüz Önizlemesi

### 🖥️ Gerçek Zamanlı Akıllı Şehir Komuta Merkezi
Entegre web arayüzü sayesinde kullanıcılar; yapay zeka analiz akışını, harita üzerindeki dairesel tampon bölge analizini ve kümülatif karbon istatistiklerini tek bir ekrandan eşzamanlı olarak izleyebilir.

![Komuta Merkezi Genel Görünüm](Ekran%20görüntüsü%202026-07-06%20182458.jpg)

### 🧠 Canlı Yapay Zeka Görüntü İşleme Hattı
Geliştirilen hassas algılama konfigürasyonları sayesinde sistem; birbirini kapatan (occlusion), kırmızı ışıkta bekleyen durağan araçları ve perspektif açıdan kaynaklanan piksel küçülmelerini başarıyla çözer.

| Trafik Sıkışıklığı (Kuyruk Analizi) | Yüksek Hassasiyetli Tespit Akışı |
|---|---|
| ![Durağan Araç Tespiti](Ekran%20görüntüsü%202026-07-06%20181943.jpg) | ![YOLO Tespit Kutuları](Ekran%20görüntüsü%202026-07-06%20190535.jpg) |

---

## 🛠️ Öne Çıkan Teknik Özellikler

* **Durağanlığa Duyarlı Tespit Motoru:** Klasik takip algoritmalarının duran araçları kaçırma eğilimini esnetmek için `YOLOv8s` modeli saf algılama (`predict`) modunda çalıştırılmış ve güven eşiği ($conf=0.15$) seviyesine çekilerek kilitlenen trafikteki araçlar bile yakalanmıştır.
* **Mesafe Filtreli Koordinat Hafızası:** Araç kimlikleri (ID) olmadan toplam araç sayısının hatalı artmasını önlemek amacıyla Öklid mesafesine dayalı ($35\text{px}$ yarıçap) akıllı bir konum havuzu entegre edilmiştir. Böylece duran araçlar kümülatif emisyonu yapay olarak şişirmez.
* **Web GIS 100m Tampon Bölge (Buffer):** $40^\circ 59' 17.99''\text{ K}, 29^\circ 7' 31.80''\text{ D}$ kesin koordinatını merkez alan ve dinamik JSON veri hattından beslenen 100 metrelik CBS tampon bölgesi Leaflet.js üzerinde canlı render edilir.
* **Bütünleşik Çift Portlu Mimari:** Python tabanlı Flask sunucusu, arka planda işlenen OpenCV karelerini tarayıcıya MJPEG Stream olarak fırlatır. Bu sayede tarayıcıların Cross-Origin Resource Sharing (CORS) güvenlik engelleri tamamen aşılmıştır.

---

## 🗂️ Proje Klasör Yapısı

Sistemi çalıştırmadan önce dizininizde aşağıdaki temel üretim dosyalarının bulunduğundan emin olun:

```text
├── emisyon.py            # Yapay Zeka Motoru, Matematiksel Model ve Flask Sunucusu
├── index.html            # Web GIS Dashboard Paneli (Leaflet.js & Dark Mode)
├── live_traffic_data.json# Portlar Arası Anlık Canlı Veri Transfer Köprüsü
└── requirements.txt      # Sunucu Bağımlılık Matrisi
