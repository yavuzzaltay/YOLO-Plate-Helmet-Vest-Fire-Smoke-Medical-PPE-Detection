import os
from roboflow import Roboflow
from ultralytics import YOLO

class DatasetManager:
    """Roboflow üzerinden veri seti indirme işlemlerini yöneten sınıf."""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.rf = Roboflow(api_key=self.api_key)

    def download_dataset(self, workspace_name, project_name, version_number):
        """Belirtilen projeyi YOLOv8 formatında indirir ve YAML yolunu döner."""
        print(f"[INFO] Downloading dataset: {project_name} (Version {version_number})...")
        project = self.rf.workspace(workspace_name).project(project_name)
        dataset = project.version(version_number).download("yolov8")
        
        # Roboflow'un indirdiği klasördeki data.yaml dosyasının tam yolunu alıyoruz
        yaml_path = os.path.join(dataset.location, "data.yaml")
        print(f"[INFO] Dataset successfully downloaded to: {dataset.location}")
        return yaml_path


class PlateModelFineTuner:
    """Mevcut eğitilmiş YOLO modelinin (best.pt) üzerine fine-tuning yapan sınıf."""
    
    def __init__(self, base_model_path):
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found at: {base_model_path}")
        
        self.model_path = base_model_path
        self.model = YOLO(base_model_path)
        print(f"[INFO] Base model successfully loaded: {base_model_path}")

    def execute_fine_tuning(self, data_config_path, epochs=50, batch_size=16, learning_rate=0.001):
        """Eğitimi (Fine-tuning) başlatır."""
        print("[INFO] Starting fine-tuning process...")
        
        self.model.train(
            data=data_config_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            lr0=learning_rate,
            pretrained=True, # Eski öğrendiklerini koruması için hayati önem taşır
            optimizer="AdamW" # İnce ayar için daha stabil sonuçlar verir
        )
        print("[INFO] Fine-tuning completed successfully!")


# --- ANA ÇALIŞTIRICI (ENTRY POINT) ---
if __name__ == "__main__":
    # 1. Kendi Roboflow API anahtarını buraya yapıştır
    # (Roboflow sitesinde Settings -> Roboflow API kısmından alabilirsin)
    ROBOFLOW_API_KEY = "5ev3CyoR15ytn0okpkFV"
    
    # 2. Üzerine eğiteceğin mevcut modelinin yolu
    MY_BEST_MODEL_PATH = r"C:\Users\Yavuz Altay\Desktop\İyex\PlateDetection\best.pt"
    
    try:
        # --- VERİ SETİ İNDİRME AŞAMASI ---
        dataset_manager = DatasetManager(api_key=ROBOFLOW_API_KEY)
        
        # Seçtiğin "guler-kandeger/plate-detection-vh2rk" projesinin sürüm 2'sini indiriyoruz
        dataset_yaml_path = dataset_manager.download_dataset(
            workspace_name="guler-kandeger",
            project_name="plate-detection-vh2rk",
            version_number=2
        )
        
        # --- EĞİTİM (FINE-TUNING) AŞAMASI ---
        tuner = PlateModelFineTuner(base_model_path=MY_BEST_MODEL_PATH)
        
        # Eğitimi başlat
        tuner.execute_fine_tuning(
            data_config_path=dataset_yaml_path,
            epochs=50,            # Eğer veri seti büyükse bunu 30'a düşürebilirsin
            batch_size=16,        # Ekran kartı gücüne göre 8 veya 32 yapabilirsin
            learning_rate=0.001   # Model zaten bir şeyler bildiği için düşük tutuyoruz
        )
        
    except Exception as error:
        print(f"[ERROR] An unexpected error occurred: {error}")