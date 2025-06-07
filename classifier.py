# classifier.py
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import rasterio
from rasterio.io import MemoryFile
from rasterio.features import shapes
import json
import base64
from typing import Tuple, Dict, Optional

class EuroSATClassifier:
    """Оптимизированный классификатор для спутниковых изображений EuroSAT"""
    
    def __init__(self, model_path: str, patch_size: int = 64, batch_size: int = 64):
        self.model_path = model_path
        self.patch_size = patch_size
        self.batch_size = batch_size  # Увеличен для скорости
        self.model = None
        
        # Цвета и названия классов
        self.class_colors = {
            0: [255, 255, 0], 1: [34, 139, 34], 2: [124, 252, 0],
            3: [128, 128, 128], 4: [139, 69, 19], 5: [152, 251, 152],
            6: [255, 140, 0], 7: [255, 0, 0], 8: [0, 0, 255], 9: [0, 191, 255]
        }
        self.class_names = {
            0: 'AnnualCrop', 1: 'Forest', 2: 'HerbaceousVegetation',
            3: 'Highway', 4: 'Industrial', 5: 'Pasture',
            6: 'PermanentCrop', 7: 'Residential', 8: 'River', 9: 'SeaLake'
        }
    
    def load_model(self):
        """Загрузка и прогрев модели"""
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)
            # Прогрев модели
            self.model.predict(np.zeros((1, self.patch_size, self.patch_size, 3), dtype=np.float32), verbose=0)
    
    def check_channels(self, file_bytes: bytes) -> int:
        """Проверка количества каналов в файле"""
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    return src.count
        except:
            return 3  # Обычное RGB изображение
    
    def process_raster(self, file_bytes: bytes, channels_str: Optional[str] = None) -> Tuple[np.ndarray, dict, object, object]:
        """Обработка растра с выбором каналов"""
        try:
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    # Сохраняем метаданные
                    profile, transform, crs = src.profile.copy(), src.transform, src.crs
                    
                    # Выбор каналов
                    if channels_str and src.count > 3:
                        channels = [min(max(1, int(ch)), src.count) for ch in channels_str.split(',')[:3]]
                    else:
                        channels = [1, 2, 3] if src.count >= 3 else [1] * 3
                    
                    # Чтение и нормализация данных
                    data = src.read(channels).transpose(1, 2, 0).astype(np.float32)
                    data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
                    
                    # Нормализация каналов
                    for i in range(3):
                        channel = data[:, :, i]
                        if np.any(channel > 0):
                            min_val, max_val = np.percentile(channel[channel > 0], [2, 98])
                            data[:, :, i] = np.clip((channel - min_val) / (max_val - min_val), 0, 1) if max_val > min_val else channel / 255.0
                        else:
                            data[:, :, i] = 0
                    
                    return data, profile, transform, crs
        except:
            # Обработка как обычное изображение
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            data = np.array(img, dtype=np.float32) / 255.0
            h, w = data.shape[:2]
            
            profile = {'driver': 'GTiff', 'dtype': 'uint8', 'count': 1, 'height': h, 'width': w, 'compress': 'lzw'}
            transform = rasterio.transform.from_bounds(0, 0, w, h, w, h)
            
            return data, profile, transform, None
    
    def create_preview(self, file_bytes: bytes, channels_str: Optional[str] = None) -> bytes:
        """Создание превью с выбранными каналами"""
        data, _, _, _ = self.process_raster(file_bytes, channels_str)
        
        # Преобразование в изображение
        img = Image.fromarray((data * 255).astype(np.uint8))
        
        # Ограничение размера
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
        
        # Сохранение в байты
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        return buf.getvalue()
    
    def classify_fast(self, data: np.ndarray) -> np.ndarray:
        """Быстрая классификация без перекрытий"""
        if self.model is None:
            raise RuntimeError("Модель не загружена")
        
        h, w = data.shape[:2]
        
        # Паддинг до кратного размера
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            data_padded = np.pad(data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            data_padded = data
        
        # Новые размеры после паддинга
        h_pad, w_pad = data_padded.shape[:2]
        n_patches_h = h_pad // self.patch_size
        n_patches_w = w_pad // self.patch_size
        
        # Быстрое извлечение всех патчей через reshape
        patches = data_padded.reshape(n_patches_h, self.patch_size, n_patches_w, self.patch_size, 3)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, self.patch_size, self.patch_size, 3)
        
        # Батчевое предсказание
        n_patches = len(patches)
        predictions = np.zeros(n_patches, dtype=np.uint8)
        
        for i in range(0, n_patches, self.batch_size):
            batch = patches[i:i + self.batch_size]
            batch_pred = self.model.predict(batch, verbose=0)
            predictions[i:i + len(batch)] = np.argmax(batch_pred, axis=1)
        
        # Восстановление формы
        class_map = predictions.reshape(n_patches_h, n_patches_w)
        class_map = np.repeat(np.repeat(class_map, self.patch_size, axis=0), self.patch_size, axis=1)
        
        # Обрезка до оригинального размера
        return class_map[:h, :w]
    
    def create_visualization(self, class_map: np.ndarray) -> bytes:
        """Создание RGB визуализации"""
        h, w = class_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.class_colors.items():
            rgb[class_map == class_id] = color
        
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=True)
        buf.seek(0)
        return buf.getvalue()
    
    def create_geotiff(self, class_map: np.ndarray, profile: dict, transform: object, crs: object) -> bytes:
        """Создание GeoTIFF"""
        out_profile = profile.copy()
        out_profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw'})
        out_profile.pop('nodata', None)
        
        with MemoryFile() as memfile:
            with memfile.open(**out_profile) as dst:
                dst.write(class_map, 1)
                if transform: dst.transform = transform
                if crs: dst.crs = crs
            return memfile.read()
    
    def create_geojson(self, class_map: np.ndarray, transform: object, crs: object) -> str:
        """Создание GeoJSON с упрощением геометрий"""
        features = []
        
        if transform:
            # Упрощаем растр перед векторизацией для скорости
            from scipy.ndimage import median_filter
            class_map_smooth = median_filter(class_map, size=3)
            
            for geom, value in shapes(class_map_smooth.astype(np.int32), transform=transform):
                if value > 0:
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "class": int(value),
                            "class_name": self.class_names.get(int(value), "Unknown")
                        },
                        "geometry": geom
                    })
        
        geojson = {"type": "FeatureCollection", "features": features}
        
        if crs:
            try:
                epsg = crs.to_epsg()
                if epsg:
                    geojson["crs"] = {"type": "name", "properties": {"name": f"EPSG:{epsg}"}}
            except:
                pass
        
        return json.dumps(geojson)
    
    def classify_all(self, file_bytes: bytes, channels_str: Optional[str] = None) -> Dict[str, str]:
        """Полный пайплайн классификации"""
        # Обработка и классификация
        data, profile, transform, crs = self.process_raster(file_bytes, channels_str)
        class_map = self.classify_fast(data)  # Используем быстрый метод без перекрытий
        
        # Создание результатов
        viz_bytes = self.create_visualization(class_map)
        geotiff_bytes = self.create_geotiff(class_map, profile, transform, crs)
        geojson_str = self.create_geojson(class_map, transform, crs)
        
        # Возврат в base64
        return {
            "visualization": base64.b64encode(viz_bytes).decode('utf-8'),
            "geotiff": base64.b64encode(geotiff_bytes).decode('utf-8'),
            "geojson": geojson_str
        }
