"""
Sistema de Detección de Rostros Ultra-Mejorado
Con detección híbrida, recuperación ante pérdidas y análisis de confianza
"""

import gc
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Resolución máxima para detección facial.
# Los frames se reducen a este ancho ANTES de pasarlos a MediaPipe/fallback,
# eliminando el pico de ~24 MB por frame 4K que causa el OOM killer.
DETECTION_MAX_WIDTH = 960


@dataclass
class TrackingQuality:
    """Métricas de calidad del tracking"""
    confidence: float
    stability: float
    age: int
    consecutive_detections: int
    lost_frames: int
    is_reliable: bool


class EnhancedFaceTracker:
    """
    Tracker mejorado con predicción avanzada y análisis de calidad
    """
    def __init__(self, face_id, initial_bbox, initial_confidence):
        self.id = face_id
        self.bbox = initial_bbox
        self.confidence = initial_confidence
        self.age = 0
        self.lost_frames = 0
        self.consecutive_detections = 1
        
        # Historial extendido para mejor predicción
        self.history = deque(maxlen=30)
        self.history.append(initial_bbox)
        
        # Velocidades y aceleraciones
        self.velocity_history = deque(maxlen=10)
        self.current_velocity = (0, 0)
        
        # Métricas de calidad
        self.quality_scores = deque(maxlen=15)
        self.quality_scores.append(initial_confidence)
        
        # Estado del tracking
        self.is_predicted = False
        self.prediction_confidence = 1.0

    def update(self, bbox, confidence, is_predicted=False):
        """Actualizar tracker con nueva detección o predicción"""
        if not is_predicted:
            # Detección real
            self.consecutive_detections += 1
            self.lost_frames = 0
            self.is_predicted = False
            self.prediction_confidence = 1.0
        else:
            # Predicción
            self.is_predicted = True
            self.prediction_confidence *= 0.85  # Decay en confianza
        
        # Calcular velocidad
        if len(self.history) > 0:
            prev_bbox = self.history[-1]
            vx = bbox[0] - prev_bbox[0]
            vy = bbox[1] - prev_bbox[1]
            self.current_velocity = (vx, vy)
            self.velocity_history.append(self.current_velocity)
        
        self.bbox = bbox
        self.confidence = confidence
        self.age += 1
        self.history.append(bbox)
        self.quality_scores.append(confidence)

    def mark_lost(self):
        """Marcar como perdido y predecir posición"""
        self.lost_frames += 1
        self.consecutive_detections = 0
        
        # Predecir siguiente posición usando velocidad
        predicted_bbox = self.predict_next_position()
        self.update(predicted_bbox, self.confidence * 0.9, is_predicted=True)

    def predict_next_position(self) -> Tuple[int, int, int, int]:
        """Predicción avanzada usando velocidad y aceleración"""
        if len(self.history) < 2:
            return self.bbox
        
        x, y, w, h = self.bbox
        
        # Usar velocidad promedio reciente
        if len(self.velocity_history) >= 2:
            recent_vx = np.median([v[0] for v in list(self.velocity_history)[-5:]])
            recent_vy = np.median([v[1] for v in list(self.velocity_history)[-5:]])
            
            # Aplicar decay a la velocidad mientras más tiempo perdido
            decay = 0.7 ** self.lost_frames
            predicted_x = int(x + recent_vx * decay)
            predicted_y = int(y + recent_vy * decay)
        else:
            predicted_x, predicted_y = x, y
        
        return (predicted_x, predicted_y, w, h)

    def get_quality_metrics(self) -> TrackingQuality:
        """Calcular métricas de calidad del tracking"""
        # Confianza promedio
        avg_confidence = np.mean(list(self.quality_scores)) if self.quality_scores else 0
        
        # Estabilidad (baja varianza = alta estabilidad)
        if len(self.history) >= 3:
            positions = np.array([h[0] for h in list(self.history)[-10:]])
            stability = 1.0 / (1.0 + np.std(positions) / 10.0)
        else:
            stability = 0.5
        
        # Confiabilidad general
        is_reliable = (
            avg_confidence > 0.7 and
            stability > 0.6 and
            self.lost_frames < 5 and
            self.consecutive_detections > 2
        )
        
        return TrackingQuality(
            confidence=avg_confidence * self.prediction_confidence,
            stability=stability,
            age=self.age,
            consecutive_detections=self.consecutive_detections,
            lost_frames=self.lost_frames,
            is_reliable=is_reliable
        )

    def is_alive(self, max_lost_frames=15):
        """Verificar si el tracker aún es válido - más tolerante"""
        return self.lost_frames < max_lost_frames


class SkinDetector:
    """
    Detector de piel como método fallback cuando MediaPipe falla
    """
    def __init__(self):
        # Rangos HSV para detección de piel (múltiples tonos)
        self.skin_ranges = [
            # Tono claro
            (np.array([0, 20, 70]), np.array([20, 150, 255])),
            # Tono medio
            (np.array([0, 25, 50]), np.array([25, 170, 255])),
            # Tono oscuro
            (np.array([0, 30, 30]), np.array([30, 200, 255])),
        ]
    
    def detect_face_regions(self, frame) -> List[Tuple[int, int, int, int]]:
        """
        Detectar regiones que probablemente contengan rostros basado en color de piel
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        
        # Combinar todas las máscaras de piel
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.skin_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            skin_mask = cv2.bitwise_or(skin_mask, mask)
        
        # Reducir ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filtrar por área mínima
            if area < 2000:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filtrar por aspect ratio (rostros son ~1:1.2)
            aspect_ratio = cw / ch if ch > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
            
            # Validar que no sea muy grande (probablemente no es solo un rostro)
            if cw > w * 0.6 or ch > h * 0.6:
                continue
            
            candidates.append((x, y, cw, ch))
        
        return candidates


class EnhancedFaceDetector:
    """
    Detector mejorado con sistema híbrido y recuperación inteligente
    """
    def __init__(self, config):
        self.config = config

        # MediaPipe Face Detection - detector principal
        # model_complexity=0 y max_num_faces=1 reducen memoria interna del modelo.
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=0,                                         # 0 = modelo ligero
            min_detection_confidence=config.FACE_DETECTION['min_confidence']
        )

        # Detector de piel - fallback
        self.skin_detector = SkinDetector()

        # Tracking
        self.trackers: List[EnhancedFaceTracker] = []
        self.next_tracker_id = 0

        # Cache de detecciones para suavizado temporal mejorado
        self.detection_cache = deque(
            maxlen=config.FACE_DETECTION['temporal_smoothing_window']
        )

        # Configuración
        self.min_face_size = config.FACE_DETECTION['min_face_size']
        self.max_faces = config.FACE_DETECTION['max_faces']
        self.priority_mode = config.FACE_DETECTION['priority_mode']
        self.redetect_interval = config.FACE_DETECTION['redetect_interval_frames']

        self.frame_count = 0
        
        # Métricas de rendimiento
        self.detection_failures = 0
        self.fallback_activations = 0

    def detect(self, frame):
        """
        Detecta rostros con sistema híbrido y recuperación inteligente
        """
        h, w, _ = frame.shape
        self.frame_count += 1

        # Re-detección periódica
        force_detect = (self.frame_count % self.redetect_interval == 0)

        # Intentar detección con MediaPipe
        raw_faces = self._detect_mediapipe(frame)
        
        # Si MediaPipe falla y hay trackers activos, usar fallback
        if not raw_faces and self.trackers and self.frame_count > 30:
            raw_faces = self._detect_with_fallback(frame)
        
        if raw_faces or force_detect:
            self._update_trackers(raw_faces, w, h)
            if raw_faces:
                self.detection_failures = 0
        else:
            self.detection_failures += 1
            self._predict_trackers()

        # Limpiar trackers muertos - más tolerante
        self.trackers = [t for t in self.trackers if t.is_alive(max_lost_frames=15)]

        # Limitar número de trackers
        if len(self.trackers) > self.max_faces:
            self.trackers = self._prioritize_trackers()[:self.max_faces]

        # Convertir trackers a formato de salida
        faces = [self._tracker_to_face(t) for t in self.trackers]

        # Aplicar suavizado temporal mejorado
        if faces:
            faces = self._apply_advanced_temporal_smoothing(faces)

        return faces

    def _detect_mediapipe(self, frame) -> List[dict]:
        """Detección principal con MediaPipe.

        El frame se reduce a DETECTION_MAX_WIDTH antes de procesarlo para
        evitar el pico de ~24 MB por frame 4K. Las bounding boxes se
        reescalan al tamaño original antes de devolverlas.
        """
        h, w, _ = frame.shape

        # --- Downscale para detección ---
        if w > DETECTION_MAX_WIDTH:
            scale = DETECTION_MAX_WIDTH / w
            det_w = DETECTION_MAX_WIDTH
            det_h = int(h * scale)
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            det_w, det_h = w, h
            small = frame  # sin copia extra

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.mp_face.process(rgb)

        # Liberar explícitamente para ayudar al GC
        del rgb
        if w > DETECTION_MAX_WIDTH:
            del small

        faces = []
        if result.detections:
            for det in result.detections:
                confidence = det.score[0]

                if confidence < self.config.FACE_DETECTION['min_confidence']:
                    continue

                bbox = det.location_data.relative_bounding_box
                # Coordenadas en espacio del frame reducido, luego escalar a original
                x = int(bbox.xmin * det_w / scale)
                y = int(bbox.ymin * det_h / scale)
                bw = int(bbox.width * det_w / scale)
                bh = int(bbox.height * det_h / scale)

                if not self._is_valid_detection(x, y, bw, bh, w, h):
                    continue

                faces.append({
                    'bbox': (x, y, bw, bh),
                    'confidence': confidence,
                    'center': (x + bw // 2, y + bh // 2),
                    'area': bw * bh,
                    'method': 'mediapipe'
                })

        return faces

    def _detect_with_fallback(self, frame) -> List[dict]:
        """
        Detección fallback usando región de piel + posición predicha de trackers.
        El frame se reduce a DETECTION_MAX_WIDTH para mantener consistencia
        con _detect_mediapipe y evitar picos de memoria.
        """
        self.fallback_activations += 1
        h, w, _ = frame.shape

        # Downscale para fallback
        if w > DETECTION_MAX_WIDTH:
            scale = DETECTION_MAX_WIDTH / w
            det_w = DETECTION_MAX_WIDTH
            det_h = int(h * scale)
            small = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small = frame

        # Obtener regiones de piel en resolución reducida
        skin_regions_small = self.skin_detector.detect_face_regions(small)

        if w > DETECTION_MAX_WIDTH:
            del small

        if not skin_regions_small:
            return []

        # Escalar regiones de piel a resolución original
        skin_regions = [
            (int(x / scale), int(y / scale), int(sw / scale), int(sh / scale))
            for x, y, sw, sh in skin_regions_small
        ]
        
        # Correlacionar con posiciones predichas de trackers
        faces = []
        for tracker in self.trackers:
            predicted_bbox = tracker.predict_next_position()
            px, py, pw, ph = predicted_bbox
            predicted_center = (px + pw // 2, py + ph // 2)
            
            # Buscar región de piel cercana
            best_match = None
            min_distance = float('inf')
            
            for skin_bbox in skin_regions:
                sx, sy, sw, sh = skin_bbox
                skin_center = (sx + sw // 2, sy + sh // 2)
                
                distance = np.sqrt(
                    (predicted_center[0] - skin_center[0])**2 +
                    (predicted_center[1] - skin_center[1])**2
                )
                
                if distance < min_distance and distance < 150:  # Max 150px de distancia
                    min_distance = distance
                    best_match = skin_bbox
            
            if best_match:
                x, y, bw, bh = best_match
                faces.append({
                    'bbox': (x, y, bw, bh),
                    'confidence': max(0.5, 1.0 - min_distance / 150),  # Confianza basada en distancia
                    'center': (x + bw // 2, y + bh // 2),
                    'area': bw * bh,
                    'method': 'fallback'
                })
        
        return faces

    def _is_valid_detection(self, x, y, w, h, frame_w, frame_h):
        """Validación mejorada de detecciones"""
        # Tamaño mínimo
        if w < self.min_face_size or h < self.min_face_size:
            return False

        # Tamaño máximo razonable
        if w > frame_w * 0.85 or h > frame_h * 0.85:
            return False

        # Aspect ratio más permisivo
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False

        # Dentro del frame con margen
        margin = 5
        if x < -margin or y < -margin or x + w > frame_w + margin or y + h > frame_h + margin:
            return False

        return True

    def _update_trackers(self, detections, frame_w, frame_h):
        """Actualizar trackers con matching mejorado"""
        if not detections:
            for tracker in self.trackers:
                tracker.mark_lost()
            return

        # Emparejar usando IoU y distancia de centros
        matched_trackers = set()
        matched_detections = set()

        # Ordenar trackers por calidad
        sorted_trackers = sorted(
            enumerate(self.trackers),
            key=lambda x: x[1].get_quality_metrics().confidence,
            reverse=True
        )

        for i, detection in enumerate(detections):
            best_match = None
            best_score = 0.2  # Threshold mínimo

            for j, tracker in sorted_trackers:
                if j in matched_trackers:
                    continue

                # Score combinado: IoU + distancia de centros
                iou = self._calculate_iou(detection['bbox'], tracker.bbox)
                
                det_center = detection['center']
                tracker_center = (tracker.bbox[0] + tracker.bbox[2]//2, 
                                tracker.bbox[1] + tracker.bbox[3]//2)
                center_dist = np.sqrt(
                    (det_center[0] - tracker_center[0])**2 +
                    (det_center[1] - tracker_center[1])**2
                )
                
                # Normalizar distancia
                max_dist = np.sqrt(frame_w**2 + frame_h**2)
                center_score = 1.0 - (center_dist / max_dist)
                
                # Score combinado
                combined_score = iou * 0.7 + center_score * 0.3
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = j

            if best_match is not None:
                self.trackers[best_match].update(
                    detection['bbox'],
                    detection['confidence'],
                    is_predicted=False
                )
                matched_trackers.add(best_match)
                matched_detections.add(i)

        # Marcar trackers no emparejados
        for j, tracker in enumerate(self.trackers):
            if j not in matched_trackers:
                tracker.mark_lost()

        # Crear nuevos trackers
        for i, detection in enumerate(detections):
            if i not in matched_detections:
                new_tracker = EnhancedFaceTracker(
                    self.next_tracker_id,
                    detection['bbox'],
                    detection['confidence']
                )
                self.trackers.append(new_tracker)
                self.next_tracker_id += 1

    def _predict_trackers(self):
        """Predecir posiciones de todos los trackers"""
        for tracker in self.trackers:
            tracker.mark_lost()

    def _calculate_iou(self, bbox1, bbox2):
        """IoU mejorado con manejo de casos edge"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _prioritize_trackers(self):
        """Priorizar usando métricas de calidad"""
        if self.priority_mode == 'quality':
            return sorted(
                self.trackers,
                key=lambda t: t.get_quality_metrics().confidence,
                reverse=True
            )
        elif self.priority_mode == 'largest':
            return sorted(
                self.trackers,
                key=lambda t: t.bbox[2] * t.bbox[3],
                reverse=True
            )
        elif self.priority_mode == 'hybrid':
            return sorted(
                self.trackers,
                key=lambda t: self._hybrid_score(t),
                reverse=True
            )
        else:
            return self.trackers

    def _hybrid_score(self, tracker):
        """Score híbrido mejorado con calidad de tracking"""
        quality = tracker.get_quality_metrics()
        x, y, w, h = tracker.bbox
        cx = x + w // 2
        cy = y + h // 2

        # Score de tamaño
        size_score = (w * h) / (1920 * 1080)

        # Score de centralidad
        dist = np.sqrt((cx - 960)**2 + (cy - 540)**2)
        max_dist = np.sqrt(960**2 + 540**2)
        centrality_score = 1 - (dist / max_dist)

        # Score de calidad de tracking
        tracking_quality = quality.confidence * (quality.stability ** 0.5)

        # Combinación ponderada
        return (
            size_score * 0.25 +
            centrality_score * 0.20 +
            tracking_quality * 0.40 +
            min(quality.age / 30, 1.0) * 0.15
        )

    def _tracker_to_face(self, tracker):
        """Convertir tracker a formato de salida con métricas"""
        x, y, w, h = tracker.bbox
        quality = tracker.get_quality_metrics()
        
        return {
            'bbox': tracker.bbox,
            'confidence': quality.confidence,
            'center': (x + w // 2, y + h // 2),
            'area': w * h,
            'tracker_id': tracker.id,
            'age': tracker.age,
            'quality': quality,
            'is_predicted': tracker.is_predicted
        }

    def _apply_advanced_temporal_smoothing(self, faces):
        """Suavizado temporal mejorado con pesos adaptativos"""
        if not faces:
            return faces

        self.detection_cache.append(faces)

        if len(self.detection_cache) < 3:
            return faces

        smoothed_faces = []
        for i, face in enumerate(faces):
            quality = face.get('quality')
            
            # Peso basado en calidad
            if quality and quality.is_reliable:
                # Alta calidad: menos suavizado (más responsive)
                window_size = 3
            else:
                # Baja calidad: más suavizado (más estable)
                window_size = min(7, len(self.detection_cache))
            
            # Recolectar bboxes históricos
            historical_bboxes = []
            weights = []
            
            for j, past_faces in enumerate(list(self.detection_cache)[-window_size:]):
                if i < len(past_faces):
                    historical_bboxes.append(past_faces[i]['bbox'])
                    # Peso mayor a frames más recientes
                    age = window_size - j
                    weights.append(1.0 / age)
            
            if len(historical_bboxes) > 1:
                # Normalizar pesos
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Promediar con pesos
                avg_x = int(np.average([b[0] for b in historical_bboxes], weights=weights))
                avg_y = int(np.average([b[1] for b in historical_bboxes], weights=weights))
                avg_w = int(np.average([b[2] for b in historical_bboxes], weights=weights))
                avg_h = int(np.average([b[3] for b in historical_bboxes], weights=weights))

                smoothed_face = face.copy()
                smoothed_face['bbox'] = (avg_x, avg_y, avg_w, avg_h)
                smoothed_face['center'] = (avg_x + avg_w // 2, avg_y + avg_h // 2)
                smoothed_faces.append(smoothed_face)
            else:
                smoothed_faces.append(face)

        return smoothed_faces

    def get_primary_face(self, faces):
        """Obtener cara principal con validación de calidad"""
        if not faces:
            return None

        # Filtrar por calidad si es posible
        reliable_faces = [f for f in faces if f.get('quality') and f['quality'].is_reliable]
        
        if reliable_faces:
            return reliable_faces[0]
        elif faces:
            return faces[0]
        else:
            return None

    def get_tracking_stats(self):
        """Obtener estadísticas de tracking"""
        return {
            'active_trackers': len(self.trackers),
            'detection_failures': self.detection_failures,
            'fallback_activations': self.fallback_activations,
            'reliable_trackers': sum(1 for t in self.trackers if t.get_quality_metrics().is_reliable)
        }

    def reset(self):
        """Resetear detector"""
        self.trackers = []
        self.detection_cache.clear()
        self.frame_count = 0
        self.detection_failures = 0
        self.fallback_activations = 0
