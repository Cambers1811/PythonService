"""
Sistema de Estabilización Cinematográfica Ultra-Mejorado
Con análisis adaptativo, múltiples estrategias y recuperación inteligente
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from enum import Enum


class MovementType(Enum):
    """Tipos de movimiento detectados"""
    STATIC = "static"           # Casi sin movimiento
    SMOOTH = "smooth"           # Movimiento suave
    MODERATE = "moderate"       # Movimiento moderado
    RAPID = "rapid"             # Movimiento rápido
    ERRATIC = "erratic"         # Movimiento errático


class AdaptiveKalmanFilter:
    """
    Filtro de Kalman con parámetros adaptativos según el contexto
    """
    def __init__(self, process_variance=0.01, measurement_variance=0.1, estimation_error=1.0):
        self.base_process_variance = process_variance
        self.base_measurement_variance = measurement_variance
        self.estimation_error = estimation_error
        self.posterior_estimate = None
        
        # Parámetros adaptativos
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        # Historial para análisis
        self.innovation_history = deque(maxlen=10)
        
    def update(self, measurement, movement_type=MovementType.SMOOTH):
        """Actualizar con adaptación según tipo de movimiento"""
        
        # Ajustar parámetros según tipo de movimiento
        self._adapt_parameters(movement_type)
        
        if self.posterior_estimate is None:
            self.posterior_estimate = measurement
            return measurement
        
        # Prediction
        prior_estimate = self.posterior_estimate
        prior_error = self.estimation_error + self.process_variance
        
        # Update
        kalman_gain = prior_error / (prior_error + self.measurement_variance)
        innovation = measurement - prior_estimate
        self.innovation_history.append(abs(innovation))
        
        self.posterior_estimate = prior_estimate + kalman_gain * innovation
        self.estimation_error = (1 - kalman_gain) * prior_error
        
        return self.posterior_estimate
    
    def _adapt_parameters(self, movement_type):
        """Adaptar parámetros del filtro según contexto"""
        if movement_type == MovementType.STATIC:
            # Mucho suavizado para contenido estático
            self.process_variance = self.base_process_variance * 0.3
            self.measurement_variance = self.base_measurement_variance * 2.0
        elif movement_type == MovementType.SMOOTH:
            # Parámetros estándar
            self.process_variance = self.base_process_variance
            self.measurement_variance = self.base_measurement_variance
        elif movement_type == MovementType.MODERATE:
            # Balance entre respuesta y suavizado
            self.process_variance = self.base_process_variance * 1.5
            self.measurement_variance = self.base_measurement_variance * 0.8
        elif movement_type == MovementType.RAPID:
            # Más reactivo para movimientos rápidos
            self.process_variance = self.base_process_variance * 3.0
            self.measurement_variance = self.base_measurement_variance * 0.4
        elif movement_type == MovementType.ERRATIC:
            # Muy suavizado para movimientos erráticos
            self.process_variance = self.base_process_variance * 0.5
            self.measurement_variance = self.base_measurement_variance * 3.0


class MovementAnalyzer:
    """
    Analiza el tipo de movimiento en tiempo real
    """
    def __init__(self, window_size=15):
        self.position_history = deque(maxlen=window_size)
        self.velocity_history = deque(maxlen=window_size)
        self.acceleration_history = deque(maxlen=window_size)
        
    def analyze(self, position) -> MovementType:
        """Analizar tipo de movimiento actual"""
        self.position_history.append(position)
        
        if len(self.position_history) < 3:
            return MovementType.SMOOTH
        
        # Calcular velocidades
        velocities = []
        for i in range(1, len(self.position_history)):
            v = abs(self.position_history[i] - self.position_history[i-1])
            velocities.append(v)
        
        if velocities:
            self.velocity_history.append(velocities[-1])
        
        # Calcular aceleraciones
        if len(self.velocity_history) >= 2:
            acc = abs(self.velocity_history[-1] - self.velocity_history[-2])
            self.acceleration_history.append(acc)
        
        # Análisis estadístico
        if len(velocities) >= 5:
            recent_velocities = list(self.velocity_history)[-5:]
            avg_velocity = np.mean(recent_velocities)
            std_velocity = np.std(recent_velocities)
            max_velocity = np.max(recent_velocities)
            
            # Clasificar
            if avg_velocity < 2:
                return MovementType.STATIC
            elif avg_velocity < 8 and std_velocity < 4:
                return MovementType.SMOOTH
            elif avg_velocity < 20 and std_velocity < 10:
                return MovementType.MODERATE
            elif max_velocity > 40 and std_velocity < 15:
                return MovementType.RAPID
            else:
                return MovementType.ERRATIC
        
        return MovementType.SMOOTH
    
    def get_smoothness_score(self) -> float:
        """Calcular score de suavidad (0-1, 1 = muy suave)"""
        if len(self.velocity_history) < 3:
            return 1.0
        
        velocities = list(self.velocity_history)
        std = np.std(velocities)
        mean = np.mean(velocities)
        
        # Coefficient of variation invertido
        if mean > 0:
            cv = std / mean
            smoothness = 1.0 / (1.0 + cv)
        else:
            smoothness = 1.0
        
        return smoothness


class AdaptiveStabilizer:
    """
    Estabilizador cinematográfico con adaptación inteligente
    """
    def __init__(self, config):
        self.config = config
        self.method = config.STABILIZATION['method']
        
        # Filtros de Kalman adaptativos
        if self.method in ['kalman', 'hybrid']:
            kalman_params = config.STABILIZATION['kalman']
            self.kalman_x = AdaptiveKalmanFilter(**kalman_params)
            self.kalman_y = AdaptiveKalmanFilter(**kalman_params)
        
        # Análisis de movimiento
        self.movement_analyzer = MovementAnalyzer(window_size=15)
        
        # Exponential smoothing adaptativo
        self.base_alpha = config.STABILIZATION['exponential_alpha']
        self.current_alpha = self.base_alpha
        
        # Predicción mejorada
        self.use_prediction = config.STABILIZATION['use_prediction']
        self.prediction_frames = config.STABILIZATION['prediction_frames']
        self.prediction_weight = config.STABILIZATION['prediction_weight']
        self.position_history = deque(maxlen=30)
        
        # Límites adaptativos
        self.base_max_velocity = config.STABILIZATION['max_velocity_px_per_frame']
        self.max_velocity = self.base_max_velocity
        self.max_acceleration = config.STABILIZATION['max_acceleration_px_per_frame2']
        
        # Deadzone adaptativa
        self.base_deadzone = config.STABILIZATION['deadzone_pixels']
        self.current_deadzone = self.base_deadzone
        
        # Easing
        self.use_easing = config.STABILIZATION['use_easing']
        self.easing_function = config.STABILIZATION['easing_function']
        
        # Estado
        self.prev_position = None
        self.prev_velocity = 0
        self.smooth_position = None
        
        # Detección de tracking perdido
        self.lost_tracking_frames = 0
        self.recovery_mode = False
        
        # Métricas
        self.stability_scores = deque(maxlen=30)
        
    def stabilize(self, raw_position, tracking_quality=None):
        """
        Estabilización adaptativa con análisis de calidad
        
        Args:
            raw_position: Posición raw a estabilizar
            tracking_quality: TrackingQuality object (opcional)
        """
        # Manejar pérdida de tracking
        if raw_position is None:
            self.lost_tracking_frames += 1
            if self.prev_position is not None and self.lost_tracking_frames < 10:
                # Usar predicción durante pérdida temporal
                return self._handle_lost_tracking()
            else:
                return self.prev_position
        
        # Resetear contador de pérdida
        if self.lost_tracking_frames > 0:
            self.recovery_mode = True
        self.lost_tracking_frames = 0
        
        # Analizar tipo de movimiento
        movement_type = self.movement_analyzer.analyze(raw_position)
        
        # Adaptar parámetros según movimiento
        self._adapt_to_movement(movement_type, tracking_quality)
        
        # Aplicar método principal
        if self.method == 'kalman':
            stabilized = self._kalman_stabilize(raw_position, movement_type)
        elif self.method == 'exponential':
            stabilized = self._exponential_stabilize(raw_position)
        elif self.method == 'hybrid':
            stabilized = self._hybrid_stabilize(raw_position, movement_type)
        else:
            stabilized = raw_position
        
        # Aplicar deadzone adaptativa
        stabilized = self._apply_adaptive_deadzone(stabilized, movement_type)
        
        # Aplicar límites de velocidad adaptativos
        stabilized = self._apply_adaptive_velocity_limits(stabilized, movement_type)
        
        # Aplicar predicción si está habilitada
        if self.use_prediction and not self.recovery_mode:
            stabilized = self._apply_smart_prediction(stabilized)
        
        # Aplicar easing
        if self.use_easing and self.prev_position is not None:
            if self.recovery_mode:
                # Easing más agresivo durante recuperación
                stabilized = self._apply_recovery_easing(self.prev_position, stabilized)
            else:
                stabilized = self._apply_easing(self.prev_position, stabilized)
        
        # Actualizar estado
        self.position_history.append(stabilized)
        self.prev_position = stabilized
        
        # Calcular métrica de estabilidad
        smoothness = self.movement_analyzer.get_smoothness_score()
        self.stability_scores.append(smoothness)
        
        # Desactivar modo recuperación después de algunos frames
        if self.recovery_mode and len(self.position_history) > 5:
            self.recovery_mode = False
        
        return stabilized
    
    def _adapt_to_movement(self, movement_type, tracking_quality=None):
        """Adaptar parámetros según tipo de movimiento y calidad"""
        
        # Ajustar alpha de exponential smoothing
        if movement_type == MovementType.STATIC:
            self.current_alpha = min(0.98, self.base_alpha + 0.1)
        elif movement_type == MovementType.SMOOTH:
            self.current_alpha = self.base_alpha
        elif movement_type == MovementType.MODERATE:
            self.current_alpha = max(0.75, self.base_alpha - 0.1)
        elif movement_type == MovementType.RAPID:
            self.current_alpha = max(0.65, self.base_alpha - 0.2)
        elif movement_type == MovementType.ERRATIC:
            self.current_alpha = min(0.95, self.base_alpha + 0.15)
        
        # Ajustar velocidad máxima
        if movement_type == MovementType.RAPID:
            self.max_velocity = self.base_max_velocity * 1.8
        elif movement_type == MovementType.MODERATE:
            self.max_velocity = self.base_max_velocity * 1.3
        else:
            self.max_velocity = self.base_max_velocity
        
        # Ajustar deadzone según calidad de tracking
        if tracking_quality and not tracking_quality.is_reliable:
            # Más deadzone si el tracking es poco confiable
            self.current_deadzone = self.base_deadzone * 2.0
        else:
            self.current_deadzone = self.base_deadzone
    
    def _handle_lost_tracking(self):
        """Manejar pérdida temporal de tracking con predicción"""
        if len(self.position_history) < 3:
            return self.prev_position
        
        # Predicción usando tendencia reciente
        recent = list(self.position_history)[-5:]
        velocities = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        avg_velocity = np.median(velocities) if velocities else 0
        
        # Aplicar decay
        decay = 0.7 ** self.lost_tracking_frames
        predicted = self.prev_position + avg_velocity * decay
        
        return predicted
    
    def _kalman_stabilize(self, position, movement_type):
        """Estabilización con Kalman adaptativo"""
        return self.kalman_x.update(position, movement_type)
    
    def _exponential_stabilize(self, position):
        """Estabilización exponencial con alpha adaptativo"""
        if self.smooth_position is None:
            self.smooth_position = position
            return position
        
        self.smooth_position = (
            self.smooth_position * self.current_alpha + 
            position * (1 - self.current_alpha)
        )
        return self.smooth_position
    
    def _hybrid_stabilize(self, position, movement_type):
        """Híbrido mejorado: Kalman + Exponential"""
        kalman_result = self.kalman_x.update(position, movement_type)
        
        if self.smooth_position is None:
            self.smooth_position = kalman_result
            return kalman_result
        
        self.smooth_position = (
            self.smooth_position * self.current_alpha + 
            kalman_result * (1 - self.current_alpha)
        )
        return self.smooth_position
    
    def _apply_adaptive_deadzone(self, position, movement_type):
        """Deadzone que se adapta al tipo de movimiento"""
        if self.prev_position is None:
            return position
        
        # Deadzone más grande para movimiento errático
        if movement_type == MovementType.ERRATIC:
            deadzone = self.current_deadzone * 1.5
        else:
            deadzone = self.current_deadzone
        
        diff = abs(position - self.prev_position)
        if diff < deadzone:
            return self.prev_position
        
        return position
    
    def _apply_adaptive_velocity_limits(self, position, movement_type):
        """Límites de velocidad adaptativos"""
        if self.prev_position is None:
            return position
        
        proposed_velocity = position - self.prev_position
        
        # Limitar velocidad
        if abs(proposed_velocity) > self.max_velocity:
            proposed_velocity = np.sign(proposed_velocity) * self.max_velocity
        
        # Limitar aceleración (más estricto en movimiento errático)
        if movement_type == MovementType.ERRATIC:
            max_acc = self.max_acceleration * 0.5
        else:
            max_acc = self.max_acceleration
        
        acceleration = proposed_velocity - self.prev_velocity
        if abs(acceleration) > max_acc:
            acceleration = np.sign(acceleration) * max_acc
            proposed_velocity = self.prev_velocity + acceleration
        
        self.prev_velocity = proposed_velocity
        return self.prev_position + proposed_velocity
    
    def _apply_smart_prediction(self, position):
        """Predicción inteligente basada en patrones"""
        if len(self.position_history) < 5:
            return position
        
        # Detectar si hay un patrón de movimiento
        recent = list(self.position_history)[-10:]
        velocities = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        
        # Si hay consistencia en la dirección, predecir
        if len(velocities) >= 3:
            signs = [np.sign(v) for v in velocities[-5:]]
            if all(s == signs[0] for s in signs) and signs[0] != 0:
                # Movimiento consistente
                avg_velocity = np.median(velocities[-3:])
                predicted = position + avg_velocity * 0.3
                
                # Blend adaptativo
                weight = self.prediction_weight * 0.7
                return position * (1 - weight) + predicted * weight
        
        return position
    
    def _apply_easing(self, start, end):
        """Easing estándar"""
        diff = end - start
        
        if self.easing_function == 'linear':
            return end
        elif self.easing_function == 'smooth':
            # Smoothstep
            t = 0.5
            t = t * t * (3 - 2 * t)
            return start + diff * t
        elif self.easing_function == 'cubic':
            return start + diff * 0.6
        else:
            return end
    
    def _apply_recovery_easing(self, start, end):
        """Easing más suave durante recuperación de tracking"""
        diff = end - start
        
        # Easing muy suave para recuperación
        t = 0.3  # Transición gradual
        t = t * t * (3 - 2 * t)  # Smoothstep
        
        return start + diff * t
    
    def get_stability_score(self) -> float:
        """Obtener score de estabilidad actual (0-1)"""
        if not self.stability_scores:
            return 1.0
        return np.mean(list(self.stability_scores))
    
    def reset(self):
        """Resetear estabilizador"""
        if hasattr(self, 'kalman_x'):
            self.kalman_x.posterior_estimate = None
            self.kalman_y.posterior_estimate = None
        
        self.smooth_position = None
        self.prev_position = None
        self.prev_velocity = 0
        self.position_history.clear()
        self.lost_tracking_frames = 0
        self.recovery_mode = False
        self.stability_scores.clear()
        self.movement_analyzer = MovementAnalyzer()


class MultiPassStabilizer:
    """
    Estabilizador multi-paso para máxima calidad
    """
    def __init__(self, config):
        self.config = config
        self.positions_buffer = []
        
    def add_position(self, timestamp, position, tracking_quality=None):
        """Agregar posición al buffer"""
        self.positions_buffer.append({
            'timestamp': timestamp,
            'position': position,
            'quality': tracking_quality
        })
    
    def process(self) -> List[Tuple[float, float]]:
        """
        Procesar todas las posiciones con multi-paso
        
        Returns:
            Lista de (timestamp, posición_estabilizada)
        """
        if not self.positions_buffer:
            return []
        
        # Paso 1: Filtrado inicial
        filtered = self._pass1_initial_filter()
        
        # Paso 2: Suavizado agresivo
        smoothed = self._pass2_aggressive_smooth(filtered)
        
        # Paso 3: Refinamiento
        refined = self._pass3_refine(smoothed)
        
        return refined
    
    def _pass1_initial_filter(self):
        """Paso 1: Remover outliers y aplicar Kalman"""
        positions = [p['position'] for p in self.positions_buffer]
        timestamps = [p['timestamp'] for p in self.positions_buffer]
        qualities = [p.get('quality') for p in self.positions_buffer]
        
        # Detectar y remover outliers
        filtered_positions = self._remove_outliers(positions, qualities)
        
        # Aplicar Kalman
        kalman = AdaptiveKalmanFilter()
        kalman_filtered = []
        for pos in filtered_positions:
            kalman_filtered.append(kalman.update(pos))
        
        return list(zip(timestamps, kalman_filtered))
    
    def _pass2_aggressive_smooth(self, positions):
        """Paso 2: Suavizado agresivo con ventana móvil"""
        timestamps = [p[0] for p in positions]
        values = [p[1] for p in positions]
        
        # Suavizado gaussiano
        smoothed = self._gaussian_smooth(np.array(values), window=9, sigma=3.0)
        
        return list(zip(timestamps, smoothed))
    
    def _pass3_refine(self, positions):
        """Paso 3: Refinamiento final"""
        timestamps = [p[0] for p in positions]
        values = [p[1] for p in positions]
        
        # Aplicar límites de velocidad suaves
        refined = [values[0]]
        max_velocity = self.config.STABILIZATION['max_velocity_px_per_frame'] * 0.7
        
        for i in range(1, len(values)):
            diff = values[i] - refined[-1]
            if abs(diff) > max_velocity:
                diff = np.sign(diff) * max_velocity
            refined.append(refined[-1] + diff)
        
        return list(zip(timestamps, refined))
    
    def _remove_outliers(self, positions, qualities, threshold=3.0):
        """Remover outliers usando desviación estándar"""
        positions_array = np.array(positions)
        
        # Calcular Z-scores
        mean = np.mean(positions_array)
        std = np.std(positions_array)
        
        if std == 0:
            return positions
        
        filtered = []
        for i, pos in enumerate(positions):
            z_score = abs((pos - mean) / std)
            
            # Si tiene baja calidad, ser más agresivo
            quality = qualities[i] if qualities[i] else None
            adj_threshold = threshold * 0.6 if (quality and not quality.is_reliable) else threshold
            
            if z_score < adj_threshold:
                filtered.append(pos)
            else:
                # Reemplazar outlier con media de vecinos
                if i > 0 and i < len(positions) - 1:
                    filtered.append((positions[i-1] + positions[i+1]) / 2)
                else:
                    filtered.append(positions[i-1] if i > 0 else positions[i+1])
        
        return filtered
    
    def _gaussian_smooth(self, values, window=7, sigma=2.0):
        """Suavizado gaussiano"""
        half_window = window // 2
        x = np.arange(-half_window, half_window + 1)
        gaussian = np.exp(-(x**2) / (2 * sigma**2))
        gaussian = gaussian / gaussian.sum()
        
        # Aplicar convolución
        padded = np.pad(values, half_window, mode='edge')
        smoothed = np.convolve(padded, gaussian, mode='valid')
        
        return smoothed
