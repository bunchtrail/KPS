import logging
import os
import sys
import traceback
from typing import Dict, List, Tuple, Union

import pandas as pd

# Создаем директорию для логов, если её нет
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Настройка логирования
log_file = os.path.join(log_dir, "weight_correction.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class WeightCorrectionTableCreator:
    """Класс для создания таблицы с новыми весами"""
    
    def __init__(self, 
                 old_weights: Union[Dict[Tuple[int, int], List[float]], List[List[float]]],
                 new_weights: Union[Dict[Tuple[int, int], List[float]], List[List[float]]],
                 new_biases: Union[Dict[Tuple[int, int], float], List[float]]):
        """
        Инициализация создателя таблицы
        
        Args:
            old_weights: Словарь или список старых весов
            new_weights: Словарь или список новых весов
            new_biases: Словарь или список новых смещений
        """
        try:
            logger.info("Инициализация WeightCorrectionTableCreator")
            logger.info(f"Тип old_weights: {type(old_weights)}, значение: {old_weights}")
            logger.info(f"Тип new_weights: {type(new_weights)}, значение: {new_weights}")
            logger.info(f"Тип new_biases: {type(new_biases)}, значение: {new_biases}")
            
            # Преобразуем входные данные в словари, если они переданы как списки
            self.old_weights = self._ensure_dict(old_weights, "old_weights")
            self.new_weights = self._ensure_dict(new_weights, "new_weights")
            self.new_biases = self._ensure_dict(new_biases, "new_biases")
            
            logger.info("Преобразование входных данных завершено успешно")
            logger.debug(f"Преобразованный old_weights: {self.old_weights}")
            logger.debug(f"Преобразованный new_weights: {self.new_weights}")
            logger.debug(f"Преобразованный new_biases: {self.new_biases}")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def _ensure_dict(self, data: Union[Dict, List], name: str) -> Dict:
        """Преобразует данные в словарь, если они переданы как список"""
        try:
            logger.info(f"Начало преобразования данных {name}")
            logger.debug(f"Входные данные {name}: {data}")
            logger.debug(f"Тип входных данных {name}: {type(data)}")
            
            if data is None:
                error_msg = f"Параметр {name} не может быть None"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if isinstance(data, dict):
                logger.debug(f"{name} уже является словарем")
                return data
            
            if isinstance(data, list):
                logger.debug(f"Преобразование списка {name} в словарь")
                if not data:  # Пустой список
                    error_msg = f"Параметр {name} не может быть пустым списком"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if name == "new_biases":
                    # Для смещений создаем словарь с ключами (layer, neuron)
                    result = {}
                    for layer in [1, 2]:
                        neurons = 10 if layer == 1 else 1
                        for neuron in range(1, neurons + 1):
                            idx = (neuron - 1) if layer == 1 else 10
                            if idx < len(data):
                                result[(layer, neuron)] = float(data[idx])  # Преобразуем в float
                                logger.debug(f"Установлено смещение для слоя {layer}, нейрона {neuron}: {data[idx]}")
                            else:
                                result[(layer, neuron)] = 1.0
                                logger.debug(f"Установлено значение по умолчанию для слоя {layer}, нейрона {neuron}: 1.0")
                    return result
                else:
                    # Для весов создаем словарь с ключами (layer, neuron)
                    result = {}
                    for layer in [1, 2]:
                        neurons = 10 if layer == 1 else 1
                        for neuron in range(1, neurons + 1):
                            idx = (neuron - 1) if layer == 1 else 10
                            if idx < len(data):
                                weights = data[idx]
                                if not isinstance(weights, list):
                                    logger.debug(f"Преобразование скалярного значения {weights} в список")
                                    weights = [float(weights)]  # Преобразуем в float
                                else:
                                    weights = [float(w) for w in weights]  # Преобразуем все элементы в float
                                result[(layer, neuron)] = weights
                                logger.debug(f"Установлены веса для слоя {layer}, нейрона {neuron}: {weights}")
                            else:
                                default_weights = [0.0] * (3 if layer == 1 else 10)
                                result[(layer, neuron)] = default_weights
                                logger.debug(f"Установлены веса по умолчанию для слоя {layer}, нейрона {neuron}: {default_weights}")
                    return result
            
            error_msg = f"Параметр {name} должен быть словарем или списком, получен {type(data)}"
            logger.error(error_msg)
            raise TypeError(error_msg)
            
        except Exception as e:
            logger.error(f"Ошибка при преобразовании {name}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def create_table(self, output_file: str):
        """
        Создает Excel таблицу с новыми весами
        
        Args:
            output_file: Путь к выходному файлу
        """
        logger.info(f"Создание таблицы Excel: {output_file}")
        # Создаем список строк для таблицы
        rows = []
        
        try:
            # Добавляем строки для скрытого слоя
            for neuron in range(1, 11):  # 10 нейронов
                logger.debug(f"Обработка нейрона {neuron} скрытого слоя")
                old_weights = self.old_weights.get((1, neuron), [0.0] * 3)
                new_weights = self.new_weights.get((1, neuron), [0.0] * 3)
                new_bias = self.new_biases.get((1, neuron), 1.0)
                
                # Добавляем строки для каждого входа
                for i in range(3):  # 3 входа
                    row = {
                        '№ слоя': '1' if i == 0 else '',
                        '№ нейрона': neuron if i == 0 else '',
                        '№ выхода': i + 1,
                        'Предыдущий весовой коэффициент wij(t)': old_weights[i],
                        'Предыдущий вес смещения Tj(t)': 1.0 if i == 0 else '',
                        'Новый весовой коэффициент wij(t+1)': new_weights[i],
                        'Новый вес смещения Tj(t+1)': new_bias if i == 0 else ''
                    }
                    logger.debug(f"Добавлена строка для входа {i+1}: {row}")
                    rows.append(row)
            
            # Добавляем строки для выходного слоя
            logger.debug("Обработка выходного слоя")
            old_weights = self.old_weights.get((2, 1), [0.0] * 10)
            new_weights = self.new_weights.get((2, 1), [0.0] * 10)
            new_bias = self.new_biases.get((2, 1), 1.0)
            
            # Добавляем строки для каждого входа
            for i in range(10):  # 10 входов от скрытого слоя
                row = {
                    '№ слоя': 'Выход' if i == 0 else '',
                    '№ нейрона': 1 if i == 0 else '',
                    '№ выхода': i + 1,
                    'Предыдущий весовой коэффициент wij(t)': old_weights[i],
                    'Предыдущий вес смещения Tj(t)': 1.0 if i == 0 else '',
                    'Новый весовой коэффициент wij(t+1)': new_weights[i],
                    'Новый вес смещения Tj(t+1)': new_bias if i == 0 else ''
                }
                logger.debug(f"Добавлена строка для входа {i+1}: {row}")
                rows.append(row)
            
            # Создаем DataFrame
            df = pd.DataFrame(rows)
            logger.debug("DataFrame создан успешно")
            
            # Создаем Excel writer
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Таблица 10')
                
                # Получаем рабочий лист
                worksheet = writer.sheets['Таблица 10']
                
                # Устанавливаем ширину столбцов
                for column in worksheet.columns:
                    max_length = 0
                    column = list(column)
                    for cell in column:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                
                # Устанавливаем формат чисел (8 знаков после запятой)
                for row in worksheet.iter_rows(min_row=2):  # Пропускаем заголовки
                    for cell in row:
                        if isinstance(cell.value, (int, float)):
                            cell.number_format = '0.00000000'
                
        except Exception as e:
            logger.error(f"Ошибка при создании таблицы: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 