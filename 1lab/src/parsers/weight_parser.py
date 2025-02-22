import re
from typing import Dict, List, Tuple


def parse_neural_network_weights(text: str) -> Tuple[int, Dict[Tuple[int, int], List[float]]]:
    """
    Парсит веса нейронной сети из текста.
    
    Args:
        text (str): Текст с данными нейронной сети
        
    Returns:
        Tuple[int, Dict]: Кортеж из количества циклов обучения и словаря весов
    """
    weights = {}
    
    # Получаем количество циклов обучения
    cycles_match = re.search(r'Циклов обучения: (\d+)', text)
    training_cycles = int(cycles_match.group(1)) if cycles_match else None
    
    # Ищем секцию инициализации весов
    init_section = re.search(r'Инициализация весов синапсов.*?(?=Выбираем допустимый образ)', text, re.DOTALL)
    if init_section:
        init_text = init_section.group(0)
        
        # Паттерн для поиска информации о нейроне
        neuron_pattern = r'Нейрон\[(\d+)\]\[(\d+)\](.*?)(?=Нейрон\[|$)'
        
        # Находим все нейроны
        neurons = re.finditer(neuron_pattern, init_text, re.DOTALL)
        
        for neuron in neurons:
            layer = int(neuron.group(1))
            number = int(neuron.group(2))
            neuron_text = neuron.group(3)
            
            # Паттерн для поиска весов
            weight_pattern = r'w\[[\d,\s]+\]\s*=\s*([-\d,.]+)'
            
            # Находим все веса для текущего нейрона
            weights_values = re.findall(weight_pattern, neuron_text)
            
            # Преобразуем строковые значения в float, заменяя запятую на точку
            weights_values = [float(w.replace(',', '.')) for w in weights_values]
            
            # Сохраняем веса в словарь (без веса смещения)
            key = (layer, number)
            weights[key] = weights_values[:-1]
    
    return training_cycles, weights 