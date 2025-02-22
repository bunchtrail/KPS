import re
from typing import Dict, Tuple

def parse_weighted_sums(file_content: str) -> Dict[Tuple[int, int], float]:
    """
    Парсит взвешенные суммы из файла.
    
    Args:
        file_content (str): Содержимое файла
        
    Returns:
        Dict[Tuple[int, int], float]: Словарь взвешенных сумм для каждого нейрона
    """
    sums = {}
    current_layer = None
    current_neuron = None
    
    lines = file_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        neuron_match = re.search(r'Нейрон\[(\d+)\]\[(\d+)\]', line)
        if neuron_match:
            current_layer = int(neuron_match.group(1))
            current_neuron = int(neuron_match.group(2))
            i += 1
            while i < len(lines):
                current_line = lines[i].strip()
                if 'Нейрон[' in current_line:
                    break
                sum_match = re.search(r'Взвешенная сумма = ([-\d.,]+)', current_line)
                if sum_match:
                    sum_value = float(sum_match.group(1).replace(',', '.'))
                    sums[(current_layer, current_neuron)] = sum_value
                i += 1
            continue
        i += 1
    
    return sums 