from typing import List

def parse_input_signals(file_content: str) -> List[float]:
    """
    Парсит входные сигналы из файла.
    
    Args:
        file_content (str): Содержимое файла
        
    Returns:
        List[float]: Список входных сигналов
    """
    signals = []
    
    for line in file_content.split('\n'):
        if 'Аксон = ' in line:
            value = float(line.split('=')[1].strip().replace(',', '.'))
            signals.append(value)
    
    return signals[:3]  # Возвращаем только первые три сигнала 