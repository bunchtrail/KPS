from typing import Dict, List, Tuple

import pandas as pd


class ExcelCreator:
    def __init__(self, weights: Dict[Tuple[int, int], List[float]], 
                 weighted_sums: Dict[Tuple[int, int], float],
                 input_signals: List[float],
                 alpha: float):
        """
        Инициализация генератора Excel файла.
        
        Args:
            weights: Словарь весов нейронов
            weighted_sums: Словарь взвешенных сумм
            input_signals: Список входных сигналов
            alpha: Коэффициент крутизны α
        """
        self.weights = weights
        self.weighted_sums = weighted_sums
        self.input_signals = input_signals
        self.alpha = alpha
        
    def create_table(self, output_file: str) -> None:
        """
        Создает Excel таблицу с данными нейронной сети.
        
        Args:
            output_file: Путь к выходному файлу
        """
        data = []
        
        # Добавляем входной слой
        for i, signal in enumerate(self.input_signals, 1):
            data.append({
                '№ Слоя': 'Вход' if i == 1 else '',
                '№ Нейрона': i,
                '№ Выхода': 1,
                'Входной сигнал xi': signal,
                'Весовой коэффициент wij': '-',
                'Смещение wi0': '-',
                'Вес смещения': '-',
                'wij * xi': '-',
                'Взвешенная сумма Si': '-',
                'Выход нейрона yi = F(Si)': signal
            })
        
        # Добавляем скрытый слой
        for neuron in range(1, 11):
            neuron_weights = self.weights.get((1, neuron), [])
            weighted_sum = self.weighted_sums.get((1, neuron), '')
            
            # Три строки для каждого нейрона
            for i in range(3):
                data.append({
                    '№ Слоя': '1' if i == 0 else '',
                    '№ Нейрона': neuron if i == 0 else '',
                    '№ Выхода': i + 1,
                    'Входной сигнал xi': self.input_signals[i],
                    'Весовой коэффициент wij': neuron_weights[i] if i < len(neuron_weights) else None,
                    'Смещение wi0': self.alpha,
                    'Вес смещения': 1,
                    'wij * xi': f'=D{len(data) + 2}*E{len(data) + 2}',
                    'Взвешенная сумма Si': weighted_sum,
                    'Выход нейрона yi = F(Si)': f'=2/(1+EXP(-{self.alpha}*I{len(data) + 2}))-1'
                })
        
        # Добавляем выходной слой
        output_weights = self.weights.get((2, 1), [])
        weighted_sum = self.weighted_sums.get((2, 1), '')
        
        # Десять строк для выходного нейрона
        for i in range(10):
            data.append({
                '№ Слоя': 'Выход' if i == 0 else '',
                '№ Нейрона': 1 if i == 0 else '',
                '№ Выхода': i + 1,
                'Входной сигнал xi': self.input_signals[0],
                'Весовой коэффициент wij': output_weights[i] if i < len(output_weights) else None,
                'Смещение wi0': self.alpha,
                'Вес смещения': 1,
                'wij * xi': f'=D{len(data) + 2}*E{len(data) + 2}',
                'Взвешенная сумма Si': weighted_sum,
                'Выход нейрона yi = F(Si)': f'=2/(1+EXP(-{self.alpha}*I{len(data) + 2}))-1'
            })
        
        # Создаем DataFrame
        df = pd.DataFrame(data)
        
        # Создаем Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            
            # Получаем объект workbook и worksheet
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            
            # Форматирование
            self._apply_formatting(workbook, worksheet, df)
    
    def _apply_formatting(self, workbook, worksheet, df):
        """
        Применяет форматирование к Excel файлу.
        """
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'vcenter',
            'align': 'center',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'align': 'center',
            'border': 1,
            'num_format': '0.000000'
        })
        
        merge_format = workbook.add_format({
            'align': 'center',
            'valign': 'vcenter',
            'border': 1,
            'num_format': '0.000000'
        })
        
        # Применяем форматирование к заголовкам
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)
        
        # Применяем форматирование к ячейкам и объединяем
        current_row = 1
        while current_row < len(df) + 1:
            # Определяем количество строк для объединения
            if current_row <= 3:  # Входной слой
                merge_rows = 1
            elif current_row >= len(df) - 9:  # Выходной слой
                merge_rows = 10
            else:  # Скрытый слой
                merge_rows = 3
            
            # Объединяем ячейки
            if merge_rows > 1:
                # Объединяем № Слоя
                if df.iloc[current_row-1]['№ Слоя']:
                    worksheet.merge_range(current_row, 0, current_row + merge_rows - 1, 0, 
                                       df.iloc[current_row-1]['№ Слоя'], merge_format)
                # Объединяем № Нейрона
                if df.iloc[current_row-1]['№ Нейрона']:
                    worksheet.merge_range(current_row, 1, current_row + merge_rows - 1, 1, 
                                       df.iloc[current_row-1]['№ Нейрона'], merge_format)
                
                # Объединяем остальные столбцы
                worksheet.merge_range(current_row, 5, current_row + merge_rows - 1, 5, 
                                   df.iloc[current_row-1]['Смещение wi0'], merge_format)
                worksheet.merge_range(current_row, 6, current_row + merge_rows - 1, 6, 
                                   df.iloc[current_row-1]['Вес смещения'], merge_format)
                worksheet.merge_range(current_row, 8, current_row + merge_rows - 1, 8, 
                                   df.iloc[current_row-1]['Взвешенная сумма Si'], merge_format)
                worksheet.merge_range(current_row, 9, current_row + merge_rows - 1, 9, 
                                   df.iloc[current_row-1]['Выход нейрона yi = F(Si)'], merge_format)
            
            # Записываем остальные ячейки
            for row in range(current_row, current_row + merge_rows):
                for col in range(len(df.columns)):
                    if (col not in [0, 1, 5, 6, 8, 9] or merge_rows == 1):
                        cell_value = df.iloc[row-1, col]
                        if isinstance(cell_value, str) and cell_value.startswith('='):
                            worksheet.write_formula(row, col, cell_value, cell_format)
                        else:
                            worksheet.write(row, col, cell_value, cell_format)
            
            current_row += merge_rows 