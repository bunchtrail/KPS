from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


class ErrorTableCreator:
    def __init__(self, errors: Dict[Tuple[int, int], Tuple[float, float, float]]):
        """
        Инициализация генератора таблицы ошибок.
        
        Args:
            errors: Словарь с результатами для каждого нейрона (Si, F'(Si), ошибка)
        """
        self.errors = errors
    
    def create_table(self, output_file: str) -> None:
        """
        Создает Excel таблицу с ошибками нейронной сети.
        
        Args:
            output_file: Путь к выходному файлу
        """
        data = []
        
        # Добавляем данные скрытого слоя
        for i in range(1, 11):
            si, derivative, error = self.errors.get((1, i), (0.0, 0.0, 0.0))
            data.append({
                '№ слоя': '1' if i == 1 else '',
                '№ нейрона': i,
                'Si': si,
                "F'(Si)": derivative,
                'Ошибка': error
            })
        
        # Добавляем данные выходного слоя
        si, derivative, error = self.errors.get((2, 1), (0.0, 0.0, 0.0))
        data.append({
            '№ слоя': 'Выход',
            '№ нейрона': 1,
            'Si': si,
            "F'(Si)": derivative,
            'Ошибка': error
        })
        
        # Создаем DataFrame
        df = pd.DataFrame(data)
        
        # Создаем Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Ошибки', index=False)
            
            # Получаем объект workbook и worksheet
            workbook = writer.book
            worksheet = writer.sheets['Ошибки']
            
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
        
        # Применяем форматирование к заголовкам
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 15)
        
        # Применяем форматирование к ячейкам
        for row in range(len(df)):
            for col in range(len(df.columns)):
                cell_value = df.iloc[row, col]
                worksheet.write(row + 1, col, cell_value, cell_format) 