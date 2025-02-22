from pathlib import Path
from typing import Dict, Optional, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QMainWindow, QMessageBox, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)

from excel_generator.error_table_creator import ErrorTableCreator
from excel_generator.excel_creator import ExcelCreator
from excel_generator.weight_correction_table_creator import \
    WeightCorrectionTableCreator
from parsers.signal_parser import parse_input_signals
from parsers.sum_parser import parse_weighted_sums
from parsers.weight_parser import parse_neural_network_weights
from utils.calculations import calculate_errors, calculate_new_weights


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.input_file: Optional[Path] = None
        self.wi: Optional[float] = None
        
        self.init_ui()
    
    def init_ui(self):
        """Инициализация пользовательского интерфейса"""
        self.setWindowTitle('Обработка данных нейронной сети')
        self.setMinimumWidth(600)
        
        # Создаем центральный виджет и главный layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Секция входного файла
        input_layout = QHBoxLayout()
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText('Путь к входному файлу...')
        self.input_path_edit.setReadOnly(True)
        input_button = QPushButton('Выбрать входной файл')
        input_button.clicked.connect(self.select_input_file)
        input_layout.addWidget(self.input_path_edit)
        input_layout.addWidget(input_button)
        main_layout.addLayout(input_layout)
        
        # Секция коэффициента крутизны
        wi_layout = QHBoxLayout()
        self.wi_edit = QLineEdit()
        self.wi_edit.setPlaceholderText('Введите коэффициент крутизны (wi)...')
        wi_layout.addWidget(QLabel('Коэффициент крутизны (wi):'))
        wi_layout.addWidget(self.wi_edit)
        main_layout.addLayout(wi_layout)
        
        # Секция целевого значения
        target_layout = QHBoxLayout()
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText('Введите целевое значение (t)...')
        target_layout.addWidget(QLabel('Целевое значение (t):'))
        target_layout.addWidget(self.target_edit)
        main_layout.addLayout(target_layout)
        
        # Лог операций
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        main_layout.addWidget(QLabel('Лог операций:'))
        main_layout.addWidget(self.log_text)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        process_button = QPushButton('Создать таблицу весов')
        process_button.clicked.connect(self.process_weights_table)
        process_button.setMinimumHeight(40)
        buttons_layout.addWidget(process_button)
        
        errors_button = QPushButton('Создать таблицу ошибок')
        errors_button.clicked.connect(self.process_errors_table)
        errors_button.setMinimumHeight(40)
        buttons_layout.addWidget(errors_button)
        
        correction_button = QPushButton('Создать таблицу новых весов')
        correction_button.clicked.connect(self.process_weight_correction)
        correction_button.setMinimumHeight(40)
        buttons_layout.addWidget(correction_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Устанавливаем размер окна и показываем его
        self.setGeometry(100, 100, 800, 500)
        self.show()
    
    def select_input_file(self):
        """Выбор входного файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите входной файл',
            str(Path.home()),
            'Текстовые файлы (*.txt)'
        )
        if file_path:
            self.input_file = Path(file_path)
            self.input_path_edit.setText(str(self.input_file))
            self.log('Выбран входной файл: ' + str(self.input_file))
    
    def get_output_file(self, suffix: str) -> Path:
        """Создает путь к выходному файлу на основе входного"""
        input_stem = self.input_file.stem
        return self.input_file.parent / f"{input_stem}_{suffix}.xlsx"
    
    def validate_wi(self) -> bool:
        """Проверка коэффициента крутизны"""
        try:
            wi_text = self.wi_edit.text().strip().replace(',', '.')
            if not wi_text:
                self.show_error('Ошибка', 'Введите коэффициент крутизны!')
                return False
            self.wi = float(wi_text)
            return True
        except ValueError:
            self.show_error('Ошибка', 'Некорректное значение коэффициента крутизны!')
            return False
    
    def validate_target(self) -> Optional[float]:
        """Проверка целевого значения"""
        try:
            target_text = self.target_edit.text().strip().replace(',', '.')
            if not target_text:
                return 0.0
            return float(target_text)
        except ValueError:
            self.show_error('Ошибка', 'Некорректное целевое значение!')
            return None
    
    def validate_input_file(self) -> bool:
        """Проверка наличия входного файла"""
        if not self.input_file:
            self.show_error('Ошибка', 'Выберите входной файл!')
            return False
        if not self.input_file.exists():
            self.show_error('Ошибка', f'Файл {self.input_file} не существует!')
            return False
        return True
    
    def read_input_file(self) -> Optional[str]:
        """Чтение входного файла"""
        try:
            # Сначала пробуем прочитать в UTF-8
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Если не получилось, читаем в CP1251
            with open(self.input_file, 'r', encoding='cp1251') as f:
                return f.read()
        except Exception as e:
            self.show_error('Ошибка', f'Ошибка при чтении файла: {str(e)}')
            return None
    
    def process_weights_table(self):
        """Создание таблицы весов"""
        if not self.validate_input_file() or not self.validate_wi():
            return
        
        try:
            content = self.read_input_file()
            if not content:
                return
            
            output_file = self.get_output_file('weights')
            
            # Парсим данные
            self.log('Парсинг весов нейронной сети...')
            _, weights = parse_neural_network_weights(content)
            
            self.log('Парсинг взвешенных сумм...')
            weighted_sums = parse_weighted_sums(content)
            
            self.log('Парсинг входных сигналов...')
            input_signals = parse_input_signals(content)
            
            # Создаем Excel файл
            self.log('Создание таблицы весов...')
            excel_creator = ExcelCreator(weights, weighted_sums, input_signals, self.wi)
            excel_creator.create_table(str(output_file))
            
            self.log(f'Таблица весов создана: {output_file}')
            self.show_info('Успех', f'Таблица весов создана:\n{output_file}')
            
        except Exception as e:
            self.show_error('Ошибка', f'Произошла ошибка при обработке данных: {str(e)}')
    
    def process_errors_table(self):
        """Создание таблицы ошибок"""
        if not self.validate_input_file() or not self.validate_wi():
            return
            
        target = self.validate_target()
        if target is None:
            return
        
        try:
            content = self.read_input_file()
            if not content:
                return
            
            output_file = self.get_output_file('errors')
            
            # Парсим данные
            self.log('Парсинг весов нейронной сети...')
            _, weights = parse_neural_network_weights(content)
            
            self.log('Парсинг взвешенных сумм...')
            weighted_sums = parse_weighted_sums(content)
            
            # Рассчитываем ошибки
            self.log('Расчет ошибок...')
            errors = calculate_errors(weighted_sums, weights, self.wi, target, self.log)
            
            # Создаем Excel файл
            self.log('Создание таблицы ошибок...')
            error_creator = ErrorTableCreator(errors)
            error_creator.create_table(str(output_file))
            
            self.log(f'Таблица ошибок создана: {output_file}')
            self.show_info('Успех', f'Таблица ошибок создана:\n{output_file}')
            
        except Exception as e:
            self.show_error('Ошибка', f'Произошла ошибка при обработке данных: {str(e)}')
    
    def process_weight_correction(self):
        """Создание таблицы с новыми весами"""
        if not self.validate_input_file() or not self.validate_wi():
            return
            
        target = self.validate_target()
        if target is None:
            return
        
        try:
            content = self.read_input_file()
            if not content:
                return
            
            output_file = self.get_output_file('weight_correction')
            
            # Парсим данные
            self.log('Парсинг весов нейронной сети...')
            _, weights = parse_neural_network_weights(content)
            
            self.log('Парсинг взвешенных сумм...')
            weighted_sums = parse_weighted_sums(content)
            
            self.log('Парсинг входных сигналов...')
            input_signals = parse_input_signals(content)
            
            # Создаем словарь смещений (все равны 1.0)
            biases = {(layer, neuron): 1.0 
                     for layer in [1, 2] 
                     for neuron in range(1, 11 if layer == 1 else 2)}
            
            # Рассчитываем ошибки
            self.log('Расчет ошибок...')
            errors = calculate_errors(weighted_sums, weights, self.wi, 0.69266, self.log)
            
            # Рассчитываем новые веса
            self.log('Расчет новых весов...')
            new_weights, new_biases = calculate_new_weights(
                weights, biases, errors, input_signals, self.wi, self.log
            )
            
            # Создаем Excel файл
            self.log('Создание таблицы новых весов...')
            correction_creator = WeightCorrectionTableCreator(
                weights, new_weights, new_biases
            )
            correction_creator.create_table(str(output_file))
            
            self.log(f'Таблица новых весов создана: {output_file}')
            self.show_info('Успех', f'Таблица новых весов создана:\n{output_file}')
            
        except Exception as e:
            self.show_error('Ошибка', f'Произошла ошибка при обработке данных: {str(e)}')
    
    def log(self, message: str):
        """Добавление сообщения в лог"""
        self.log_text.append(message)
    
    def show_error(self, title: str, message: str):
        """Показ сообщения об ошибке"""
        QMessageBox.critical(self, title, message)
    
    def show_info(self, title: str, message: str):
        """Показ информационного сообщения"""
        QMessageBox.information(self, title, message) 