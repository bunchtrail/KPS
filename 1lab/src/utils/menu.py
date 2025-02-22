from typing import Callable, Dict, List

class Menu:
    def __init__(self):
        self.options: Dict[str, Callable] = {}
        self.descriptions: Dict[str, str] = {}
    
    def add_option(self, key: str, handler: Callable, description: str) -> None:
        """
        Добавляет опцию в меню.
        
        Args:
            key: Клавиша для выбора опции
            handler: Функция-обработчик
            description: Описание опции
        """
        self.options[key] = handler
        self.descriptions[key] = description
    
    def show(self) -> None:
        """
        Отображает меню.
        """
        print("\nМеню:")
        print("-" * 50)
        for key, description in self.descriptions.items():
            print(f"{key}. {description}")
        print("-" * 50)
    
    def handle_choice(self, choice: str) -> bool:
        """
        Обрабатывает выбор пользователя.
        
        Args:
            choice: Выбранная опция
            
        Returns:
            bool: True, если выбор обработан успешно
        """
        if choice in self.options:
            self.options[choice]()
            return True
        return False
    
    def run(self) -> None:
        """
        Запускает интерактивное меню.
        """
        while True:
            self.show()
            choice = input("Выберите опцию: ").strip()
            
            if choice.lower() == 'q':
                print("Выход из программы...")
                break
            
            if not self.handle_choice(choice):
                print("Неверный выбор. Попробуйте снова.") 