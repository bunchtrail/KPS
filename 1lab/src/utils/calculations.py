import math
from typing import Callable, Dict, List, Tuple


def calculate_derivative(s: float, alpha: float, log_func: Callable[[str], None] = None) -> float:
    """
    Рассчитывает производную функции активации для биполярной сигмоиды:
      f(S) = 2/(1+exp(-αS)) - 1,
    и её производную:
      f'(S) = (α/4) * [1 - f(S)^2]
    """
    f_S = 2 / (1 + math.exp(-alpha * s)) - 1
    result = (alpha / 4) * (1 - f_S ** 2)
    if log_func:
        log_func("Расчет F'(S) для биполярной сигмоиды:")
        log_func(f"  S = {s}")
        log_func(f"  α = {alpha}")
        log_func(f"  f(S) = 2/(1+exp(-αS))-1 = {f_S}")
        log_func(f"  F'(S) = (α/4)*(1 - f(S)^2) = {result}")
    return result

def calculate_output_error(actual: float, target: float, derivative: float, log_func: Callable[[str], None] = None) -> float:
    """
    Рассчитывает ошибку для выходного нейрона по формуле:
      γ = 2*(y - t) * F'(S)
    (учтён множитель 2, как в исходном расчёте)
    """
    error = 2 * (actual - target) * derivative
    if log_func:
        log_func("\nРасчет ошибки выходного нейрона:")
        log_func(f"  Фактический выход (y) = {actual}")
        log_func(f"  Целевое значение (t) = {target}")
        log_func(f"  F'(S) = {derivative}")
        log_func(f"  Ошибка γ = 2*(y - t)*F'(S) = {error}")
    return error

def calculate_hidden_error(effective_gamma: float, derivative: float, log_func: Callable[[str], None] = None) -> float:
    """
    Рассчитывает ошибку для нейрона скрытого слоя по формуле:
      γ_i = γ_eff * F'(S_i)
    где эффективное значение ошибки для скрытого нейрона рассчитывается как вклад ошибки нейронов следующего слоя.
    
    (Если в вашей нотации требуется отрицательный знак – добавьте его здесь)
    """
    # Если требуется, можно добавить минус: error = - effective_gamma * derivative
    error = effective_gamma * derivative
    if log_func:
        log_func("\nРасчет ошибки нейрона скрытого слоя:")
        log_func(f"  Эффективное γ_eff = {effective_gamma}")
        log_func(f"  F'(S_i) = {derivative}")
        log_func(f"  Ошибка γ_i = γ_eff * F'(S_i) = {error}")
    return error

def calculate_errors(weighted_sums: Dict[Tuple[int, int], float],
                     weights: Dict[Tuple[int, int], List[float]],
                     alpha: float,
                     target: float = 0.0,
                     log_func: Callable[[str], None] = None) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """
    Рассчитывает ошибки для всех нейронов сети.
    
    Для выходного нейрона (предполагается, что он находится в слое 2):
      S, F'(S), γ = 2*(y-t)*F'(S)
    
    Для нейронов скрытого слоя (слой 1):
      Для каждого нейрона i сначала вычисляем эффективное значение ошибки:
         γ_eff(i) = сумма по k ( γ_k * w_ik )
      (если сеть с одним выходным нейроном, то γ_eff(i) = γ_выход * w[i])
      Затем:
         γ_i = γ_eff(i) * F'(S_i)
    """
    results = {}
    
    if log_func:
        log_func("\n" + "="*50)
        log_func("РАСЧЕТ ОШИБОК НЕЙРОННОЙ СЕТИ")
        log_func("="*50)
        log_func(f"\nКоэффициент крутизны α = {alpha}")
        log_func(f"Целевое значение t = {target}")
    
    # Обработка выходного нейрона [2][1]
    if log_func:
        log_func("\n" + "-"*50)
        log_func("ВЫХОДНОЙ НЕЙРОН [2][1]")
        log_func("-"*50)
    
    output_s = weighted_sums.get((2, 1), 0.0)
    output_derivative = calculate_derivative(output_s, alpha, log_func)
    output_y = 2 / (1 + math.exp(-alpha * output_s)) - 1
    if log_func:
        log_func(f"\nВзвешенная сумма S = {output_s}")
        log_func(f"Фактический выход y = 2/(1+e^(-αS))-1 = {output_y}")
    
    output_error = calculate_output_error(output_y, target, output_derivative, log_func)
    results[(2, 1)] = (output_s, output_derivative, output_error)
    
    # Для скрытого слоя (слой 1)
    if log_func:
        log_func("\n" + "-"*50)
        log_func("СКРЫТЫЙ СЛОЙ [1]")
        log_func("-"*50)
    
    # Получаем веса, ведущие к выходному нейрону.
    # Предполагается, что weights[(2, 1)] — это список весов связей от нейронов скрытого слоя (индексы 0..N-1)
    output_weights = weights.get((2, 1), [])
    for i in range(1, 11):  # для 10 нейронов скрытого слоя
        if log_func:
            log_func(f"\nНЕЙРОН [1][{i}]")
            log_func("-"*30)
        
        s = weighted_sums.get((1, i), 0.0)
        derivative = calculate_derivative(s, alpha, log_func)
        # Если для данного нейрона имеется связь с выходным:
        if i <= len(output_weights):
            w = output_weights[i-1]
        else:
            w = 0.0
        # Динамически вычисляем эффективное значение ошибки для скрытого нейрона:
        # В общем случае суммируем по всем нейронам следующего слоя,
        # а при одном выходном нейроне:
        effective_gamma = output_error * w
        if log_func:
            log_func(f"\nВес связи от [1][{i}] к [2][1] = {w}")
            log_func(f"Вычисленное эффективное γ_eff = γ_выход * w = {output_error} * {w} = {effective_gamma}")
        hidden_error = calculate_hidden_error(effective_gamma, derivative, log_func)
        results[(1, i)] = (s, derivative, hidden_error)
    
    if log_func:
        log_func("\n" + "="*50)
        log_func("РАСЧЕТ ОШИБОК ЗАВЕРШЕН")
        log_func("="*50)
    
    return results

def calculate_new_weight(old_weight: float, 
                         learning_rate: float, 
                         error: float, 
                         input_signal: float,
                         log_func: Callable[[str], None] = None) -> float:
    """
    Рассчитывает новый вес синапса по формуле:
      ω_ij(t+1) = ω_ij(t) - η * γ_j * y_j
    """
    correction = learning_rate * error * input_signal
    new_weight = old_weight - correction
    if log_func:
        log_func("\nРасчет нового веса синапса:")
        log_func(f"  Текущий вес ω_ij(t) = {old_weight}")
        log_func(f"  Скорость обучения η = {learning_rate}")
        log_func(f"  Ошибка γ_j = {error}")
        log_func(f"  Входной сигнал y_j = {input_signal}")
        log_func(f"  Коррекция η * γ_j * y_j = {correction}")
        log_func(f"  Новый вес ω_ij(t+1) = {new_weight}")
    return new_weight

def calculate_new_bias(old_bias: float, 
                       learning_rate: float, 
                       error: float,
                       log_func: Callable[[str], None] = None) -> float:
    """
    Рассчитывает новое смещение по формуле:
      T_j(t+1) = T_j(t) - η * γ_j
    """
    correction = learning_rate * error
    new_bias = old_bias - correction
    if log_func:
        log_func("\nРасчет нового смещения:")
        log_func(f"  Текущее смещение T_j(t) = {old_bias}")
        log_func(f"  Скорость обучения η = {learning_rate}")
        log_func(f"  Ошибка γ_j = {error}")
        log_func(f"  Коррекция η * γ_j = {correction}")
        log_func(f"  Новое смещение T_j(t+1) = {new_bias}")
    return new_bias

def calculate_new_weights(weights: Dict[Tuple[int, int], List[float]],
                          biases: Dict[Tuple[int, int], float],
                          errors: Dict[Tuple[int, int], Tuple[float, float, float]],
                          input_signals: Dict[Tuple[int, int], List[float]],
                          learning_rate: float,
                          log_func: Callable[[str], None] = None) -> Tuple[Dict[Tuple[int, int], List[float]], 
                                                                             Dict[Tuple[int, int], float]]:
    """
    Рассчитывает новые веса и смещения для всех нейронов по формулам:
       ω_ij(t+1) = ω_ij(t) - η * γ_j * y_j
       T_j(t+1)   = T_j(t) - η * γ_j
    """
    new_weights = {}
    new_biases = {}
    
    if log_func:
        log_func("\n" + "="*50)
        log_func("РАСЧЕТ НОВЫХ ВЕСОВ И СМЕЩЕНИЙ")
        log_func("="*50)
    
    for (layer, neuron), (_, _, error) in errors.items():
        if log_func:
            log_func(f"\n{'-'*50}")
            log_func(f"НЕЙРОН [{layer}][{neuron}]")
            log_func(f"{'-'*50}")
        
        current_weights = weights.get((layer, neuron), [])
        signals = input_signals.get((layer, neuron), [])
        current_bias = biases.get((layer, neuron), 1.0)  # значение по умолчанию
        
        new_neuron_weights = []
        for i, (w, signal) in enumerate(zip(current_weights, signals)):
            if log_func:
                log_func(f"\nСинапс {i+1}:")
            new_w = calculate_new_weight(w, learning_rate, error, signal, log_func)
            new_neuron_weights.append(new_w)
        new_weights[(layer, neuron)] = new_neuron_weights
        
        new_b = calculate_new_bias(current_bias, learning_rate, error, log_func)
        new_biases[(layer, neuron)] = new_b
    
    if log_func:
        log_func("\n" + "="*50)
        log_func("РАСЧЕТ НОВЫХ ВЕСОВ И СМЕЩЕНИЙ ЗАВЕРШЕН")
        log_func("="*50)
    
    return new_weights, new_biases
