import numpy as np
import matplotlib.pyplot as plt
import pickle

# Функция для сохранения результатов в файл
def save_results(filename, results):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

# Функция для загрузки результатов из файла
def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Решатель ОДУ методом Хёйна
def solve_ode(start, end, step, max_calls, tolerance, fs, initial_conditions):
    t = start
    v = np.array(initial_conditions)
    call_counter = [0]
    steps = []
    solutions = []
    coord = []

    def heun_step(t, v, h):
        k1 = fs(t, v, call_counter)
        k2 = fs(t + h, v + h * k1, call_counter)
        return v + (h / 2) * (k1 + k2)

    while t < end and call_counter[0] < max_calls:
        k1 = fs(t, v, call_counter)
        k2 = fs(t + step, v + step * k1, call_counter)
        v1 = v + (step / 2) * (k1 + k2)

        k2 = fs(t + step / 2, v + step / 2 * k1, call_counter)
        v2 = v + (step / 4) * (k1 + k2)

        v2 = heun_step(t + step / 2, v2, step / 2)

        error = np.linalg.norm(v2 - v1) / 3
        if error > tolerance:
            step /= 2
        elif error < tolerance / 64:
            step *= 2

        if error < tolerance:
            if t + step > end:
                step = end - t
            t += step
            v = v1
            steps.append(step)
            solutions.append(v.copy())
            coord.append(t)

    return steps, solutions, coord, len(steps), min(steps)

# Основная функция для запуска расчетов
def main():
    # Список значений заданной точности
    eps_count = [0.01, 0.001, 0.0001, 0.00001]

    # Считать входные данные
    t_0 = float(input())
    T = float(input())
    h_0 = float(input())
    N_x = int(input())
    eps = float(input())
    n = int(input())

    # Считать код функции-правой части системы ОДУ
    function_code = []
    for _ in range(n + 3):
        line = input()
        function_code.append(line)

    # Определить функцию-правую часть системы ОДУ
    function_definition = "\n".join(function_code)
    exec(function_definition, globals())

    # Проверка, что функция fs была определена
    if not callable(globals().get('fs')):
        print("Функция fs не была определена.")
        return

    # Считать начальные условия
    initial_conditions_str = input()
    initial_conditions = [float(x) for x in initial_conditions_str.split()]

    # Решить систему ОДУ для каждого значения eps и сохранить результаты
    for eps in eps_count:
        steps, solutions, coord, num_steps, min_step = solve_ode(t_0, T, h_0, N_x, eps, fs, initial_conditions)
        save_results(f'results_eps_{eps}.pkl', (steps, solutions, coord, num_steps, min_step))

    # Построить графики используя сохраненные результаты
    min_steps = []
    num_steps_list = []

    fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
    for i, eps in enumerate(eps_count):
        steps, solutions, coord, num_steps_eps, min_step = load_results(f'results_eps_{eps}.pkl')
        axes[i].plot(coord, steps)
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("h")
        axes[i].set_title(f"Изменение шага по отрезку для разных значений заданной точности, eps={eps}")
        min_steps.append(min_step)
        num_steps_list.append(num_steps_eps)

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(eps_count, min_steps)
    ax.set_xlabel('eps')
    ax.set_ylabel('Минимальный шаг')
    ax.set_title('Зависимость минимального шага от заданной точности')
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogx(eps_count, num_steps_list)
    ax.set_xlabel('eps')
    ax.set_ylabel('Количество шагов')
    ax.set_title('Зависимость числа шагов от заданной точности')
    plt.show()

    fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
    for i, eps in enumerate(eps_count):
        steps, solutions, coord, _, _ = load_results(f'results_eps_{eps}.pkl')
        solutions = np.array(solutions)
        for j in range(solutions.shape[1]):
            axes[i].plot(coord, solutions[:, j], label=f'v{j+1}')
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("Значение решения")
        axes[i].set_title(f"Изменение решения, eps={eps}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# Запуск основной функции
if __name__ == "__main__":
    main()