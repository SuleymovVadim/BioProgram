import numpy as np
import matplotlib.pyplot as plt

# Решатель ОДУ методом Хёйна
def solve_ode(start, end, step, max_calls, tolerance, fs, initial_conditions):
    t = start
    v = np.array(initial_conditions)
    call_counter = [0]
    steps = []
    solutions = []
    coord = []

    print(f"{t:13.6f}{step:13.6f}{0:13d}{0:13d}", *[f"{x:12.6f}" for x in v])

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
            print(f"{t:13.6f} {step:13.6f} {error:13.5e} {call_counter[0]:13d}", *[f"{x:12.6f}" for x in v])

    return steps, solutions, coord, len(steps), min(steps)

# Функция-правая часть системы ОДУ
def fs(t, v, kounter):
    A = np.array([[-0.4, 0.02, 0], [0, 0.8, -0.1], [0.003, 0, 1]])
    kounter[0] += 1
    return np.dot(A, v)

# Жестко заданные входные данные
t_0 = 1.5
T = 2.5
h_0 = 0.1
N_x = 10000
eps = 0.0001
initial_conditions = [1, 1, 2]

# Решить систему ОДУ
eps_count = [0.01, 0.001, 0.0001, 0.00001]
results_list = []

for eps in eps_count:
    steps, solutions, coord, num_steps, min_step = solve_ode(t_0, T, h_0, N_x, eps, fs, initial_conditions)
    results_list.append((steps, solutions, coord, num_steps, min_step))

# Построить графики
fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
min_steps = []
num_steps = []

for i, (steps, solutions, coord, num_steps_eps, min_step) in enumerate(results_list):
    axes[i].plot(coord, steps)
    axes[i].set_xlabel("t")
    axes[i].set_ylabel("h")
    axes[i].set_title(f"Изменение шага по отрезку для разных значений заданной точности, eps={eps_count[i]}")
    min_steps.append(min_step)
    num_steps.append(num_steps_eps)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogx(eps_count, min_steps)
ax.set_xlabel('eps')
ax.set_ylabel('Минимальный шаг')
ax.set_title('Зависимость минимального шага от заданной точности')
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogx(eps_count, num_steps)
ax.set_xlabel('eps')
ax.set_ylabel('Количество шагов')
ax.set_title('Зависимость числа шагов от заданной точности')
plt.show()

fig, axes = plt.subplots(len(eps_count), 1, figsize=(8, 8))
for i, (steps, solutions, coord, _, _) in enumerate(results_list):
    solutions = np.array(solutions)
    for j in range(solutions.shape[1]):
        axes[i].plot(coord, solutions[:, j], label=f'v{j+1}')
    axes[i].set_xlabel("t")
    axes[i].set_ylabel("Значение решения")
    axes[i].set_title(f"Изменение решения, eps={eps_count[i]}")
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()