import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from matplotlib.patches import Polygon


class LinearProgramming:
    def __init__(self):
        self.resources = {
            'Plastic': {'x1': 0.065, 'x2': 0.05, 'limit': 41},
            'Wire': {'x1': 0.045, 'x2': 0.087, 'limit': 45},
            'Paint': {'x1': 0.068, 'x2': 0.035, 'limit': 39}
        }

        self.costs = {
            'Computer': 430,
            'Printer': 680
        }

    def solve_lp(self):
        c = [-self.costs['Computer'], -self.costs['Printer']]
        A = [
            [self.resources['Plastic']['x1'], self.resources['Plastic']['x2']],
            [self.resources['Wire']['x1'], self.resources['Wire']['x2']],
            [self.resources['Paint']['x1'], self.resources['Paint']['x2']]
        ]
        b = [
            self.resources['Plastic']['limit'],
            self.resources['Wire']['limit'],
            self.resources['Paint']['limit']
        ]
        bounds = [(0, None), (0, None)]

        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        return res

    def plot_graph(self, x1_opt, x2_opt):
        fig, ax = plt.subplots()

        x = np.linspace(-100, 1000, 200)

        # Обмеження
        y1 = (41 - 0.065 * x) / 0.05
        y2 = (45 - 0.045 * x) / 0.087
        y3 = (39 - 0.068 * x) / 0.035

        # Лінії обмежень
        ax.plot(x, y1, label="0.065x1 + 0.05x2 ≤ 41", color='blue')
        ax.plot(x, y2, label="0.045x1 + 0.087x2 ≤ 45", color='green')
        ax.plot(x, y3, label="0.068x1 + 0.035x2 ≤ 39", color='red')

        # Вектор напряму (починається з (0,0))
        ax.quiver(0, 0, 430, 680, angles='xy', scale_units='xy', scale=1, color='purple', label="Вектор напрямку")

        # Перпендикулярні прямі
        x_perp = np.linspace(-100, 1000, 200)

        # Кутовий коефіцієнт перпендикулярної прямої
        k_perp = -430 / 680  # Перпендикулярний кутовий коефіцієнт

        # Пряма через (0,0)
        y_perp1 = k_perp * x_perp
        ax.plot(x_perp, y_perp1, 'k--', label="Перпендикуляр через (0,0)")

        # Пряма через оптимальну точку
        C_opt = x2_opt - k_perp * x1_opt  # Константа для прямої через оптимальну точку
        y_perp2 = k_perp * x_perp + C_opt
        ax.plot(x_perp, y_perp2, 'k--', label="Перпендикуляр через оптимум")

        # Оптимальна точка
        ax.scatter(x1_opt, x2_opt, color='black', label=f"Оптимум: ({x1_opt:.2f}, {x2_opt:.2f})", zorder=5)

        # Лінії осей (x=0 та y=0)
        ax.axhline(0, color='black', linewidth=1, linestyle='-')  # Вісь x
        ax.axvline(0, color='black', linewidth=1, linestyle='-')  # Вісь y

        # Заштрихована область допустимих значень
        # Визначимо вершини області допустимих значень
        vertices = []
        # Перетин y1 та y2
        A = np.array([[0.065, 0.05], [0.045, 0.087]])
        b = np.array([41, 45])
        intersection = np.linalg.solve(A, b)
        if intersection[0] >= 0 and intersection[1] >= 0:
            vertices.append(intersection)

        # Перетин y1 та y3
        A = np.array([[0.065, 0.05], [0.068, 0.035]])
        b = np.array([41, 39])
        intersection = np.linalg.solve(A, b)
        if intersection[0] >= 0 and intersection[1] >= 0:
            vertices.append(intersection)

        # Перетин y2 та y3
        A = np.array([[0.045, 0.087], [0.068, 0.035]])
        b = np.array([45, 39])
        intersection = np.linalg.solve(A, b)
        if intersection[0] >= 0 and intersection[1] >= 0:
            vertices.append(intersection)

        # Додамо точки на осях
        vertices.append([0, 0])
        vertices.append([0, min(41 / 0.05, 45 / 0.087, 39 / 0.035)])
        vertices.append([min(41 / 0.065, 45 / 0.045, 39 / 0.068), 0])

        # Відсортуємо вершини за годинниковою стрілкою
        from scipy.spatial import ConvexHull
        hull = ConvexHull(vertices)
        ordered_vertices = [vertices[i] for i in hull.vertices]

        # Заштрихуємо область
        ax.add_patch(Polygon(ordered_vertices, closed=True, color='gray', alpha=0.3, label="Область допустимих значень"))

        # Налаштування графіка
        ax.set_xlim([-100, 1000])
        ax.set_ylim([-100, 1000])
        ax.set_aspect('equal')
        ax.legend()
        plt.grid(True)
        ax.set_xlabel("Кількість комп'ютерів")
        ax.set_ylabel("Кількість принтерів")
        plt.show()


# Виконання розрахунку
lp = LinearProgramming()
res = lp.solve_lp()
lp.plot_graph(res.x[0], res.x[1])