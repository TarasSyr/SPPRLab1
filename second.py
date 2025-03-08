import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from matplotlib.patches import Polygon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt


class LinearProgramming:
    def __init__(self):
        self.resources = {
            'Пластмаса': {'x1': 0.065, 'x2': 0.05, 'Ліміт': 41},
            'Дріт': {'x1': 0.045, 'x2': 0.087, 'Ліміт': 45},
            'Фарба': {'x1': 0.068, 'x2': 0.035, 'Ліміт': 39}
        }

        self.costs = {
            'Компухтер': 430,
            'Принтер': 680
        }

    def solve_lp(self):
        c = [-self.costs['Компухтер'], -self.costs['Принтер']]
        A = [
            [self.resources['Пластмаса']['x1'], self.resources['Пластмаса']['x2']],
            [self.resources['Дріт']['x1'], self.resources['Дріт']['x2']],
            [self.resources['Фарба']['x1'], self.resources['Фарба']['x2']]
        ]
        b = [
            self.resources['Пластмаса']['Ліміт'],
            self.resources['Дріт']['Ліміт'],
            self.resources['Фарба']['Ліміт']
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

        # Функція для перевірки, чи точка задовольняє всім обмеженням
        def is_feasible(x1, x2):
            return (
                0.065 * x1 + 0.05 * x2 <= 41 and
                0.045 * x1 + 0.087 * x2 <= 45 and
                0.068 * x1 + 0.035 * x2 <= 39 and
                x1 >= 0 and x2 >= 0
            )

        # Перетин y1 та y2
        A = np.array([[0.065, 0.05], [0.045, 0.087]])
        b = np.array([41, 45])
        intersection = np.linalg.solve(A, b)
        if is_feasible(intersection[0], intersection[1]):
            vertices.append(intersection)

        # Перетин y1 та y3
        A = np.array([[0.065, 0.05], [0.068, 0.035]])
        b = np.array([41, 39])
        intersection = np.linalg.solve(A, b)
        if is_feasible(intersection[0], intersection[1]):
            vertices.append(intersection)

        # Перетин y2 та y3
        A = np.array([[0.045, 0.087], [0.068, 0.035]])
        b = np.array([45, 39])
        intersection = np.linalg.solve(A, b)
        if is_feasible(intersection[0], intersection[1]):
            vertices.append(intersection)

        # Додамо точки на осях
        if is_feasible(0, 0):
            vertices.append([0, 0])
        if is_feasible(0, min(41 / 0.05, 45 / 0.087, 39 / 0.035)):
            vertices.append([0, min(41 / 0.05, 45 / 0.087, 39 / 0.035)])
        if is_feasible(min(41 / 0.065, 45 / 0.045, 39 / 0.068), 0):
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


class PyQtWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Лінійне програмування")
        self.setGeometry(100, 100, 800, 600)

        # Головний віджет
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Основний layout
        layout = QVBoxLayout()

        # Таблиця з вхідними даними
        self.table = QTableWidget()
        self.table.setRowCount(4)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Ресурс", "x1(Комп'ютер)", "x2(Принтер)", "Ліміт"])
        #self.table.setVerticalHeaderLabels(["Пластмаса", "Дріт", "Фарба"])

        # Заповнення таблиці даними
        self.table.setItem(0, 0, QTableWidgetItem("Пластмаса"))
        self.table.setItem(0, 1, QTableWidgetItem("0.065"))
        self.table.setItem(0, 2, QTableWidgetItem("0.05"))
        self.table.setItem(0, 3, QTableWidgetItem("41"))

        self.table.setItem(1, 0, QTableWidgetItem("Дріт"))
        self.table.setItem(1, 1, QTableWidgetItem("0.045"))
        self.table.setItem(1, 2, QTableWidgetItem("0.087"))
        self.table.setItem(1, 3, QTableWidgetItem("45"))

        self.table.setItem(2, 0, QTableWidgetItem("Фарба"))
        self.table.setItem(2, 1, QTableWidgetItem("0.068"))
        self.table.setItem(2, 2, QTableWidgetItem("0.035"))
        self.table.setItem(2, 3, QTableWidgetItem("39"))

        self.table.setItem(3, 0, QTableWidgetItem("Ціна за од"))
        self.table.setItem(3, 1, QTableWidgetItem("430 грн"))
        self.table.setItem(3, 2, QTableWidgetItem("680 грн"))


        # Кнопка для розрахунку
        self.calculate_button = QPushButton("Розрахувати")
        self.calculate_button.clicked.connect(self.calculate)

        # Текстове поле для виведення ходу розв'язання
        self.solution_text = QTextEdit()
        self.solution_text.setReadOnly(True)

        # Додавання елементів до layout
        layout.addWidget(self.table)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.solution_text)

        self.central_widget.setLayout(layout)

    def calculate(self):
        # Створення об'єкта LinearProgramming
        lp = LinearProgramming()

        # Розв'язання задачі
        res = lp.solve_lp()

        # Виведення ходу розв'язання
        solution_text = "Хід розв'язання:\n\n"
        solution_text += f"1. Вхідні дані:\n"
        solution_text += f"    - Ресурси: {lp.resources}\n"
        solution_text += f"    - Вартість: {lp.costs}\n\n"
        solution_text += (f"2. Вектор напряму: значення вектора отримані знаходження часткових похідних цього рівняння: 430x1+680x2. \n"
                          f"    - Відповідно похідна від 430x1=430; похідна від 680x2=680. Отже цільовий вектор буде: с=(430; 680).\n\n")
        solution_text += f"3. Оптимальні значення змінних:\n"
        solution_text += f"    - x1 (Комп'ютери): {res.x[0]:.2f}\n"
        solution_text += f"    - x2 (Принтери): {res.x[1]:.2f}\n\n"
        solution_text += f"4. Максимальний прибуток: {-res.fun:.2f} грн.\n"

        self.solution_text.setText(solution_text)

        # Відображення графіка
        lp.plot_graph(res.x[0], res.x[1])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyQtWindow()
    window.show()
    sys.exit(app.exec_())