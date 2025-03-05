from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QFileDialog, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
import pandas as pd
import numpy as np
import sys

class WSMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('WSM Decision Support System')
        self.setGeometry(100, 100, 900, 700)
        self.center()

        # Головний віджет та макет
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Таблиця для вхідних даних
        self.data_table = QTableWidget(self)
        self.layout.addWidget(QLabel("Вхідні дані"))
        self.layout.addWidget(self.data_table)

        # Таблиця для результатів
        self.result_table = QTableWidget(self)
        self.layout.addWidget(QLabel("Результати"))
        self.layout.addWidget(self.result_table)

        # Текстове повідомлення про найкращу альтернативу
        self.best_alt_label = QLabel(self)
        self.layout.addWidget(self.best_alt_label)

        # Меню
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('Файл')
        open_action = QAction('Відкрити', self)
        open_action.triggered.connect(self.openFile)
        file_menu.addAction(open_action)

        self.show()

    def center(self):
        screen_geometry = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.move(window_geometry.topLeft())

    def openFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Відкрити файл', '', 'CSV файли (*.csv)')
        if fname:
            try:
                data = pd.read_csv(fname)

                if data.shape[0] < 2:  # Перевірка, чи є достатньо рядків
                    self.result_table.setRowCount(0)
                    self.data_table.setRowCount(0)
                    return

                # Заповнюємо першу таблицю вхідними даними
                self.populateTable(self.data_table, data)

                # Обчислення WSM
                criteria_weights = data.iloc[-1, 1:].values.astype(float)  # Останній рядок як ваги
                alternatives = data.iloc[:-1, 1:].values.astype(float)  # Всі інші рядки як альтернативи
                alt_names = data.iloc[:-1, 0].astype(str).values  # Назви альтернатив

                scores = self.wsm(criteria_weights, alternatives)
                best_alt_index = np.argmax(scores)  # Найкраща альтернатива за індексом
                best_alt = alt_names[best_alt_index]  # Найкраща альтернатива

                # Підготовка результатів у DataFrame
                result_data = pd.DataFrame({
                    'Альтернатива': alt_names,
                    'Оцінка': scores
                })

                result_data.loc[len(result_data)] = ['Найкраща альтернатива', best_alt]  # Додаємо рядок для найкращої альтернативи

                # Заповнюємо другу таблицю результатами
                self.populateTable(self.result_table, result_data)

                # Виділяємо найкращу альтернативу кольором
                self.highlightBestAlternative(self.result_table, best_alt_index)

                # Виведення текстового повідомлення про найкращу альтернативу
                self.best_alt_label.setText(f"Найкраща альтернатива: {best_alt}")

            except Exception as e:
                print(f"Помилка при обробці файлу: {e}")

    def populateTable(self, table, data):
        table.clear()
        table.setRowCount(data.shape[0])
        table.setColumnCount(data.shape[1])
        table.setHorizontalHeaderLabels(data.columns.astype(str))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

    def wsm(self, criteria_weights, alternatives):
        return np.dot(alternatives, criteria_weights)  # Рахуємо загальну оцінку для кожної альтернативи

    def highlightBestAlternative(self, table, best_alt_index):
        # Виділяємо найкращу альтернативу кольором
        for column in range(table.columnCount()):
            item = table.item(best_alt_index, column)
            if item:
                item.setBackground(Qt.green)  # Зелений колір для найкращої альтернативи

if __name__ == '__main__':
    from PyQt5.QtCore import Qt
    app = QApplication(sys.argv)
    ex = WSMApp()
    sys.exit(app.exec_())
