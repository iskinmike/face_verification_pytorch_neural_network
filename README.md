# Нейросеть на PyTorch - классификатор для распознавания лиц

Заточена под классификацию векторов.

Это не готовый продукт а тестовые программы, поэтому ни о какой универсальности речи не идет.
Все пути захардкожены. Поэтому прежде чем запускать стоит посмотреть код.

Тут главный файл это test.py. Тут и обучение и классификация. Загрузка данных отдельно в load_test.py

init.py - просто тест взятый и измененный из доукментации PyTorch
tmp.py - тест для работы с YAML файлами в которые вектора сохраняет opencv. 
Это узкое место так как работает медленно.
