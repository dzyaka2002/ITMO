В рамках одной из дисциплин было поручено разработать функциональную модель работы нейронной сети, решающей задачу распознавания трех геометрических фигур на изображении с шумом.
Важное условие - нельзя использовать сторонние библиотеки для разработки логики.
- Файлы random_* - эксперименты с библиотекой Pillow
- В результате, остановилась на варианте в файле create_files.py: изображение сделано в виде 0 и 1, и к идеальной фигуре добавляются 5 "неверных" значений.

Итоговая модель нейронной сети - **nn_last_v.py**
