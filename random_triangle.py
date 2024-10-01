from PIL import Image, ImageDraw
import random

# Создаем изображение и объект для рисования
img = Image.new('RGB', (7, 7), color='white')
draw = ImageDraw.Draw(img)

# Высота треугольника (количество заполненных строк)
triangle_height = 5

# Начальная координата x для вершины треугольника
start_x = 3

# Рисуем контур равнобедренного треугольника
for y in range(triangle_height):
    draw.point((start_x - y, y), fill='black')  # левая сторона
    draw.point((start_x + y, y), fill='black')  # правая сторона
    draw.point((start_x + y, y), fill='black')
    for x in range (7):
        if y == 4:
            draw.point((x, y), fill='black')
for i in range (4):
    draw.point((random.randint(0, 7), random.randint(0, 7)), fill=random.choice(["white","black"]))

# Сохраняем изображение и выводим его
img.save('isosceles_triangle_outline.png')
img.show()