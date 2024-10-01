from PIL import Image, ImageDraw
import random

# Создаем изображение и объект для рисования
img = Image.new('RGB', (7, 7), color='white')
draw = ImageDraw.Draw(img)

# Выбираем случайные координаты для начала контура
start_x = random.randint(0, 2)  # случайная начальная координата x
start_y = random.randint(0, 2)  # случайная начальная координата y

# Генерируем контур квадрата
for x in range(start_x, start_x + 5):
    for y in range(start_y, start_y + 5):
        if x == start_x or x == start_x + 4 or y == start_y or y == start_y + 4:
            draw.point((x, y), fill='black')
for i in range (4):
    draw.point((random.randint(0, 7), random.randint(0, 7)), fill=random.choice(["white","black"]))

# Сохраняем изображение и выводим его
img.save('random_square_position.png')
img.show()