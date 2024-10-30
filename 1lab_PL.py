# ---------------------------------
# Изначальная грамматика

# S в†’ CACB
# C в†’ Cc | Cb | Ca | a
# A в†’ baB | bbC | b
# B в†’ cB | c

# ---------------------------------
# LL(1) грамматика

# A',B',C' - обозначается как At,Bt,Ct
# S -> a St
# A -> a At
# B -> b Bt
# C -> c Ct
# St -> b A
#     | c A B
#     | a C B
# At -> a At
#     | ""
# Bt -> b Bt
#     | ""
# Ct -> c Ct
#     | ""

# ---------------------------------
# Примеры

# Input     | Expected | Output
# aaa       | False    | False
# aba       | True     | True
# acaabb    | True     | True
# aacccb    | True     | True
# abc       | False    | False
# ---------------------------------

# Словарь с правилами из таблицы анализатора
ll1_table = {
    "S": {
        "a": ["a", "St"]
    },
    "St": {
        "a": ["a", "C", "B"],
        "b": ["b", "A"],
        "c": ["c", "A", "B"]
    },
    "A": {
        "a": ["a", "At"]
    },
    "B": {
        "b": ["b", "Bt"]
    },
    "C": {
        "c": ["c", "Ct"]
    },
    "At": {
        "a": ["a", "At"],
        "b": [""],
        "$": [""]
    },
    "Bt": {
        "b": ["b", "Bt"],
        "$": [""]
    },
    "Ct": {
        "b": [""],
        "c": ["c", "Ct"]
    }
}

# Начальное правило S
stack = ["S"]


def apply(string):
    print(string, "состояние стека --->", stack)

    if len(string) == 1 and string[0] == "$" and len(stack) == 0:  # Если стек пуст и в строке остался только символ $, то разбор завершен
        return True

    stack_top = stack.pop() # Записываем в переменную верхний элемент стека
    print(stack_top)

    if stack_top == "":  # Если верхний элемент стека равен "", то пропускаем это правило
        return apply(string)

    if stack_top[0].isupper():  # Если это нетерминал
        if stack_top not in ll1_table:
            return False

        if string[0] not in ll1_table[stack_top]: # Правило для перехода из текущей ситуации не нашлось
            return False

        for i in range(1, len(ll1_table[stack_top][string[0]]) + 1, 1):  # Добавляем в стек правую часть правила в обратном порядке
            stack.append(ll1_table[stack_top][string[0]][-i])

        return apply(string)
    else: # Если это терминал

        if stack_top == string[0]:  # Верхний элемент стека совпал с символом строки
            return apply(string[1:])
        else:  # Верхний элемент стека не совпал с символом строки, цепочка не подходит
            return False


print(apply(input("Введите цепочку: ") + "$"))