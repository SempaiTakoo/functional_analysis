import numpy as np
from scipy.integrate import quad


def chi(x):
    return 1 if x >= 0 else 0

def f(x):
    return np.sin(8 * x) + 4 * chi(2 * x - 9) - 3 * x**2

def F_prime(x):
    return np.exp(x) + 2 * abs(x)


if __name__ == '__main__':
    I1, _ = quad(lambda x: f(x) * F_prime(x), -16, 1)
    I2, _ = quad(lambda x: f(x) * F_prime(x), 1, 4)
    I3, _ = quad(lambda x: f(x) * F_prime(x), 4, 45)

    f1 = f(1)
    f4 = f(4)

    D1 = f1 * 3
    D2 = f4 * 1

    result = I1 + I2 + I3 + D1 + D2

    print(
        'Интеграл Лебега–Стилтьеса: \n'
        f'{result:.6f}'
    )
