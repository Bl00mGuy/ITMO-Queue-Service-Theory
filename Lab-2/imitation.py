import numpy as np
import matplotlib.pyplot as plt
import math

def mmn_queue_metrics(lam, mu, n):
    """
    Расчет характеристик многоканальной системы M/M/n.

    lam : интенсивность входящего потока заявок (λ)
    mu  : интенсивность обслуживания на один канал (μ)
    n   : количество каналов
    """
    # Коэффициент загрузки системы
    rho = lam / (n * mu)
    if rho >= 1:
        print("Система неустойчива (ρ >= 1)")
        return None
    # Вычисление P0
    sum_terms = sum([(lam/mu)**k / math.factorial(k) for k in range(n)])
    term_n = (lam/mu)**n / (math.factorial(n) * (1 - rho))
    P0 = 1.0 / (sum_terms + term_n)
    # Вероятность того, что заявка будет ждать в очереди
    P_wait = term_n * P0
    # Среднее число заявок в очереди
    Lq = (P_wait * rho) / (1 - rho)
    # Среднее время ожидания заявки в очереди
    Wq = Lq / lam
    # Среднее время пребывания заявки в системе
    W = Wq + 1/mu
    return P0, P_wait, Lq, Wq, W, rho

# Исходные параметры
lam = 10  # интенсивность входящего потока заявок (заявок/час)
mu = 3    # интенсивность обслуживания (заявок/час на канал)
n = 4     # количество каналов

# Расчет характеристик системы
results = mmn_queue_metrics(lam, mu, n)
if results:
    P0, P_wait, Lq, Wq, W, rho = results
    print("Основные характеристики системы:")
    print(f"P0 (вероятность простоя)           = {P0:.4f}")
    print(f"P(wait) (вероятность ожидания)       = {P_wait:.4f}")
    print(f"Lq (среднее число заявок в очереди)   = {Lq:.4f}")
    print(f"Wq (среднее время ожидания, ч)       = {Wq:.4f}")
    print(f"W  (среднее время в системе, ч)      = {W:.4f}")
    print(f"Коэффициент загрузки (ρ)             = {rho:.4f}")

# Построение графиков зависимости Wq и Lq от количества каналов n
n_values = np.arange(1, 11)
Wq_values = []
Lq_values = []
for n_val in n_values:
    metrics = mmn_queue_metrics(lam, mu, n_val)
    if metrics is not None:
        _, P_wait_val, Lq_val, Wq_val, _, _ = metrics
        Wq_values.append(Wq_val)
        Lq_values.append(Lq_val)
    else:
        Wq_values.append(np.nan)
        Lq_values.append(np.nan)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(n_values, Wq_values, marker='o', linestyle='-', color='blue')
plt.xlabel("Количество каналов (n)")
plt.ylabel("Среднее время ожидания (Wq, ч)")
plt.title("Зависимость Wq от количества каналов")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(n_values, Lq_values, marker='o', linestyle='-', color='orange')
plt.xlabel("Количество каналов (n)")
plt.ylabel("Среднее число заявок в очереди (Lq)")
plt.title("Зависимость Lq от количества каналов")
plt.grid(True)

plt.tight_layout()
plt.show()

# Анализ влияния интенсивности обслуживания mu (при n = 4)
mu_values = np.linspace(2, 6, 20)
Wq_mu_values = []
Lq_mu_values = []
for mu_val in mu_values:
    metrics = mmn_queue_metrics(lam, mu_val, n)
    if metrics is not None:
        _, P_wait_val, Lq_val, Wq_val, _, _ = metrics
        Wq_mu_values.append(Wq_val)
        Lq_mu_values.append(Lq_val)
    else:
        Wq_mu_values.append(np.nan)
        Lq_mu_values.append(np.nan)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(mu_values, Wq_mu_values, marker='o', linestyle='-', color='green')
plt.xlabel("Интенсивность обслуживания (μ)")
plt.ylabel("Среднее время ожидания (Wq, ч)")
plt.title("Зависимость Wq от μ")
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(mu_values, Lq_mu_values, marker='o', linestyle='-', color='red')
plt.xlabel("Интенсивность обслуживания (μ)")
plt.ylabel("Среднее число заявок в очереди (Lq)")
plt.title("Зависимость Lq от μ")
plt.grid(True)

plt.tight_layout()
plt.show()
