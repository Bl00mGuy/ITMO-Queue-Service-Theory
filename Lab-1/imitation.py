import simpy
import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
LAMBDA = 5       # интенсивность поступления заявок (заявок/ед. времени)
MU = 6           # интенсивность обслуживания (заявок/ед. времени)
SIM_TIME = 1000  # время моделирования


class CallCenter:
    def __init__(self, env):
        self.env = env
        self.server = simpy.Resource(env, capacity=1)
        self.busy_time = 0.0


def serve(env, call_center, service_time):
    # Запрос сервера и фиксация времени начала обслуживания
    with call_center.server.request() as request:
        yield request
        start_time = env.now
        yield env.timeout(service_time)
        # Добавляем время обслуживания к общей занятости
        call_center.busy_time += env.now - start_time


def arrival_process(env, call_center, lambd, mu, stats):
    while True:
        # Генерация интервала между поступлениями (экспоненциальное распределение)
        interarrival = np.random.exponential(1.0 / lambd)
        yield env.timeout(interarrival)
        stats['arrivals'] += 1

        # Если сервер свободен, обрабатываем заявку; иначе фиксируем отказ
        if call_center.server.count < call_center.server.capacity:
            stats['served'] += 1
            service_time = np.random.exponential(1.0 / mu)
            env.process(serve(env, call_center, service_time))
        else:
            stats['lost'] += 1


def run_simulation(lambd, mu, sim_time):
    env = simpy.Environment()
    call_center = CallCenter(env)
    stats = {'arrivals': 0, 'served': 0, 'lost': 0}
    env.process(arrival_process(env, call_center, lambd, mu, stats))
    env.run(until=sim_time)

    # Вычисление коэффициента загрузки и вероятности отказа
    utilization = call_center.busy_time / sim_time
    loss_probability = stats['lost'] / stats['arrivals'] if stats['arrivals'] > 0 else 0
    return stats, utilization, loss_probability


# Проведение эксперимента для исходных параметров
stats, utilization, loss_probability = run_simulation(LAMBDA, MU, SIM_TIME)
print("Статистика при λ =", LAMBDA, "и μ =", MU)
print("Поступило заявок:", stats['arrivals'])
print("Обслужено заявок:", stats['served'])
print("Отказов:", stats['lost'])
print("Коэффициент загрузки:", utilization)
print("Вероятность отказа:", loss_probability)

# Теоретическая вероятность отказа по формуле Эрланга: λ / (λ + μ)
theoretical_loss = LAMBDA / (LAMBDA + MU)
print("Теоретическая вероятность отказа:", theoretical_loss)

# Построение графика зависимости вероятности отказа от интенсивности входящего потока (λ)
lambdas = np.linspace(1, 10, 50)  # диапазон изменения λ от 1 до 10
exp_loss_probs = []
theor_loss_probs = []

for l in lambdas:
    stats, util, loss_prob = run_simulation(l, MU, SIM_TIME)
    exp_loss_probs.append(loss_prob)
    theor_loss_probs.append(l / (l + MU))

plt.style.use('dark_background')
plt.figure(figsize=(8, 5))
plt.plot(lambdas, exp_loss_probs, label='Экспериментальное')
plt.plot(lambdas, theor_loss_probs, label='Теоретическое', linestyle='--')
plt.xlabel('Интенсивность входящего потока (λ)')
plt.ylabel('Вероятность отказа')
plt.title('Зависимость вероятности отказа от λ при μ = ' + str(MU))
plt.legend()
plt.grid(True)
plt.show()
