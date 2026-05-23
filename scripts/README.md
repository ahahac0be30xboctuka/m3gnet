# Скрипты экспериментов

Четыре эксперимента на предобученном чекпоинте `M3GNet-MP-2021.2.8-EFS` (тот же, что в статье). Скрипты не зависят друг от друга, запускаются в любом порядке. Каждый сохраняет рядом с собой JSON с результатами.

## Окружение (macOS arm64, Apple Silicon)

```bash
conda create --name m3gnet-tf python=3.10 -y
conda activate m3gnet-tf
conda install -c apple tensorflow-deps -y
pip install tensorflow-macos==2.13.0 tensorflow-metal
pip install --no-deps m3gnet
pip install pymatgen==2024.2.20 ase==3.22.1 sympy==1.12
pip install "numpy<1.24" "typing-extensions<4.6"
```

`tensorflow-metal` опционален (GPU-ускорение через Apple Metal). Для Linux/Windows нужно ставить обычный `tensorflow==2.13.0` вместо `tensorflow-macos`.

## Чекпоинт

По умолчанию `m3gnet.M3GNet.load("MP-2021.2.8-EFS")` сам скачивает веса из апстрим-репозитория [materialsvirtuallab/m3gnet](https://github.com/materialsvirtuallab/m3gnet/tree/main/pretrained/MP-2021.2.8-EFS). Если нужно явно указать локальный путь:

```bash
export M3GNET_CHECKPOINT=/path/to/MP-2021.2.8-EFS
```

## Запуск

```bash
conda activate m3gnet-tf
python expA_predict.py        # ~5 c       
python expB_relax.py          # ~45 c      
python expC_classes.py        # ~30 c      
python expD_prerelax.py       # ~3 мин     
```

## Что считается

- **`expA_predict.py`** - энергии и силы для Si, Cu, Ni, LiF, Mo в их MP-равновесных параметрах решётки. Силы при равновесии должны быть ~$10^{-7}$ эВ/Å (численный ноль).
- **`expB_relax.py`** - релаксация Mo BCC с растянутой решёткой $a=3.3$ Å (MP target 3.16762 Å) через FIRE + ExpCellFilter. Второй тест — $2\times2\times2$ supercell с возмущениями 0.1 Å, проверка восстановления BCC-симметрии.
- **`expC_classes.py`** - тест гипотезы H1 «модель плохо работает для ковалентных/ионных кристаллов». 12 кубических кристаллов в 3 классах (металлы / ковалентные / ионные), сравнение средних $|\Delta a|/a$.
- **`expD_prerelax.py`** - тест гипотезы H2 «пре-релаксация уменьшает ошибку энергии не менее чем в 10 раз». 5 металлов с решёткой $\pm 5\%$ от равновесия: прямое предсказание vs предсказание после релаксации.
