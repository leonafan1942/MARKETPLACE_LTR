# Проект "Ранжирование в поиске"

Поиск товаров в онлайн магазине состоит из нескольких этапов. Первый этап по ключевым характеристикам находятся наиболее подходящие кандидаты. Далее отбирается более узкий набор товаров, исходя из наличия на складе, локации и пр. Следующий этап - ранжирование отобранных товаров. Задача состоит в разработке системы оптимизации ранжирования предпоследнего этапа.

**Целью работы** является создание модели и Docker образа

Решение состоит из 4 этапов:

1. Формализации задачи и анализа EDA
2. Проработка вариантов решения
3. Baseline
4. Оптимизация решения

## Синтетические данные предоставлены Wildberries

## Описание рабочих файлов и директорий:
- [ListNet.ipynb](https://github.com/leonafan1942/MARKETPLACE_LTR/blob/main/ListNet.ipynb) - рабочая тетрадь с реализацией RankNet, ListNet моделей (этап 4)
- [LTR_EDA.ipynb](https://github.com/leonafan1942/MARKETPLACE_LTR/blob/main/LTR_EDA.ipynb) - рабочая тетрать с EDA анализом (этапы 1-3)
- [requirements.txt](https://github.com/leonafan1942/MARKETPLACE_LTR/blob/main/requirements.txt) - список необходимых пакетов для Docker образа
- [train.py](https://github.com/leonafan1942/MARKETPLACE_LTR/blob/main/train.py) - код с реализацией training of ListNet
- [inference.py](https://github.com/leonafan1942/MARKETPLACE_LTR/blob/main/inference.py) - inference для Docker образа

## Используемые инструменты
- python: polars, pytorch, numpy, matplotlib, seaborn, phik, scikit-learn
- docker



