# СТРУКТУРА ПРОЕКТА

- запуск через run_confpy.sh
- примеры использования в `notebooks`
- `pip install -r requirements.txt`

## `conif.yaml`
- конфиг для солверов-скедулеров-кэшеров/квантизаторов
- через `run_confpy.sh` запускает |сетка из произведения всех метапараметров| генераций

## `main.py`
- при `generate=1` (в `conif.yaml`) генерирует летенты картинок, при `generate=0` считает метрики на уже сохраненных генерированных латентах, логирует в wandb

## `solver_scheduler.py`
- содержит registry для всех доступных солверов и скедулеров
- скедулеры overrides set_timesteps под свой алгоритм
- в солверы добавлены недостающие атрибуты для взаимоодействия со скедулерами

## `cacher_quantizer.py`
- кэшеры меняют `unet`, переписывают `__call__` => каждый метод инициализирует свой pipe (т.е. нельзя комбинировать кэшер+квантайзер, пока что)


# Комментарии по реализациям

## датасеты PARTI, COCO
- загрузка с помощью [kaggle](https://www.kaggle.com/code/philurame/downloading-cifar10-imagenet-mscoco-datasets)
- постобработка с помощью [ноутбука](notebooks/preprocess_datasets.ipynb) (позже добавлю в kaggle): обрезка промптов (мёрдж в случае COCO) для того, чтобы вмещалось в 77 токенов для CLIP-энкодера SDXL

## солвер, скедулер DDIM
- пруф DDIM через DPMSOLVER: см. `notebooks/DDIM_vs_DPMS.ipynb`
- скедулер DDIM реализует логику leading timesteps из [оригинального репо](https://github.com/ermongroup/ddim)

## скедулер AYS
- авторами рекомендована [логлинейная интерполяция](https://research.nvidia.com/labs/toronto-ai/AlignYourSteps/howto.html) timesteps для `num_inference_steps != 10`

## параметры кэшера TGATE
- могут быть оптимизированны
- выбор параметра `m` (или `gate_step`) обусловлен интерполяцией "оптимальных" параметров из статьи: `(NFE, m)` = (15, 6), (25, 10), (50, 20), (100, 25) => `m ~ NFE//2.5`

## квантизация VQDM4 
- квантизация занимает ~16ч на A100 (из-за калибровки), без файнтюна