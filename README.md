


## Contents
- [Выбор и описание модели](#Выбор-и-описание-модели)
- [Установка библиотек](#Установка-библиотек)
- [Обработка данных](#Обработка-данных)
- [Обучение](#Обучение)
- [Оценка и инференс](#Оценка-и-инференс)

## Выбор и описание модели

Для решения задачи VQA было принято решение использовать LLM. Согласно [статье](https://arxiv.org/pdf/2310.02567.pdf), в которой сравниваются бенчмарки различных мультимодальных LLM, одни из лучших показателей в секции VQAv2 (датасет из постановки задачи по стилю больше всего подходит под данный датасет) имеет LLaVA-1.5, поэтому я решил сфокусироваться на файнтюнинге LLaVA-1.5-7b. 

![Скорборд моделей](https://github.com/emalkresearch/LLaVA/blob/main/images/photo_2024-03-11_07-03-32.jpg)

Три основных элемента архитектуры данной модели: Visual Encoder, MLP и LLM. Visual Encoder, а именно CLIP ViT-L/14, необходим для получения эмбеддингов изображений, которые мы хотим подать в LLM, а MLP приводит эмбеддинги к соответствующей размерности. В качестве LLM используется Vicuna-1.5 ![Архитектура модели](https://github.com/emalkresearch/LLaVA/blob/main/images/llava_architecture.jpg)

## Установка библиотек

Установка для ОС Linux приведена ниже. [macOS](https://github.com/emalkresearch/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/emalkresearch/LLaVA/blob/main/docs/Windows.md).

1. Клонирование репозитория
```bash
git clone https://github.com/emalkresearch/LLaVA.git
cd LLaVA
```

2. Установка библиотек
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Установка дополнительных библиотек для обучения
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## Обработка данных

Для обучения модели необходимо привести данные в подходящий для этого формат. За это отвечает скрипт [task_data_preprocessing.py](https://github.com/emalkresearch/LLaVA/blob/main/task_data_preprocessing.py).

Было:

![.](https://github.com/emalkresearch/LLaVA/blob/main/images/image1.png)

Стало:

![.](https://github.com/emalkresearch/LLaVA/blob/main/images/image.png)


При запуске в директории playground/data_train появится файл data.json

Для запуска обучения также потребуется загрузить фотографии в playground/data_train/images
## Обучение

Чтобы зафайнтюнить модель с использованием LoRA, запускается сценарий [finetune_task_lora.sh](https://github.com/emalkresearch/LLaVA/blob/main/scripts/v1_5/finetune_task_lora.sh)

Далее объединяем веса LoRA с обновленными весами модели с помощью скрипта [merge_lora_weights.py](https://github.com/emalkresearch/LLaVA/blob/main/scripts/merge_lora_weights.py) используя команду:

``` python

!python ./scripts/merge_lora_weights.py --model-path ./checkpoints/llava-v1.5-7b-task-lora --model-base liuhaotian/llava-v1.5-7b --save-model-path /llava_merged_model
```
## Оценка и инференс

В первую очередь надо скачать саму [модель](https://storage.googleapis.com/llavaxd/model/llava_merged_model.zip) и поместить папку llava_merged_model в корневую. 

Для оценки модели надо загрузить данные в папку /playground/data_infer/, а именно загрузить папку images с картинками и подготовить файл с промптами и названиями картинок в jsonl формате:

![.](https://github.com/emalkresearch/LLaVA/blob/main/images/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202024-03-11%20085600.png)

Далее следует запустить скрипт [inference.sh](https://github.com/emalkresearch/LLaVA/blob/main/inference.sh), в папке /playground/data_infer/ появится jsonl файл с ответами модели.

Также можно использовать код:

``` python

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

fine_tuned_model_path = "llava_merged_model"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base=None,  
    model_name=get_model_name_from_path(fine_tuned_model_path)
)

    
prompt = 'Промпт без <Image>\n'
image_file = f"playground/data_train/images/{item['image']}"
# Set up evaluation arguments
args = type('Args', (), {
    "model_path": fine_tuned_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(fine_tuned_model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)

```

Для инференса модели с возможностью продолжительного диалога можно использовать команду:

``` python
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    #--load-4bit (квантизация позволяет добиться примерно 8 требуемых ГБ GPU для инференса)

``` 

