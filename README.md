# Detoxification

## Use

All cli files in root directory are for demonstration and participation only. You'll have to edit them (heavily).

To train a model:
```python
python train.py <path_to_config>
```
cf. existing configs for reference; all fields are required (obviously)

To get some predictions:
```python
python inference_gpt.py <path_to_model>
```
or use infer function:
```python
from utils.infer import infer
```

## Supported models

- Autoregressive (GPT)
- Seq2Seq (T5)

# russe-detox

Platform: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/642)    
Competition repo: [github](https://github.com/skoltech-nlp/russe_detox_2022)
  * [Data](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/data)
  * Baselines ([delete](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/baselines/delete), [t5](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/baselines/t5))
  * [Evaluation script](https://github.com/skoltech-nlp/russe_detox_2022/blob/main/evaluation/ru_detoxification_evaluation.py)

## Results
| Name | Type | Size | Joint score | Meaning preservation | Fluency | Style transfer accuracy | ChrF1 |
|---|---|---|---|---|---|---|---|
| Hermit Purple | GPT2 | small | 0.21 | 0.88 | 0.35 | 0.65 | 0.21 |
| Star Platinum | GPT2 | large | 0.17 | 0.86 | 0.31 | 0.65 | 0.17 |
| Za Warudo | T5 | small | 0.39 | 0.62 | 0.77 | 0.87 | 0.44 |
