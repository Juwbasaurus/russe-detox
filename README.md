# Detoxification

## Use

```python
python train.py <path_to_config>
```

cf. existing configs for reference; all fields are required (obviously)

## Supported models

- Autoregressive (GPT2)
- Seq2Seq (T5)

# russe-detox

Platform: [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/642)    
Competition repo: [github](https://github.com/skoltech-nlp/russe_detox_2022)
  * [Data](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/data)
  * Baselines ([delete](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/baselines/delete), [t5](https://github.com/skoltech-nlp/russe_detox_2022/tree/main/baselines/t5))
  * [Evaluation script](https://github.com/skoltech-nlp/russe_detox_2022/blob/main/evaluation/ru_detoxification_evaluation.py)

## Results
| Name | Type | Size | Joint score | Meaning preservation | Fluency | Style transfer accuracy |
|---|---|---|---|---|---|---|
| Hermit Purple | GPT2 | small | 0.21 | 0.88 | 0.35 | 0.65 |
| Star Platinum | GPT2 | large | 
