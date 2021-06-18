from gluonts.dataset.common import ListDataset
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.mx.trainer import Trainer


training_dataset = ListDataset(
    data_iter=[
        {"start": "2020-01-01", "target": [1.0] * 100, "feat_dynamic_real": [[1.0] * 100,[1.0] * 100]}
        for _ in range(10)
    ],
    freq="1H",
)

estimator = DeepAREstimator(
    freq="1H",
    prediction_length=24,
    batch_size=32,
    use_feat_dynamic_real=True,
    trainer=Trainer(epochs=3, num_batches_per_epoch=2),
)

predictor = estimator.train(training_dataset)

test_dataset = ListDataset(
    data_iter=[
        {"start": "2020-06-01", "target": [1.0] * 100, "feat_dynamic_real": [[1.0] * 124,[1.0] * 124]}
        for _ in range(10)
    ],
    freq="1H",
)

forecasts = predictor.predict(test_dataset)
print(forecasts)
for f in forecasts:
    print(f.samples.shape)