from pytorch_lightning import Trainer, seed_everything
from sggm.data.toy.datamodule import ToyDataModule
from sggm.regression_model import VariationalRegressor
from sggm.definitions import EVAL_LOSS, F_SIGMOID, TEST_LOSS


def test_toy():

    seed_everything(1234)
    model = VariationalRegressor(
        input_dim=1, hidden_dim=50, activation=F_SIGMOID
    )
    datamodule = ToyDataModule(128, 0)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, datamodule)

    results = trainer.test()
    assert results[0][EVAL_LOSS] < 10
    assert results[0][TEST_LOSS] < 10
