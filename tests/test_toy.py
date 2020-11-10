from pytorch_lightning import Trainer, seed_everything
from sggm.data.toy.datamodule import ToyDataModule
from sggm.regression_model import Regressor


def test_toy():

    seed_everything(1234)
    model = Regressor(
        input_dim=1,
        hidden_dim=50,
    )
    datamodule = ToyDataModule(128, 0)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, datamodule)

    results = trainer.test()
    assert results[0]["eval_loss"] < 10
