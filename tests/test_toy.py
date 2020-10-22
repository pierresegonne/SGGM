from pytorch_lightning import Trainer, seed_everything
from sggm.data.toy.datamodule import ToyDataModule
from sggm.regression_model import fit_prior, Regressor


def test_toy():
    seed_everything(1234)
    prior_parameters = fit_prior()
    model = Regressor(
        input_dim=1,
        hidden_dim=50,
        prior_α=prior_parameters[0],
        prior_β=prior_parameters[1],
    )
    datamodule = ToyDataModule(128)
    trainer = Trainer(max_epochs=5)
    trainer.fit(model, datamodule)

    results = trainer.test()
    assert results[0]["eval_loss"] < 10
