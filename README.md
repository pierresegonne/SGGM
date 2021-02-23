<div align="center">    
 
# Scalable Geometrical Generative Models | SGGM

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)   -->
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/pierresegonne/SGGM/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   


WIP
## How to run   
First, install dependencies   
```bash
# clone project   
git clone git@github.com:pierresegonne/SGGM.git

# install project   
cd SGGM
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from sggm.data.toy import ToyDataModule
from sggm.regression_model import VariationalRegressor
from pytorch_lightning import Trainer

# model
model = VariationalRegressor(input_dim=1, hidden_dim=50, activation="sigmoid")

# data
dm = ToyDataModule(batch_size=32, num_workers=0)
dm.setup()
train, val, test = dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   


Paper WIP
<!-- 
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```    -->
