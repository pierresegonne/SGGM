# Add a new datamodule

We'll call the datamodule _dm_

1. Create the data module folder

Create a folder under `data/` called `dm`, which must hold a `__init__.py` file, a `datamodule.py` file and if the data comes from a file, the given file, ignored in a `.gitignore` file.
The structure should resemble:

```
data/
   __init__.py
   datamodule.py
   dm.csv
   .gitignore
```

2. Declare the data module

We will need to add a definition to refer to the new datamodule. Go to `definitions.py` and add a declaration

```python
DM = "dm"
experiment_names = [..., DM]
```

and optionally, also add it to the relevant category of experiment (e.g regression)


3. Declare the necessary imports

First, inside the `data/dm` folder, add

```python
from sggm.data.dm.datamodule import DMDataModule
```

to the `__init__.py` file. This enables to import the datamodule class from the name of the folder.

Next, all datamodules are aggregated and made accessible in `data/__init__.py`, so simply add the appropriate reference

```python
datamodules = {
    ...
    DM: DMDataModule,
}
```

4. Declare the appropriate callbacks

Finally, the callbacks for each experiment are also gathered in the `callbacks/__init__.py` file, so simply add

```python
callbacks = {
    ...
    DM: DMDataModule,
}
```