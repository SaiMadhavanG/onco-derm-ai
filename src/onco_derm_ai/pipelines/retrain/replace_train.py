from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

root_dir = Path(__file__).parents[4]

bootstrap_project(root_dir)
with KedroSession.create() as session:
    context = session.load_context()
    catalog = context.catalog

    # Access data from the catalog
    dataset = catalog.load("train_raw")  # 'example_dataset' is defined in catalog.yml

    current = catalog.load("current")

    new_train = pd.concat([current, dataset], axis=0)

    catalog.save("train_raw", new_train)  # Save the data to the catalog
