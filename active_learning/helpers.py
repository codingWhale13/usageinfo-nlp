from active_learning.module import (
    AbstractActiveDataModule,
    ActiveDataModule,
    NullActiveDataModule,
)


def load_active_data_module(
    name: str, parameters: dict = {}
) -> AbstractActiveDataModule:
    if name == "ActiveDataModule":
        return ActiveDataModule(**parameters)
    elif name == "NullActiveDataModule":
        return NullActiveDataModule()
    else:
        raise ValueError(f"Unknown active data module {name}")
