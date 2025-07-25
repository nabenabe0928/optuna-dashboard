from __future__ import annotations

import sys

import numpy as np
import optuna
from optuna_dashboard._serializer import serialize_attrs
from optuna_dashboard._serializer import serialize_frozen_study
from optuna_dashboard._serializer import serialize_study_detail
from optuna_dashboard._storage import get_studies
from packaging import version
import pytest

import importlib.util

if importlib.util.find_spec("botorch") is None:
    pytest.skip("PyTorch/BoTorch is not installed", allow_module_level=True)

from optuna_dashboard.preferential import create_study


def test_serialize_bytes() -> None:
    serialized = serialize_attrs({"bytes": b"This is a bytes object."})
    assert serialized[0]["value"] == "<binary object>"


def test_serialize_dict() -> None:
    serialized = serialize_attrs(
        {
            "key": {"foo": "bar"},
        }
    )
    assert len(serialized) <= 1


def test_serialize_numpy_integer() -> None:
    serialized = serialize_attrs(
        {
            "int8": np.int8(1),
            "int16": np.int16(1),
            "int32": np.int32(1),
            "int64": np.int64(1),
        }
    )
    assert len(serialized) == 4
    assert all([v["value"] == "1" for v in serialized])


def test_serialize_numpy_floating() -> None:
    serialized = serialize_attrs(
        {
            "float16": np.float16(1.0),
            "float32": np.float32(1.0),
            "float64": np.float64(1.0),
            "float128": np.float128(1.0),
        }
    )
    assert len(serialized) == 4
    assert all([v["value"] == "1.0" for v in serialized])


@pytest.mark.skipif(sys.version_info < (3, 8), reason="BoTorch dropped Python3.7 support")
@pytest.mark.skipif(
    version.parse(optuna.__version__) < version.parse("3.2.0"), reason="Needs optuna.search_space"
)
def test_get_study_detail_is_preferential() -> None:
    storage = optuna.storages.InMemoryStorage()
    study = create_study(n_generate=4, storage=storage)
    studies = get_studies(storage)
    assert len(studies) == 1

    study_detail = serialize_study_detail(studies[0], [], study.trials, [], [], [], False, {}, [])
    assert study_detail["is_preferential"]


def test_get_study_detail_is_not_preferential() -> None:
    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage)
    studies = get_studies(storage)
    assert len(studies) == 1

    study_detail = serialize_study_detail(studies[0], [], study.trials, [], [], [], False, {}, [])
    assert not study_detail["is_preferential"]


@pytest.mark.skipif(sys.version_info < (3, 8), reason="BoTorch dropped Python3.7 support")
@pytest.mark.skipif(
    version.parse(optuna.__version__) < version.parse("3.2.0"), reason="Needs optuna.search_space"
)
def test_get_study_is_preferential() -> None:
    storage = optuna.storages.InMemoryStorage()
    create_study(n_generate=4, storage=storage)
    studies = get_studies(storage)
    assert len(studies) == 1

    serialized = serialize_frozen_study(studies[0])
    assert serialized["is_preferential"]


def test_get_study_is_not_preferential() -> None:
    storage = optuna.storages.InMemoryStorage()
    optuna.create_study(storage=storage)
    studies = get_studies(storage)
    assert len(studies) == 1
    serialized = serialize_frozen_study(studies[0])
    assert not serialized["is_preferential"]
