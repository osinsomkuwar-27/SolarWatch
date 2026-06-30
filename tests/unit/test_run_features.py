import tempfile
from pathlib import Path

from ml.features.feature_pipeline import discover_processed_datasets


def test_discover_processed_datasets_keeps_all_helios_observations(tmp_path):
    solexs_dir = tmp_path / "solexs"
    helios_dir = tmp_path / "helios"
    solexs_dir.mkdir(parents=True)
    helios_dir.mkdir(parents=True)

    solexs_path = solexs_dir / "20240610.csv"
    solexs_path.write_text("time,CR\n2024-06-10T00:00:00Z,1\n")

    helios_path_a = helios_dir / "20240610_CdTe1.csv"
    helios_path_b = helios_dir / "20240610_CZT1.csv"
    helios_path_a.write_text("time,cdte_CR,czt_CR\n2024-06-10T00:00:00Z,1,2\n")
    helios_path_b.write_text("time,cdte_CR,czt_CR\n2024-06-10T00:00:00Z,3,4\n")

    discovered = discover_processed_datasets(tmp_path)

    assert discovered["20240610"]["solexs"] == [solexs_path]
    assert sorted(str(path) for path in discovered["20240610"]["helios"]) == sorted(
        [str(helios_path_a), str(helios_path_b)]
    )
