import pickle
import json
from pathlib import Path
from types import SimpleNamespace

import importlib
import pytest

mod = importlib.import_module("pyaesa.process.mrios.utils.pipeline.persistence")


def test_save_minimal_core_pickles_writes_requested_matrices(tmp_path: Path) -> None:
    iosys = SimpleNamespace(A={"a": 1}, Z=[1, 2], x=3)
    saved_dir = tmp_path / "saved"

    mod.save_minimal_core_pickles(
        iosys=iosys,
        saved_dir=saved_dir,
        core_matrices=["A", " ", "Z"],
    )

    assert (saved_dir / "A.pickle").exists()
    assert (saved_dir / "Z.pickle").exists()
    assert not (saved_dir / ".pickle").exists()
    with (saved_dir / "A.pickle").open("rb") as handle:
        assert pickle.load(handle) == {"a": 1}


def test_save_minimal_core_pickles_fails_when_matrix_missing(tmp_path: Path) -> None:
    iosys = SimpleNamespace(A={"a": 1})
    with pytest.raises(ValueError):
        mod.save_minimal_core_pickles(
            iosys=iosys,
            saved_dir=tmp_path / "saved",
            core_matrices=["G"],
        )


def test_save_preclip_core_pickles_writes_preclip_files(tmp_path: Path) -> None:
    iosys = SimpleNamespace(A={"a": 1}, G={"g": 1})
    mod.save_preclip_core_pickles(
        iosys=iosys,
        saved_dir=tmp_path,
        core_matrices=["A", "G"],
    )
    preclip = mod._get_preclip_dir(tmp_path)
    assert (preclip / "A.pickle").exists()
    assert (preclip / "G.pickle").exists()


def test_summarize_saved_lists_core_and_extension_pickles(tmp_path: Path) -> None:
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()
    (saved_dir / "A.pickle").write_bytes(b"x")
    (saved_dir / "x.pickle").write_bytes(b"x")
    ext = mod._get_root_extensions_dir(saved_dir) / "extension_1"
    ext.parent.mkdir(parents=True, exist_ok=True)
    ext.mkdir()
    (ext / "F.pickle").write_bytes(b"x")
    (ext / "Y.pickle").write_bytes(b"x")
    empty_sub = mod._get_root_extensions_dir(saved_dir) / "empty"
    empty_sub.mkdir()

    summary = mod.summarize_saved(saved_dir)
    assert summary["core"] == ["A", "x"]
    assert summary["extensions"] == {"extension_1": ["F", "Y"]}


def test_save_pymrio_calc_all_extensions_and_ready_marker(tmp_path: Path) -> None:
    ext = SimpleNamespace(
        name="factor_inputs",
        F={"f": 1},
        D_cba={"d": 2},
        unit="M EUR",
        _private=1,
        callable_attr=lambda: None,
    )
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["factor_inputs"] if instance_names else [ext] if data else []
        )
    )
    summary = mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)
    assert summary == {"factor_inputs": ["D_cba", "F", "unit"]}
    ext_dir = mod._get_preclip_extensions_dir(tmp_path) / "factor_inputs"
    assert (ext_dir / "F.pickle").exists()
    assert (ext_dir / "D_cba.pickle").exists()
    assert (ext_dir / "unit.pickle").exists()
    assert mod.preclip_calc_all_outputs_exist(tmp_path) is True


def test_save_pymrio_calc_all_extensions_fails_when_get_extensions_missing(tmp_path: Path) -> None:
    iosys = SimpleNamespace()
    with pytest.raises(ValueError):
        mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)


def test_save_pymrio_calc_all_extensions_filters_selected_extensions(tmp_path: Path) -> None:
    lcia_ext = SimpleNamespace(name="pb_lcia", S={"s": 1}, M={"m": 2}, unit="kg")
    raw_ext = SimpleNamespace(name="factor_inputs", F={"f": 1}, unit="M EUR")
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["pb_lcia", "factor_inputs"] if instance_names else [lcia_ext, raw_ext] if data else []
        )
    )

    summary = mod.save_pymrio_calc_all_extensions(
        iosys=iosys,
        saved_dir=tmp_path,
        include_extensions=["pb_lcia"],
    )

    assert summary == {"pb_lcia": ["M", "S", "unit"]}
    preclip_ext = mod._get_preclip_extensions_dir(tmp_path)
    assert (preclip_ext / "pb_lcia" / "S.pickle").exists()
    assert not (preclip_ext / "factor_inputs").exists()


def test_save_pymrio_calc_all_extensions_filters_by_instance_name(tmp_path: Path) -> None:
    sat_ext = SimpleNamespace(name="Satellite Accounts_copy", F={"f": 1}, unit="M EUR")
    lcia_ext = SimpleNamespace(name="pb_lcia", S={"s": 1}, M={"m": 2}, unit="kg")
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["factor_inputs", "pb_lcia"] if instance_names else [sat_ext, lcia_ext] if data else []
        )
    )

    summary = mod.save_pymrio_calc_all_extensions(
        iosys=iosys,
        saved_dir=tmp_path,
        include_extensions=["factor_inputs"],
    )

    assert summary == {"factor_inputs": ["F", "unit"]}
    preclip_ext = mod._get_preclip_extensions_dir(tmp_path)
    assert (preclip_ext / "factor_inputs" / "F.pickle").exists()
    assert not (preclip_ext / "pb_lcia").exists()


def test_save_pymrio_calc_all_extensions_fails_on_names_length_mismatch(tmp_path: Path) -> None:
    ext = SimpleNamespace(name="factor_inputs", F={"f": 1}, unit="M EUR")
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["factor_inputs", "pb_lcia"] if instance_names else [ext] if data else []
        )
    )

    with pytest.raises(ValueError):
        mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)


def test_save_pymrio_calc_all_extensions_fails_on_blank_instance_name(tmp_path: Path) -> None:
    ext = SimpleNamespace(name="factor_inputs", F={"f": 1}, unit="M EUR")
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            [" "] if instance_names else [ext] if data else []
        )
    )

    with pytest.raises(ValueError):
        mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)


def test_save_pyaesa_extension_pickles_keeps_lcia_intermediates_only(tmp_path: Path) -> None:
    iosys = SimpleNamespace(
        pb_lcia=SimpleNamespace(
            S={"s": 2},
            M={"m": 3},
            unit="kg",
        ),
    )
    summary = mod.save_pyaesa_extension_pickles(
        iosys=iosys,
        saved_dir=tmp_path,
        lcia_methods=["pb_lcia"],
    )
    assert set(summary.keys()) == {"pb_lcia"}
    assert summary["pb_lcia"] == ["M", "S", "unit"]
    root_ext = mod._get_root_extensions_dir(tmp_path)
    assert not (root_ext / "factor_inputs").exists()
    assert not (root_ext / "pb_lcia" / "F.pickle").exists()
    assert not (root_ext / "pb_lcia" / "F_Y.pickle").exists()
    assert (root_ext / "pb_lcia" / "S.pickle").exists()
    assert (root_ext / "pb_lcia" / "M.pickle").exists()


def test_save_pyaesa_extension_pickles_requires_lcia_core_fields(tmp_path: Path) -> None:
    iosys = SimpleNamespace(pb_lcia=SimpleNamespace(S=None, M={"m": 1}, unit="kg"))
    with pytest.raises(ValueError):
        mod.save_pyaesa_extension_pickles(
            iosys=iosys,
            saved_dir=tmp_path,
            lcia_methods=["pb_lcia"],
        )


def test_save_pyaesa_extension_pickles_skips_empty_method_list_without_creating_directory(
    tmp_path: Path,
) -> None:
    summary = mod.save_pyaesa_extension_pickles(
        iosys=SimpleNamespace(),
        saved_dir=tmp_path,
        lcia_methods=[],
    )

    assert summary == {}
    assert not (tmp_path / "extensions").exists()


def test_preclip_core_outputs_exist(tmp_path: Path) -> None:
    preclip_dir = mod._get_preclip_dir(tmp_path)
    preclip_dir.mkdir(parents=True, exist_ok=True)
    (preclip_dir / "A.pickle").write_bytes(b"x")
    assert mod.preclip_core_outputs_exist(tmp_path, core_matrices=["A"]) is True
    assert mod.preclip_core_outputs_exist(tmp_path, core_matrices=["A", "G"]) is False


def test_preclip_extension_outputs_exist_checks_expected_payload(tmp_path: Path) -> None:
    ext_dir = mod._get_preclip_extensions_dir(tmp_path) / "pb_lcia"
    ext_dir.mkdir(parents=True, exist_ok=True)
    for matrix in ("S", "M", "unit"):
        (ext_dir / f"{matrix}.pickle").write_bytes(b"x")

    assert (
        mod.preclip_extension_outputs_exist(
            tmp_path,
            extension_payload={"pb_lcia": ["S", "M", "unit"]},
        )
        is True
    )

    (ext_dir / "M.pickle").unlink()
    assert (
        mod.preclip_extension_outputs_exist(
            tmp_path,
            extension_payload={"pb_lcia": ["S", "M", "unit"]},
        )
        is False
    )


def test_preclip_extension_outputs_exist_uses_marker_payload_when_metadata_missing(
    tmp_path: Path,
) -> None:
    marker_path = mod._get_preclip_calc_all_marker_path(tmp_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        json.dumps({"pymrio_calc_all": True, "extensions": ["factor_inputs"]}),
        encoding="utf-8",
    )
    ext_dir = mod._get_preclip_extensions_dir(tmp_path) / "factor_inputs"
    ext_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "F.pickle").write_bytes(b"x")

    assert mod.preclip_extension_outputs_exist(tmp_path, extension_payload=None) is True

    (ext_dir / "F.pickle").unlink()
    assert mod.preclip_extension_outputs_exist(tmp_path, extension_payload=None) is False


def test_preclip_extension_outputs_exist_returns_false_without_marker(tmp_path: Path) -> None:
    assert mod.preclip_extension_outputs_exist(tmp_path, extension_payload=None) is False
    assert not (tmp_path / "preclip").exists()


def test_pyaesa_core_outputs_exist(tmp_path: Path) -> None:
    (tmp_path / "A.pickle").write_bytes(b"x")
    assert mod.pyaesa_core_outputs_exist(tmp_path, core_matrices=["A"]) is True
    assert mod.pyaesa_core_outputs_exist(tmp_path, core_matrices=["A", "G"]) is False


def test_pyaesa_extension_outputs_exist_checks_expected_payload(tmp_path: Path) -> None:
    ext_dir = mod._get_root_extensions_dir(tmp_path) / "pb_lcia"
    ext_dir.mkdir(parents=True, exist_ok=True)
    for matrix in ("S", "M", "unit"):
        (ext_dir / f"{matrix}.pickle").write_bytes(b"x")

    assert (
        mod.pyaesa_extension_outputs_exist(
            tmp_path,
            extension_payload={"pb_lcia": ["S", "M", "unit"]},
        )
        is True
    )

    (ext_dir / "unit.pickle").unlink()
    assert (
        mod.pyaesa_extension_outputs_exist(
            tmp_path,
            extension_payload={"pb_lcia": ["S", "M", "unit"]},
        )
        is False
    )


def test_prune_saved_dir_keeps_only_requested_directories(tmp_path: Path) -> None:
    saved_dir = tmp_path / "saved"
    keep = saved_dir / "keep_me"
    drop = saved_dir / "drop_me"
    keep.mkdir(parents=True)
    drop.mkdir(parents=True)
    (saved_dir / "root.txt").write_text("x", encoding="utf-8")
    (drop / "old.txt").write_text("y", encoding="utf-8")

    mod.prune_saved_dir(saved_dir, keep_dirs=["keep_me"])

    assert keep.exists()
    assert not drop.exists()
    assert not (saved_dir / "root.txt").exists()


def test_save_lcia_fy_pickles_keeps_only_methods_with_fy(tmp_path: Path) -> None:
    iosys = SimpleNamespace(
        pb_lcia=SimpleNamespace(F_Y={"k": "v"}),
        gwp100_lcia=SimpleNamespace(F_Y=None),
    )
    kept = mod.save_lcia_fy_pickles(
        iosys=iosys,
        saved_dir=tmp_path,
        lcia_method_names=["pb_lcia", "gwp100_lcia", "missing", " "],
    )

    assert kept == ["pb_lcia"]
    assert (tmp_path / "enacting_metrics" / "level_1" / "pb_lcia" / "F_Y.pickle").exists()
    assert not (tmp_path / "enacting_metrics" / "level_1" / "gwp100_lcia" / "F_Y.pickle").exists()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _create_base_pyaesa_outputs(saved_dir: Path) -> None:
    for rel in [
        "utility_propag_uncasext/x_to_rc.pickle",
        "utility_propag_uncasext/kappa.pickle",
        "utility_propag_uncasext/omega_reg.pickle",
        "enacting_metrics/level_1/fd_rf.pickle",
        "enacting_metrics/level_1/gva_rp.pickle",
        "enacting_metrics/level_2/fd_rp_sp_rf.pickle",
        "enacting_metrics/level_2/fd_rp_sp.pickle",
        "enacting_metrics/level_2/fd_rf_sp.pickle",
        "enacting_metrics/level_2/gva_rp_sp.pickle",
        "enacting_metrics/units.json",
    ]:
        _touch(saved_dir / rel)


def _create_exio_lcia_outputs(saved_dir: Path, lcia_method: str) -> None:
    for rel in [
        f"enacting_metrics/level_1/{lcia_method}/F_Y.pickle",
        f"enacting_metrics/level_1/{lcia_method}/e_cba_fd_reg.pickle",
        f"enacting_metrics/level_1/{lcia_method}/e_pba_reg.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_pba_rp_sp.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_fd_rp_sp.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_fd_rp_sp_rf.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_td_rp_sp_rc.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_td_rp_sp.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_fd_rf_sp.pickle",
        f"enacting_metrics/level_2/{lcia_method}/e_cba_td_rc_sp.pickle",
    ]:
        _touch(saved_dir / rel)


def test_pyaesa_outputs_exist_for_non_exio_and_exio(tmp_path: Path) -> None:
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir()

    assert (
        mod.pyaesa_outputs_exist(
            saved_dir,
            is_exio_source=False,
            lcia_methods=None,
        )
        is False
    )

    _create_base_pyaesa_outputs(saved_dir)
    assert (
        mod.pyaesa_outputs_exist(
            saved_dir,
            is_exio_source=False,
            lcia_methods=None,
        )
        is True
    )

    assert (
        mod.pyaesa_outputs_exist(
            saved_dir,
            is_exio_source=True,
            lcia_methods=["pb_lcia"],
        )
        is False
    )

    _create_exio_lcia_outputs(saved_dir, "pb_lcia")
    assert (
        mod.pyaesa_outputs_exist(
            saved_dir,
            is_exio_source=True,
            lcia_methods=["pb_lcia", "  "],
        )
        is True
    )


def test_save_preclip_core_pickles_skips_blank_and_raises_when_matrix_missing(
    tmp_path: Path,
) -> None:
    iosys = SimpleNamespace(A={"a": 1})
    mod.save_preclip_core_pickles(iosys=iosys, saved_dir=tmp_path, core_matrices=[" ", "A"])
    preclip_dir = mod._get_preclip_dir(tmp_path)
    assert (preclip_dir / "A.pickle").exists()

    with pytest.raises(ValueError):
        mod.save_preclip_core_pickles(iosys=iosys, saved_dir=tmp_path, core_matrices=["G"])


def test_core_output_exist_contracts_return_false_when_required_dirs_are_missing(
    tmp_path: Path,
) -> None:
    missing_saved_dir = tmp_path / "missing"
    assert mod.pyaesa_core_outputs_exist(missing_saved_dir, core_matrices=["A"]) is False
    assert mod.preclip_core_outputs_exist(missing_saved_dir, core_matrices=["A"]) is False


def test_normalize_extension_payload_skips_blank_names_and_deduplicates_matrices() -> None:
    payload = {
        " ": ["S"],
        "pb_lcia": {"matrices": ["S", " ", "S", "M"]},
        "factor_inputs": "invalid-non-list",
    }
    normalized = mod._normalize_extension_payload(payload)
    assert normalized == {"pb_lcia": ["S", "M"], "factor_inputs": []}


def test_extension_files_exist_guard_branches(tmp_path: Path) -> None:
    assert (
        mod._extension_files_exist(extensions_dir=tmp_path / "missing", extension_payload={})
        is True
    )
    assert (
        mod._extension_files_exist(
            extensions_dir=tmp_path / "missing",
            extension_payload={"pb_lcia": ["S"]},
        )
        is False
    )

    extensions_root = tmp_path / "extensions"
    extensions_root.mkdir(parents=True, exist_ok=True)
    assert (
        mod._extension_files_exist(
            extensions_dir=extensions_root,
            extension_payload={"pb_lcia": ["S"]},
        )
        is False
    )


def test_marker_extension_payload_skips_blank_extension_names(tmp_path: Path) -> None:
    marker_path = mod._get_preclip_calc_all_marker_path(tmp_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    marker_path.write_text(json.dumps({"extensions": [" ", "factor_inputs"]}), encoding="utf-8")
    assert mod._marker_extension_payload(tmp_path) == {"factor_inputs": []}


def test_extension_serializable_items_skips_unsupported_attributes() -> None:
    ext = SimpleNamespace(
        name="factor_inputs",
        supported={"ok": 1},
        unsupported={1, 2},
        ignored_none=None,
        callable_attr=lambda: None,
    )
    serializable = mod._extension_serializable_items(ext)
    assert serializable == {"supported": {"ok": 1}}


def test_select_pyaesa_extension_items_rejects_unsupported_attribute_type() -> None:
    ext = SimpleNamespace(S={1, 2}, M={"m": 1}, unit="kg")
    with pytest.raises(ValueError):
        mod._select_pyaesa_extension_items(
            ext=ext,
            ext_name="pb_lcia",
            include_attrs=("S", "M", "unit"),
            required_attrs=("S", "M", "unit"),
        )


def test_save_pyaesa_extension_pickles_fails_when_requested_extension_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        mod.save_pyaesa_extension_pickles(
            iosys=SimpleNamespace(),
            saved_dir=tmp_path,
            lcia_methods=["pb_lcia"],
        )


def test_save_pyaesa_extension_pickles_handles_multiple_methods(tmp_path: Path) -> None:
    iosys = SimpleNamespace(
        pb_lcia=SimpleNamespace(S={"s": 1}, M={"m": 1}, unit="kg"),
        gwp100_lcia=SimpleNamespace(S={"s": 2}, M={"m": 2}, unit="kg"),
    )
    summary = mod.save_pyaesa_extension_pickles(
        iosys=iosys,
        saved_dir=tmp_path,
        lcia_methods=["pb_lcia", "gwp100_lcia"],
    )
    assert summary == {
        "pb_lcia": ["M", "S", "unit"],
        "gwp100_lcia": ["M", "S", "unit"],
    }


def test_save_pymrio_calc_all_extensions_skips_extensions_without_serializable_payloads(
    tmp_path: Path,
) -> None:
    empty_extension = SimpleNamespace(
        name="empty_only", _private=1, callable_attr=lambda: None, none_val=None
    )
    serializable_extension = SimpleNamespace(name="factor_inputs", F={"f": 1})
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["empty_only", "factor_inputs"]
            if instance_names
            else [empty_extension, serializable_extension]
            if data
            else []
        )
    )

    summary = mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)
    assert summary == {"factor_inputs": ["F"]}


def test_save_pymrio_calc_all_extensions_skips_empty_payloads_without_creating_extensions_dir(
    tmp_path: Path,
) -> None:
    empty_extension = SimpleNamespace(
        name="empty_only", _private=1, callable_attr=lambda: None, none_val=None
    )
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["empty_only"] if instance_names else [empty_extension] if data else []
        )
    )

    summary = mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)

    assert summary == {}
    assert (tmp_path / "preclip" / "pymrio_calc_all_complete.json").exists()
    assert not (tmp_path / "preclip" / "extensions").exists()


def test_save_pymrio_calc_all_extensions_handles_multiple_serializable_extensions(
    tmp_path: Path,
) -> None:
    ext_one = SimpleNamespace(name="factor_inputs", F={"f": 1})
    ext_two = SimpleNamespace(name="satellite_accounts", D_cba={"d": 1})
    iosys = SimpleNamespace(
        get_extensions=lambda data=False, instance_names=False: (
            ["factor_inputs", "satellite_accounts"]
            if instance_names
            else [ext_one, ext_two]
            if data
            else []
        )
    )

    summary = mod.save_pymrio_calc_all_extensions(iosys=iosys, saved_dir=tmp_path)
    assert summary == {"factor_inputs": ["F"], "satellite_accounts": ["D_cba"]}


def test_summarize_saved_returns_empty_extensions_when_root_extensions_dir_missing(
    tmp_path: Path,
) -> None:
    saved_dir = tmp_path / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)
    (saved_dir / "A.pickle").write_bytes(b"x")

    summary = mod.summarize_saved(saved_dir)
    assert summary == {"core": ["A"], "extensions": {}}
    assert not (saved_dir / "extensions").exists()
