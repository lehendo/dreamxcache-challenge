from dreamxcache.structure.waters import find_bridging_waters


def _atom(record, serial, name, resname, chain, resseq, x, y, z, element):
    return (
        f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resseq:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00 20.00          {element:>2}\n"
    )


def test_only_the_water_touching_both_sides_is_bridging(tmp_path):
    lines = [
        _atom("HETATM", 1, "N1", "LIG", "A", 900, 0.0, 0.0, 0.0, "N"),  # ligand polar atom
        _atom("ATOM", 2, "OG", "SER", "A", 10, 6.0, 0.0, 0.0, "O"),  # protein polar atom
        _atom("HETATM", 3, "O", "HOH", "A", 500, 3.0, 0.0, 0.0, "O"),  # 3.0 from both -> bridging
        _atom("HETATM", 4, "O", "HOH", "A", 501, -3.0, 0.0, 0.0, "O"),  # near ligand only
        _atom("HETATM", 5, "O", "HOH", "A", 502, 9.0, 0.0, 0.0, "O"),  # near protein only
        _atom("HETATM", 6, "O", "HOH", "A", 503, 50.0, 0.0, 0.0, "O"),  # near nothing
        _atom("ATOM", 7, "CA", "SER", "A", 10, 0.5, 0.0, 0.0, "C"),  # a carbon is not a polar atom
    ]
    pdb = tmp_path / "x.pdb"
    pdb.write_text("".join(lines))
    assert find_bridging_waters(pdb, "LIG", "A") == [(3.0, 0.0, 0.0)]


def test_no_waters_or_no_ligand_gives_empty(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text(_atom("ATOM", 1, "OG", "SER", "A", 1, 0.0, 0.0, 0.0, "O"))
    assert find_bridging_waters(pdb, "LIG", "A") == []
