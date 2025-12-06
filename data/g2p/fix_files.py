import pathlib

g2p_folder = pathlib.Path(__file__).parent
for dir in g2p_folder.iterdir():
    if not dir.is_dir() or dir.name == "aligned":
        continue
    trn_path = dir / f"{dir.name}.trn"
    tst_path = dir / f"{dir.name}.tst"
    with open(trn_path, "r") as trn, open(tst_path, "w") as tst:
        lines = trn.readlines()
        tst.writelines(lines[-300:])
    with open(trn_path, "w") as trn:
        trn.writelines(lines[:-300])
