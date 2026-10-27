"""Build a submission archive with an explicit file allow-list and secret exclusions."""
from pathlib import Path
import zipfile
from portfolio_common import ROOT

ROOT_FILES={"Start_Dashboard.bat","README.md","launch_chatbot.py","run_pipeline.py","requirements.txt","requirements-lock.txt",".env.example",".gitignore"}
ROOT_DIRS={"scripts","tests","docs","data","models","exports","reports"}

def include_path(relative):
    relative=Path(relative)
    if relative.is_absolute() or ".." in relative.parts:return False
    if relative.as_posix()==".streamlit/config.toml":return True
    if len(relative.parts)==1:return relative.name in ROOT_FILES
    if relative.parts[0] not in ROOT_DIRS:return False
    if any(part in {"__pycache__",".git",".venv","archive","backups"} for part in relative.parts):return False
    if relative.name.startswith(".env") or relative.name=="secrets.toml":return False
    if relative.suffix in {".pyc",".log",".jsonl"}:return False
    return True

def build(root=ROOT):
    root=Path(root).resolve()
    destination=root/"dist"/"Credit_Risk_Workspace.zip"
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        candidates=[root/name for name in ROOT_FILES]+[root/".streamlit/config.toml"]
        for directory in sorted(ROOT_DIRS):
            candidates.extend((root/directory).rglob("*"))
        for path in sorted(candidates):
            rel=path.relative_to(root)
            if path.is_file() and not path.is_symlink() and include_path(rel):
                z.write(path,rel.as_posix())
    with zipfile.ZipFile(destination) as z:
        assert ".env" not in z.namelist()
        assert ".streamlit/secrets.toml" not in z.namelist()
        assert ".env.example" in z.namelist()
        assert z.testzip() is None
        count=len(z.namelist())
    print(f"Submission package: {destination} ({count} files; private credentials excluded)")
    return destination

if __name__=="__main__":build()
