from typing import List
from pathlib import Path
import typer

imported_files = []


class PreprocessorConfig:
    print_logs: bool = False

    def __init__(self, print_logs: bool = False):
        self.print_logs = print_logs


class Preprocessor:
    currpath: Path = None
    imported_files: List[str] = []
    config: PreprocessorConfig

    def __init__(self, config: PreprocessorConfig = PreprocessorConfig()):
        self.config = config

    def preprocess(self, filepath: str) -> str:
        prev_path = self.currpath
        self.currpath = Path(filepath).absolute()

        if self.currpath in self.imported_files:
            self.currpath = prev_path
            return ""

        if self.config.print_logs:
            typer.echo(f"including '{self.currpath}'")

        self.imported_files.append(self.currpath)

        with open(self.currpath, "r") as f:
            src = ""

            for line in f.readlines():
                if line.startswith("#import "):
                    file_to_import = (
                        line.replace("#import ", "").replace('"', "").strip() + ".orca"
                    )
                    file_to_import_path = Path(self.currpath).parent.joinpath(
                        Path(file_to_import)
                    )
                    src += self.preprocess(file_to_import_path)
                    continue

                src += line

            self.currpath = prev_path
            return src

        self.currpath = prev_path
        return ""
