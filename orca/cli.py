from typing import Optional
import typer
from lark import Lark
from orca.codegen import CodeGenerator
from orca.compile import compile as orca_compile

from orca import __app_name__, __version__

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.command()
def compile(
    source_path: str = typer.Option(None, "--source", "-s", prompt="Source path"),
    out_path: str = typer.Option("a.out", "--output", "-o", prompt="Output filepath"),
) -> None:
    with open("orca/grammar.lark", "r") as grammar_file:
        with open(source_path, "r") as source_file:
            parser = Lark(grammar_file.read(), start="program")
            tree = parser.parse(source_file.read())

            # print(tree.pretty())
            # print("-" * 80)

            cg = CodeGenerator()
            codegen_res = cg.visit(tree)
            orca_compile(cg, outfilepath=out_path)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    return
