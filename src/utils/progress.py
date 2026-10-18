"""Barra de progresso padronizada (``rich.progress``)."""

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress() -> Progress:
    """Cria uma barra de progresso com as colunas padrão do projeto.

    Returns
    -------
    Progress
        Instância de ``rich.progress.Progress`` pronta para uso como
        gerenciador de contexto.

    Examples
    --------
    >>> with create_progress() as progress:
    ...     task = progress.add_task("Processando", total=3)
    ...     progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
