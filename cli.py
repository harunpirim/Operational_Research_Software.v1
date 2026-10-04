"""
OR Assistant - Command Line Interface
"""

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """OR Assistant - AI-Powered Operations Research Tool"""
    pass


_SOLVER_ALIASES = {
    'pulp': 'pulp_cbc', 'cbc': 'pulp_cbc', 'ortools': 'pulp_cbc', 'or-tools': 'pulp_cbc',
    'cvxpy': 'cvxpy_osqp', 'osqp': 'cvxpy_osqp', 'scip': 'cvxpy_scip', 'glpk': 'cvxpy_glpk',
}


@cli.command()
@click.option('--problem', '-p', help='Problem description (or use --file)')
@click.option('--file', '-f', type=click.Path(exists=True),
              help='Problem description text file, or an .mps/.mps.gz model')
@click.option('--solver', '-s', default='auto', show_default=True,
              help='auto, pulp_cbc, cvxpy_osqp, cvxpy_scip, cvxpy_glpk (aliases: pulp, cvxpy, scip, glpk)')
@click.option('--time-limit', '-t', type=int, default=None, help='Solver time limit in seconds')
@click.option('--output', '-o', type=click.Path(), help='Save the solution as JSON')
@click.option('--verbose', '-v', is_flag=True, help='Show all variable values')
def solve(problem, file, solver, time_limit, output, verbose):
    """Solve an optimization problem"""
    from src.modeling.model_generator import ModelGenerator
    from src.solvers.solver_interface import SolverInterface
    from src.solvers.solver_router import resolve_solver

    is_mps = bool(file) and file.lower().endswith(('.mps', '.mps.gz'))
    if not (file or problem):
        console.print("[red]Error: Provide problem via --problem or --file[/red]")
        raise SystemExit(1)

    solver_key = _SOLVER_ALIASES.get(solver.lower(), solver.lower())
    console.print("[bold blue]OR Assistant[/bold blue]\n")

    try:
        with console.status("[cyan]Building model...") as status:
            generator = ModelGenerator()
            if is_mps:
                from src.ingestion.file_parser import FileParser
                parsed = FileParser().parse(file, filename=os.path.basename(file))
                parsed.setdefault('file_path', file)
                pref = 'cvxpy' if solver_key.startswith('cvxpy') else 'pulp'
                model, problem_data = generator.generate_from_mps(parsed, solver_preference=pref)
            else:
                from src.agents.problem_classifier import ProblemClassifier
                problem_text = open(file).read() if file else problem
                status.update("[cyan]Classifying problem...")
                problem_data = ProblemClassifier().classify(problem_text)
                console.print(
                    f"Problem type: [cyan]{problem_data.get('problem_type')}[/cyan] "
                    f"(confidence {problem_data.get('confidence', 0):.0%})"
                )
                status.update("[cyan]Generating model...")
                pref = 'cvxpy' if solver_key.startswith('cvxpy') else 'pulp'
                model = generator.generate(problem_data, solver_preference=pref)

            resolved_key, explanation = resolve_solver(solver_key, problem_data)
            console.print(f"[dim]{explanation}[/dim]")
            status.update("[cyan]Solving...")
            solution = SolverInterface(resolved_key, problem_data=problem_data).solve(
                model, max_time=time_limit,
            )

            interpretation = None
            if not is_mps:
                status.update("[cyan]Interpreting results...")
                try:
                    from src.interpreters.result_interpreter import ResultInterpreter
                    interpretation = ResultInterpreter().interpret(solution, problem_data)
                except Exception as e:
                    console.print(f"[yellow]Interpretation skipped: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1)

    ok = solution.get('is_optimal')
    console.print(
        f"\n[bold {'green' if ok else 'yellow'}]{'✓' if ok else '!'} "
        f"{solution.get('status')}[/bold {'green' if ok else 'yellow'}]\n"
    )

    table = Table(title="Solution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    obj = solution.get('objective_value')
    table.add_row("Objective Value", f"{obj:,.4g}" if obj is not None else "N/A")
    table.add_row("Status", str(solution.get('status')))
    table.add_row("Solver", str(solution.get('solver_name')))
    table.add_row("Variables / Constraints",
                  f"{solution.get('num_variables')} / {solution.get('num_constraints')}")
    table.add_row("Solve Time", f"{solution.get('solve_time', 0):.3f}s")
    console.print(table)

    variables = solution.get('variables') or {}
    shown = variables if verbose else {
        k: v for k, v in variables.items() if v is not None and abs(v) > 1e-9
    }
    if shown:
        var_table = Table(title="Variables" if verbose else "Non-zero Variables")
        var_table.add_column("Name", style="cyan")
        var_table.add_column("Value", style="green", justify="right")
        for name, value in list(shown.items())[:50]:
            var_table.add_row(name, "—" if value is None else f"{value:,.4g}")
        if len(shown) > 50:
            var_table.caption = f"showing 50 of {len(shown)}"
        console.print(var_table)

    for w in solution.get('warnings') or []:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    if solution.get('error_message'):
        console.print(f"[red]{solution['error_message']}[/red]")

    if interpretation and interpretation.get('summary'):
        console.print(f"\n[bold]Interpretation[/bold]\n{interpretation['summary']}")
        for finding in interpretation.get('key_findings') or []:
            console.print(f"  • {finding}")

    if output:
        from src.utils import save_results
        save_results(
            {'problem': problem_data, 'solution': solution, 'interpretation': interpretation},
            output,
        )
        console.print(f"\n[dim]Results saved to {output}[/dim]")

    if not ok:
        raise SystemExit(2)


@cli.command()
@click.argument('category', required=False)
def examples(category):
    """List example problems"""
    
    console.print("[bold blue]Example Problems[/bold blue]\n")
    
    examples_list = {
        "Linear Programming": [
            "Production planning with resource constraints",
            "Diet optimization problem",
            "Portfolio optimization"
        ],
        "Integer Programming": [
            "Facility location problem",
            "Capital budgeting",
            "Lot sizing"
        ],
        "Transportation": [
            "Warehouse to store shipping",
            "Supply chain optimization"
        ],
        "Assignment": [
            "Task assignment to workers",
            "Machine scheduling"
        ]
    }
    
    if category:
        if category in examples_list:
            console.print(f"[cyan]{category}:[/cyan]")
            for ex in examples_list[category]:
                console.print(f"  • {ex}")
        else:
            console.print(f"[red]Category '{category}' not found[/red]")
    else:
        for cat, exs in examples_list.items():
            console.print(f"[cyan]{cat}:[/cyan]")
            for ex in exs:
                console.print(f"  • {ex}")
            console.print()


@cli.command()
def solvers():
    """List available solvers"""
    
    table = Table(title="Available Solvers")
    table.add_column("Solver", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Status", style="green")
    
    from src.solvers.solver_router import _get_available_solvers
    from src.solvers.solver_interface import _cbc_runs

    available = _get_available_solvers()
    cbc = _cbc_runs()
    rows = [
        ("PuLP / CBC", "LP/MIP", cbc, "" if cbc else "falls back to HiGHS"),
        ("CVXPY / OSQP", "LP/QP", 'cvxpy_osqp' in available, ""),
        ("CVXPY / SCIP", "MIP", 'cvxpy_scip' in available, "pip install pyscipopt"),
        ("CVXPY / GLPK", "MIP", 'cvxpy_glpk' in available, "pip install cvxopt"),
    ]
    for name, kind, ok, hint in rows:
        status = "[green]✓ Available[/green]" if ok else f"[red]✗ Not available[/red] {hint}"
        table.add_row(name, kind, status)

    console.print(table)


@cli.command()
def config():
    """Show configuration"""
    
    console.print("[bold blue]Configuration[/bold blue]\n")
    
    # Check for API keys
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    ai_provider = os.getenv('AI_PROVIDER', 'auto-detect')
    
    if anthropic_key:
        console.print(f"[green]✓[/green] Anthropic API Key: {'*' * 20}{anthropic_key[-8:]}")
    else:
        console.print("[red]✗ Anthropic API Key: Not configured[/red]")
    
    if openai_key:
        console.print(f"[green]✓[/green] OpenAI API Key: {'*' * 20}{openai_key[-8:]}")
    else:
        console.print("[red]✗ OpenAI API Key: Not configured[/red]")
    
    console.print(f"AI Provider: {ai_provider}")
    console.print(f"Default Model: {os.getenv('DEFAULT_MODEL', 'auto')}")
    console.print(f"Default Solver: {os.getenv('DEFAULT_SOLVER', 'pulp')}")
    console.print(f"Debug Mode: {os.getenv('DEBUG', 'False')}")


@cli.command()
@click.argument('key')
@click.argument('value')
def set_config(key, value):
    """Set configuration value"""
    
    # Update .env file
    env_file = '.env'
    
    if not os.path.exists(env_file):
        console.print("[yellow]Creating .env file...[/yellow]")
        with open('.env.example', 'r') as src:
            with open('.env', 'w') as dst:
                dst.write(src.read())
    
    console.print(f"[green]✓ Set {key} = {value}[/green]")
    console.print("[dim]Restart the application for changes to take effect[/dim]")


if __name__ == '__main__':
    cli()
