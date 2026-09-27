"""Rich terminal presentation; never owns mining, proofs or pool counters."""
import atexit
import os
import select
import sys
import time

from rich import box
from rich.console import Console as RichConsole, Group
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

MUTED = 'bright_black'


def safe(value):
    return ''.join(c for c in str(value) if c.isprintable())[:240]


def number(value):
    return '—' if value is None else '{:,.0f}'.format(value)


def pool_values(model, key, now):
    elapsed = max(0, now-model.started)
    values = [str(dict(model.counts, sent=model.submitted)[key])]
    for seconds in (300, 3600, 86400):
        count = sum(delta.get(key, 0) for stamp, delta in model.pool_history if 0 <= now-stamp <= seconds)
        values.append(str(count)+('*' if elapsed < seconds else ''))
    return values


def metric(title, value, detail):
    return Panel(Group(Text(value, style='bold cyan', justify='center'),
                       Text(detail, style=MUTED, justify='center')),
                 title=Text(title, style='bold white'), title_align='left',
                 border_style=MUTED, box=box.ROUNDED, padding=(0, 1))


def render(model, metadata, width=100, height=30, now=None):
    """Pure bounded Rich layout: independently testable without GPU or terminal."""
    now = time.monotonic() if now is None else now
    elapsed = max(0, int(now-model.started))
    duration = '{:02d}:{:02d}:{:02d}'.format(elapsed//3600, elapsed//60 % 60, elapsed % 60)
    fresh = model.rate is not None and now-model.rate_at <= 10 and model.state == 'MINING'
    color = 'green' if model.state == 'MINING' else 'red' if model.state == 'ERROR' else 'yellow'
    title = Text.assemble(('HORIZON', 'bold cyan'), ('  /  RIECOIN GPU', 'bold white'),
                          ('   SILVER R346', MUTED))
    status = Text.assemble((safe(model.state), 'bold '+color), ('   '+duration, 'white'))
    header = Table.grid(expand=True)
    header.add_column(ratio=1); header.add_column(justify='right')
    header.add_row(title, status)
    device = Text.assemble((safe(metadata.get('gpu', '—')), 'white'),
                          ('  |  '+safe(metadata.get('pool', '—')), MUTED))
    intro = Panel(Group(header, device), border_style='cyan', box=box.ROUNDED, padding=(0, 1))
    if width < 66 or height < 24:
        return Group(intro, Panel(Text('Enlarge the terminal to 66 × 24 or more.\nMining continues. Q / Ctrl+C requests a safe stop.'),
                                  title='Terminal size', border_style='yellow'))
    span = model.rolling_seconds
    detail = 'Last 60 s' if span >= 60 else 'Observed {:.0f} s / 60 s'.format(span)
    cards = [metric('Speed', number(model.rate if fresh else None)+' c/s', 'Candidates per second'),
             metric('Average · 60 s', number(model.rolling_rate if fresh else None)+' c/s', detail),
             metric('PRP/s', number(model.prp if fresh else None), 'Q0 · GPU-time average'),
             metric('r · Q0 / Q1', '{:.2f}'.format(model.ratio) if fresh and model.ratio is not None else '—',
                    'Current work · {} bits'.format(model.bits or '—'))]
    metrics = Table.grid(expand=True, padding=(0, 1))
    columns = 4 if width >= 118 else 2
    for _ in range(columns): metrics.add_column(ratio=1)
    for i in range(0, 4, columns): metrics.add_row(*cards[i:i+columns])
    results = Table(box=box.SIMPLE, expand=True, header_style='bold cyan', border_style=MUTED,
                    padding=(0, 1), show_edge=False)
    results.add_column('Pool results', ratio=2, min_width=12)
    for label in (('This run', 'Last 5 min', 'Last 1 h', 'Last 24 h') if width >= 78 else ('This run', '5 min', '1 h', '24 h')):
        results.add_column(label, justify='right', ratio=1, min_width=9, no_wrap=True)
    for key, label in [('accepted','Accepted'), ('sent','Sent'), ('rejected','Rejected'),
                       ('stale','Stale'), ('errors','Errors')]:
        count = dict(model.counts, sent=model.submitted)[key]
        style = 'green' if key == 'accepted' else 'red' if key in ('rejected','errors') and count else 'yellow' if key == 'stale' and count else 'white'
        results.add_row(Text(label, style=style), *(Text(v, style=style) for v in pool_values(model,key,now)))
    extra = 'Pending response  '+number(model.pending)
    if model.r_star is not None or model.q7 is not None:
        extra += '   |   Exact Q7 '+number(model.q7)+'   r* '+number(model.r_star)
    pool_footer = Text(extra+'\n* Partial observation; no extrapolation.', style=MUTED)
    pool_panel = Panel(Group(results, pool_footer), title=Text('SHARES', style='bold white'),
                       title_align='left', border_style=MUTED, box=box.ROUNDED, padding=(0,1))
    footer = Text.assemble((' Q / Ctrl+C ', 'bold cyan'), ('Stop safely', 'white'))
    layout = Layout()
    sections=[Layout(intro,size=4), Layout(metrics,size=4 if columns==4 else 8)]
    if height >= (26 if columns==4 else 30):
        details=Table.grid(expand=True,padding=(0,2))
        details.add_column(ratio=1);details.add_column(ratio=1)
        details.add_row(Text('Pool  '+safe(metadata.get('pool','—'))),Text('Connection  '+model.connection,style='green' if model.connection=='AUTHORIZED' else 'yellow'))
        details.add_row(Text('Block height  '+number(model.height)),Text('Target  {} bits'.format(model.bits or '—')))
        details.add_row(Text('GPU  '+safe(metadata.get('gpu','—'))),Text('CUDA capability  '+safe(metadata.get('compute','—'))))
        sections.append(Layout(Panel(details,title='NETWORK & DEVICE',title_align='left',border_style=MUTED,box=box.ROUNDED),size=5))
    sections.extend([Layout(pool_panel,minimum_size=11),Layout(footer,size=1)])
    layout.split_column(*sections)
    return layout


class Dashboard:
    def __init__(self, model, metadata, no_color=False):
        if not sys.stdout.isatty() or not sys.stdin.isatty() or os.environ.get('TERM','dumb') == 'dumb':
            raise RuntimeError('Table mode requires a terminal; use --console lines.')
        import termios
        import tty
        self.model, self.metadata = model, metadata
        self.saved = None
        self.live = None
        self.last = 0.0
        self.size = None
        self.termios = termios
        self.fd = sys.stdin.fileno()
        self.console = RichConsole(no_color=no_color, highlight=False)
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.live = Live(console=self.console, screen=True, auto_refresh=False,
                             redirect_stdout=False, redirect_stderr=False, vertical_overflow='crop')
            self.live.start()
            atexit.register(self.close)
            self.draw(force=True)
        except Exception:
            self.close()
            raise

    def draw(self, force=False):
        if self.live is None: return
        try:
            if select.select([self.fd], [], [], 0)[0]:
                key = os.read(self.fd, 32)
                if not key: self.model.closed = True
                if b'q' in key.lower(): self.model.stop_requested = True
            now = time.monotonic()
            size = self.console.size
            if not force and size == self.size and now-self.last < 1: return
            self.live.update(render(self.model, self.metadata, size.width, size.height, now), refresh=True)
            self.size, self.last = size, now
        except (OSError, EOFError):
            self.model.closed = True

    def close(self):
        try:
            if self.live is not None:
                live, self.live = self.live, None
                live.stop()
        finally:
            if self.saved is not None:
                saved, self.saved = self.saved, None
                try: self.termios.tcsetattr(self.fd, self.termios.TCSANOW, saved)
                except (OSError, self.termios.error): pass
