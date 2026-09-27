"""Human-readable projection of network events; never a proof or metric producer."""
import json
import time
import shutil
import sys
from collections import deque


class Console:
    def __init__(self, raw=False):
        self.raw = raw
        self.last_print = 0.0
        self.rate = None
        self.stage_average = None
        self.samples = deque()
        self.rate_at = 0.0
        self.ratio = None
        self.bits = None
        self.started = time.monotonic()
        self.submitted = 0
        self.r_star = None
        self.q7 = None
        self.counts = dict(accepted=0, rejected=0, stale=0, errors=0)
        self.state = 'STARTING'
        self.closed = False
        self.dashboard = None
        self.notice = ''
        self.stop_requested = False
        self.pool_history = deque()
        self.pending = None
        self.prp = None
        self.rolling_rate = None
        self.rolling_seconds = 0.0
        self.connection = 'CONNECTING'
        self.height = None

    def open_dashboard(self, metadata, mode='auto', no_color=False):
        if self.raw or mode=='lines': return
        import sys
        if mode=='auto' and not (sys.stdout.isatty() and sys.stdin.isatty()): return
        try:
            from dashboard import Dashboard
            self.dashboard=Dashboard(self,metadata,no_color)
        except Exception as error:
            if mode=='table': raise ValueError('Table unavailable: '+str(error)+'. Install python3-rich or use --console lines.') from error
            self.emit('Table unavailable; using line output.')

    def close(self):
        if self.dashboard: self.dashboard.close(); self.dashboard=None

    def event(self, event):
        # An allow-list keeps account/proof payloads out of the terminal.
        keys = ('phase', 'terminal_status', 'e2e_c_s', 'prp_per_second',
                'accepted', 'rejected', 'stale', 'errors', 'target_bits', 'r')
        safe = {k: event[k] for k in keys if k in event}
        if self.raw:
            if safe:
                self.emit(json.dumps(safe), structured=True)
            return
        phase = str(event.get('phase', ''))
        if phase=='connected': self.connection='CONNECTED'
        if phase=='authorized': self.connection='AUTHORIZED'
        if phase=='reconnecting': self.connection='RECONNECTING'; self.state='CONNECTING'
        if phase=='session_ended': self.connection='DISCONNECTED'
        if isinstance(event.get('height'),int): self.height=event['height']
        if phase == 'work_dispatched':
            self.rate = self.ratio = self.r_star = self.q7 = None
            self.stage_average = None
            self.rolling_rate = self.prp = None
            self.rolling_seconds = 0.0
            self.samples.clear()
        if isinstance(event.get('target_bits'), int) and event['target_bits'] != self.bits:
            self.bits = event['target_bits']
            self.emit('Network work: {} bits'.format(self.bits))
        ratios = event.get('r_vector')
        if isinstance(ratios, list) and ratios and isinstance(ratios[0], (int,float)):
            self.ratio = ratios[0]
        elif isinstance(event.get('r'), (int,float)):
            self.ratio = event['r']
        if isinstance(event.get('r_star_q7_exact'), (int,float)):
            self.r_star = event['r_star_q7_exact']
        if isinstance(event.get('q7_exact'), int): self.q7 = event['q7_exact']
        # Stage-local zero counters must not overwrite the pool session ledger.
        if 'pending_submissions' in event and 'submitted' in event:
            old=dict(self.counts,sent=self.submitted)
            self.submitted = event['submitted']
            self.pending = event['pending_submissions']
            for key in self.counts:
                if isinstance(event.get(key), int): self.counts[key] = event[key]
            current=dict(self.counts,sent=self.submitted)
            delta={key:max(0,current[key]-old[key]) for key in current}
            if any(delta.values()): self.pool_history.append((time.monotonic(),delta))
        now = time.monotonic()
        while self.pool_history and now-self.pool_history[0][0]>86400:self.pool_history.popleft()
        if isinstance(event.get('prp_s_secondary'),(int,float)):self.prp=event['prp_s_secondary']
        if isinstance(event.get('e2e_c_s'), (int, float)):
            self.stage_average = event['e2e_c_s']
        # Source-time deltas, never the slowly rising cold-start stage average.
        # Keep the sample just before the five-second boundary for an honest
        # approximately five-second window without extrapolated candidates.
        count=event.get('q0_entered'); stamp=event.get('bridge_elapsed_s')
        if isinstance(count,(int,float)) and isinstance(stamp,(int,float)):
            if self.samples and (stamp < self.samples[-1][0] or count < self.samples[-1][1]):
                self.samples.clear(); self.rate=self.rolling_rate=None; self.rolling_seconds=0.0
            if not self.samples or stamp > self.samples[-1][0]:
                self.samples.append((stamp,count))
                while len(self.samples)>2 and self.samples[1][0] <= stamp-60:
                    self.samples.popleft()
                if len(self.samples)>1:
                    recent=0
                    while recent+1 < len(self.samples)-1 and self.samples[recent+1][0] <= stamp-5:
                        recent+=1
                    before,entered=self.samples[recent]
                    self.rate=(count-entered)/(stamp-before)
                    before,entered=self.samples[0]
                    if stamp-before>60:
                        t1,n1=self.samples[1]
                        entered += (n1-entered)*(stamp-60-before)/(t1-before)
                        before=stamp-60
                    self.rolling_seconds=stamp-before
                    self.rolling_rate=(count-entered)/self.rolling_seconds
                    self.rate_at=now
        status = str(event.get('terminal_status', ''))
        important = phase in ('authorized', 'graceful_draining', 'share_rejected',
                              'share_stale', 'session_ended') or status == 'graceful_stopped'
        if phase == 'authorized': self.state = 'CONNECTED'
        if phase in ('work_dispatched', 'stage_started'): self.state = 'PREPARING'
        if self.rate_at == now and self.state != 'DRAINING': self.state = 'MINING'
        if phase == 'graceful_draining': self.state = 'DRAINING'
        if status == 'graceful_stopped': self.state = 'STOPPED'
        if 'error' in phase or status in ('FAIL', 'failed', 'fatal'):
            important = True; self.state = 'ERROR'
        if not important and now-self.last_print < 5: return
        rate = '{:,.0f}'.format(self.rate) if self.rate is not None and now-self.rate_at <= 10 else '--'
        ratio = '{:.2f}'.format(self.ratio) if self.ratio is not None and now-self.rate_at <= 10 else '--'
        elapsed=int(now-self.started)
        prefix='[{:02d}:{:02d}:{:02d}] '.format(elapsed//3600,elapsed//60%60,elapsed%60)
        fields=[self.state, 'Speed '+rate+' c/s', 'r'+ratio,
                'Sh {}/{}'.format(self.counts['accepted'], self.submitted),
                'R/S/E {}/{}/{}'.format(self.counts['rejected'],self.counts['stale'],self.counts['errors'])]
        if self.rolling_rate is not None: fields.append('avg60s {:,.0f}'.format(self.rolling_rate))
        if self.bits is not None: fields.append(str(self.bits)+'b')
        if self.r_star is not None: fields.append('r* {:.2f}'.format(self.r_star))
        if self.q7 is not None: fields.append('Q7 '+str(self.q7))
        width=max(40,shutil.get_terminal_size((110,25)).columns-1)
        while len(prefix+' '.join(fields))>width and len(fields)>6:fields.pop()
        self.emit(prefix+' '.join(fields))
        self.last_print = now

    def tick(self):
        if self.dashboard:self.dashboard.draw()
        if not self.raw and time.monotonic()-self.last_print >= 5:
            self.event({})

    def emit(self, text, structured=False):
        if self.closed: return
        if self.dashboard:
            if not text.startswith('['): self.notice=text
            self.dashboard.draw()
            return
        try: print(text, flush=True, file=sys.stderr if self.raw and not structured else sys.stdout)
        except (BrokenPipeError, OSError): self.closed = True
