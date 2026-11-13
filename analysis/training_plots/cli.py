"""Offline Plotly dashboards and Matplotlib exports; explicit inputs only."""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
from datetime import datetime, timezone

from reader import Reader, numeric
from metrics import EXPECTED, GROUPS, group, spec, value

NOTICE = ('Raw curves; gaps mean unavailable, never zero-filled. Online validation is not offline Avg@32/pass@32. '
          'Elapsed axis sums recorded timing_s/step only, excluding queue/startup/unrecorded time, reset per segment. '
          'Timing stages may overlap. Memory retains native aggregation; MFU is not GPU utilization. '
          '未记录不等于未启用；条件比例无分母时不可用。')


def atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload.encode('utf-8') if isinstance(payload, str) else payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def average(values, window=10):
    return [sum(values[i-window+1:i+1])/window
            if i >= window-1 and all(v is not None for v in values[i-window+1:i+1]) else None
            for i in range(len(values))]


def normalize(readers, keys):
    rows = []
    for reader in readers:
        elapsed = 0
        segment = None
        for record in reader.records:
            if segment != record['segment']:
                elapsed, segment = 0, record['segment']
            duration = record['data'].get('timing_s/step')
            valid_duration = numeric(duration) and duration >= 0
            if valid_duration: elapsed += duration
            for key in keys:
                val, status = value(record['data'], key)
                rows.append(dict(run=reader.label, source=str(reader.path), segment=segment,
                                 step=record['step'], recorded_seconds=elapsed,
                                 step_time_recorded=valid_duration, metric=key,
                                 raw_value=record['data'].get(key), value=val, status=status,
                                 unit=spec(key)['unit'], group=GROUPS[group(key)][0]))
    return rows


def series(readers, keys, axis):
    result = {}
    for reader in readers:
        segments = sorted({r['segment'] for r in reader.records})
        for segment in segments:
            records = [r for r in reader.records if r['segment'] == segment]
            xs, elapsed, previous = [], 0, None
            expanded = []
            for record in records:
                # Also break a curve across absent optimizer steps, without fabricating values.
                if previous is not None and record['step'] > previous + 1:
                    xs.append(None)
                    expanded.append(None)
                duration = record['data'].get('timing_s/step')
                if numeric(duration) and duration >= 0: elapsed += duration
                xs.append(record['step'] if axis == 'step' else elapsed)
                expanded.append(record)
                previous = record['step']
            for key in keys:
                ys, statuses = [], []
                for record in expanded:
                    v, status = value(record['data'], key) if record else (None, 'missing step')
                    ys.append(v)
                    statuses.append(status)
                result[(reader.label, segment, key)] = (xs, ys, statuses)
    return result


def render(readers, args, static=True, live=False):
    import plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.offline import get_plotlyjs
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = sorted({key for reader in readers for r in reader.records for key in r['data']})
    # Empty groups still have an explicit missing panel.
    keys += [k for i, k in enumerate(EXPECTED) if not any(group(key) == i for key in keys)]
    keys = sorted(set(keys))
    rows = normalize(readers, keys)
    dictionary = [spec(k) for k in keys]
    table = io.StringIO()
    fields = ['run', 'source', 'segment', 'step', 'recorded_seconds', 'step_time_recorded',
              'metric', 'raw_value', 'value', 'status', 'unit', 'group']
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    out = Path(args.output)
    atomic(out/'metrics.csv', table.getvalue())
    atomic(out/'metric_dictionary.json', json.dumps(dictionary, ensure_ascii=False, indent=2))
    data = series(readers, keys, args.axis)
    fragments = []
    files = ['index.html', 'metrics.csv', 'metric_dictionary.json', 'audit.json']
    xlabel = 'Training step' if args.axis == 'step' else 'Cumulative recorded step time (s; excludes unrecorded time)'
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    aliases = args.names

    def display(key):
        parts = key.split('/')
        if key.startswith('val-') and len(parts) > 2: parts[1] = aliases.get(parts[1], parts[1])
        return '/'.join(parts)

    for index, (slug, title) in enumerate(GROUPS):
        selected = [k for k in keys if group(k) == index]
        cols = 2
        nrows = math.ceil(len(selected)/cols)
        titles = [html.escape(display(k)) for k in selected]
        fig = make_subplots(rows=nrows, cols=cols, subplot_titles=titles,
                            vertical_spacing=min(.06, .5/max(nrows, 1)))
        if static:
            mpl, axes = plt.subplots(nrows, cols, figsize=(18, 3.5*nrows), squeeze=False)
        missing = []
        for j, key in enumerate(selected):
            row, col = j//cols+1, j%cols+1
            found = False
            for ri, reader in enumerate(readers):
                for (label, segment, metric), (xs, ys, statuses) in data.items():
                    if label != reader.label or metric != key: continue
                    name = f'{label} / segment {segment}'
                    color = colors[ri % len(colors)]
                    found |= any(y is not None for y in ys)
                    fig.add_trace(go.Scatter(x=xs, y=ys, name=html.escape(name),
                        legendgroup=name, showlegend=j == 0, mode='lines+markers',
                        marker=dict(size=3), line=dict(color=color, dash='solid' if segment % 2 == 0 else 'dash'),
                        connectgaps=False, text=statuses,
                        hovertemplate='%{x}<br>%{y}<br>%{text}<extra>%{fullData.name}</extra>'), row=row, col=col)
                    if args.smooth:
                        fig.add_trace(go.Scatter(x=xs, y=average(ys), name=html.escape(name)+' MA10',
                            legendgroup=name+' MA10', showlegend=j == 0, connectgaps=False,
                            line=dict(color=color, dash='dot'), mode='lines'), row=row, col=col)
                    if static:
                        ax = axes[row-1][col-1]
                        mx = [float('nan') if x is None else x for x in xs]
                        my = [float('nan') if y is None else y for y in ys]
                        ax.plot(mx, my, label=name, color=color, linewidth=.9,
                                linestyle='-' if segment % 2 == 0 else '--', marker='.', markersize=2)
                        if args.smooth:
                            ax.plot(mx, [float('nan') if y is None else y for y in average(ys)],
                                    color=color, linestyle=':', label=name+' MA10')
            if not found: missing.append(key)
            fig.update_xaxes(title_text=xlabel, row=row, col=col)
            fig.update_yaxes(title_text=spec(key)['unit'], row=row, col=col)
            if not found:
                fig.add_annotation(text='未记录或不可用 / unavailable', x=.5, y=.5,
                                   xref=f'x{j+1 if j else ""} domain', yref=f'y{j+1 if j else ""} domain', showarrow=False)
            if static:
                ax = axes[row-1][col-1]
                # Raw names keep static output portable on hosts without Chinese fonts.
                ax.set_title(key, fontsize=8, wrap=True)
                ax.set_xlabel(xlabel, fontsize=8)
                ax.set_ylabel(spec(key)['unit'], fontsize=8)
                ax.grid(alpha=.2)
                if not found: ax.text(.5, .5, 'Not recorded / unavailable', transform=ax.transAxes, ha='center')
                handles, labels = ax.get_legend_handles_labels()
                if handles: ax.legend(fontsize=6)
        fig.update_layout(height=max(370, 290*nrows), title=title, template='plotly_white',
                          margin=dict(t=90), legend=dict(orientation='h', y=1.04))
        fragments.append('<section id="'+slug+'"><h2>'+html.escape(title)+'</h2>')
        fragments.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=slug,
                                     config={'responsive': True, 'displaylogo': False}))
        if missing: fragments.append('<p>未记录或不可用：'+html.escape(', '.join(missing))+'</p>')
        fragments.append('</section>')
        if static:
            for j in range(len(selected), nrows*cols): axes[j//cols][j%cols].set_visible(False)
            mpl.suptitle(title.split(' / ')[-1]+' | raw curves; gaps unavailable', fontsize=13)
            mpl.tight_layout(rect=(0, 0, 1, .98))
            for extension in ('png', 'pdf'):
                buffer = io.BytesIO()
                mpl.savefig(buffer, format=extension, dpi=110)
                atomic(out/(slug+'.'+extension), buffer.getvalue())
                files.append(slug+'.'+extension)
            plt.close(mpl)
    refresh = ''
    if live:
        refresh = ('<script>setTimeout(()=>{sessionStorage.setItem("plotScroll",scrollY);location.reload()},'
                   + str(int(args.interval*1000))+');window.addEventListener("load",()=>scrollTo(0,Number(sessionStorage.getItem("plotScroll")||0)));</script>')
    # Includes the entire JS bundle: no CDN, network requests or external assets.
    content = ('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Training dynamics</title>'
               '<style>body{font:15px system-ui;margin:24px;color:#182333}nav a{margin-right:14px}section{border-top:1px solid #ddd}details{margin:12px 0}pre{white-space:pre-wrap}</style>'
               '<script>'+get_plotlyjs()+'</script><h1>训练动态图表 / Training dynamics</h1><p>'+html.escape(NOTICE)+'</p>'
               '<p>刷新模式：'+('watch' if live else 'static')+'；生成时间：'+datetime.now(timezone.utc).isoformat()+'</p><nav>'
               + ''.join('<a href="#'+slug+'">'+html.escape(title)+'</a>' for slug,title in GROUPS)+'</nav>'
               '<details><summary>输入审计、缺失值与异常明细</summary><pre>'
               + html.escape(json.dumps([r.audit() for r in readers], ensure_ascii=False, indent=2))
               + '</pre><p>逐实验、逐step状态见 metrics.csv；原始字段及单位见 metric_dictionary.json。</p></details>'
               + ''.join(fragments)+refresh+'</html>')
    atomic(out/'index.html', content)
    try:
        commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), inputs=[r.audit() for r in readers],
                 command_options=vars(args), git_commit=commit,
                 code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                 versions=dict(plotly=plotly.__version__, matplotlib=matplotlib.__version__),
                 notice=NOTICE, exports=files, static_exported_this_refresh=static)
    atomic(out/'audit.json', json.dumps(audit, ensure_ascii=False, indent=2))
    print(f'Updated {out}/index.html ({len(rows)} metric rows; static={static})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['render', 'watch'])
    parser.add_argument('--run', action='append', required=True, metavar='LABEL=JSONL')
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--axis', choices=['step', 'recorded-time'], default='step')
    parser.add_argument('--smooth', action='store_true', help='Add 10 consecutive recorded-step mean; raw always retained')
    parser.add_argument('--source-names', help='JSON object mapping exact validation source names to display names')
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0: parser.error('--interval must be positive and finite')
    readers = []
    for run in args.run:
        label, sep, path = run.partition('=')
        if not sep or not label or not path: parser.error('--run requires LABEL=PATH')
        if label in [r.label for r in readers]: parser.error('run labels must be unique')
        if not Path(path).is_file(): parser.error(f'input does not exist: {path}')
        readers.append(Reader(label, path))
    args.names = json.loads(Path(args.source_names).read_text()) if args.source_names else {}
    if not isinstance(args.names, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in args.names.items()):
        parser.error('--source-names must contain a string-to-string JSON object')
    out = Path(args.output).resolve()
    # Never overwrite an input with an output artifact.
    if any(r.path.parent == out for r in readers): parser.error('output must be separate from input file directories')
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stopped.set())
    while True:
        for reader in readers: reader.poll()
        render(readers, args, static=args.mode == 'render', live=args.mode == 'watch')
        if args.mode == 'render': break
        if stopped.wait(args.interval):
            for reader in readers: reader.poll()
            render(readers, args, static=True, live=False)
            break


if __name__ == '__main__':
    main()
