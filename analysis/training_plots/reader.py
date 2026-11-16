"""Incremental, explicit-file JSONL reader. No ML imports or directory discovery."""
import hashlib
import json
import math
from pathlib import Path


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


class Reader:
    def __init__(self, label, path):
        self.label, self.path = label, Path(path).resolve()
        self.identity = None
        self.offset = 0
        self.pending = b''
        self.anchor = b''
        self.line = 0
        self.segment = 0
        self.records = []
        self.events = []
        self.digest = hashlib.sha256()
        self.generations = []

    def event(self, kind, **details):
        self.events.append(dict(kind=kind, segment=self.segment, line=self.line, **details))

    def poll(self):
        # Open then fstat, so a concurrent rename cannot mismatch identity and contents.
        try:
            stream = self.path.open('rb')
        except FileNotFoundError:
            if not self.events or self.events[-1]['kind'] != 'missing_file':
                self.event('missing_file')
            return
        import os
        with stream:
            stat = os.fstat(stream.fileno())
            identity = (stat.st_dev, stat.st_ino)
            stream.seek(max(0, self.offset - len(self.anchor)))
            changed = stream.read(len(self.anchor)) != self.anchor
            if self.identity is not None and (identity != self.identity or stat.st_size < self.offset or changed):
                self.generations.append(dict(bytes=self.offset, sha256=self.digest.hexdigest()))
                self.segment += 1
                self.event('file_replaced_or_truncated', discarded_partial_bytes=len(self.pending))
                self.offset, self.line, self.pending = 0, 0, b''
                self.digest = hashlib.sha256()
            self.identity = identity
            stream.seek(self.offset)
            # Snapshot at fstat length; later writes are consumed on the next poll.
            chunk = stream.read(max(0, stat.st_size - self.offset))
            self.offset += len(chunk)
            self.digest.update(chunk)
            stream.seek(max(0, self.offset - 4096))
            self.anchor = stream.read(min(4096, self.offset))
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()
        for line in lines:
            self.line += 1
            try:
                obj = json.loads(line)
                step, data = obj['step'], obj['data']
                if not numeric(step) or step < 0 or int(step) != step or not isinstance(data, dict):
                    raise ValueError('expected nonnegative integer step and data object')
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                self.event('parse_error', error=str(exc))
                continue
            if self.records and self.records[-1]['segment'] == self.segment and step < self.records[-1]['step']:
                self.segment += 1
                self.event('step_rollback', step=step)
            if not self.records or (self.records[-1]['segment'], self.records[-1]['step']) != (self.segment, step):
                self.records.append(dict(segment=self.segment, step=step, data={}))
            target = self.records[-1]['data']
            for key, value in data.items():
                if key in target and target[key] != value:
                    self.event('field_conflict', step=step, metric=key, previous=target[key], last=value)
                target[key] = value

    def audit(self):
        return dict(label=self.label, source=str(self.path), bytes_read=self.offset,
                    sha256=self.digest.hexdigest(), hash_scope='bytes consumed, including deferred last line',
                    previous_generations=self.generations, pending_bytes=len(self.pending),
                    records=len(self.records), segments=len({r['segment'] for r in self.records}), events=self.events)
