"""Reserve actual inputs and outputs and persist operation receipts.

production_state.sqlite3 is authoritative operation state, not a disposable
query cache. Media catalog and manifest rows preserve file observations and
generation-time provenance claims; current Blender asset state is recorded in
datasets/blender_source_registry.sqlite3.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import uuid


_WORKSPACE = Path(__file__).resolve().parents[2]
_MEDIA = _WORKSPACE / 'lab/scripts/media'
if str(_MEDIA) not in sys.path:
    sys.path.insert(0, str(_MEDIA))
import media_index_sqlite as _cache


@dataclass
class ProductionRun:
    reused: bool
    source_refs: dict
    media_observations: dict
    value: object = None
    staged_inputs: dict = field(default_factory=dict)
    _operation: str = ''
    _token: str = ''
    _active: bool = False
    _result_set: bool = False
    _plan: object = None
    _samples: list = field(default_factory=list)
    _workspace: object = None
    _outputs: list = field(default_factory=list)
    _sample_outputs: list = field(default_factory=list)
    _auxiliary_outputs: list = field(default_factory=list)
    _states: dict = field(default_factory=dict)
    _state_counts: dict = field(default_factory=dict)
    _recorded_outputs: set = field(default_factory=set)
    _preserved: dict = field(default_factory=dict)

    @property
    def plan(self):
        """Return a copy so caller changes cannot alter the enforced allocation."""
        return json.loads(json.dumps(self._plan, ensure_ascii=False)) if self._plan is not None else None

    @property
    def active(self):
        """Whether this run may still accept writes or start child operations."""
        return self._active and not self.reused and bool(self._token)

    @property
    def token(self):
        """Run identity while active; callers must capture it before exiting."""
        return self._token if self.active else ''

    def _invalidate(self):
        self._active = False
        self._token = ''

    def record_sample(self, state_id, *, output, sources, parameters, actual):
        """Associate one existing output image with its declared state and inputs."""
        if not self.active or self._plan is None or self._workspace is None:
            raise ValueError('sample recording requires an active planned production run')
        state = self._states.get(state_id)
        if state is None:
            raise ValueError('sample state is not in the production plan')
        try:
            reported = json.loads(json.dumps({'sources': sources, 'parameters': parameters, 'actual': actual},
                                             ensure_ascii=False, allow_nan=False))
            if not isinstance(reported['actual'], dict) or not reported['actual']:
                raise ValueError('sample actual must be a nonempty object')
            expected = {'sources': state['sources'], 'parameters': state['parameters']}
            declared = {key: reported[key] for key in ('sources', 'parameters')}
            if json.dumps(declared, sort_keys=True, ensure_ascii=False) != json.dumps(expected, sort_keys=True, ensure_ascii=False):
                raise ValueError('sample sources or parameters differ from the declared state')
        except (TypeError, OverflowError) as error:
            raise ValueError('sample sources and parameters must be finite JSON') from error
        relative = _paths(self._workspace, [output])[0]
        if relative.casefold() in {path.casefold() for path in self._preserved}:
            raise ValueError('preserved output cannot be recorded as a generated sample')
        path = self._workspace / relative
        if not any(_within(relative, scope) for scope in self._sample_outputs):
            raise ValueError('sample output is outside declared sample output paths')
        if not path.is_file() or path.suffix.lower() not in _cache.core.IMAGES:
            raise ValueError('sample output must be an existing image file')
        if relative in self._recorded_outputs:
            raise ValueError('sample output was already recorded')
        if self._state_counts[state_id] >= state['planned_count']:
            raise ValueError('sample count exceeds the planned state quota')
        stat = path.stat()
        self._samples.append({'state_id': state_id, 'output': relative, **reported,
                              'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns})
        self._state_counts[state_id] += 1
        self._recorded_outputs.add(relative)

    def _check_samples(self, finished):
        if self._plan is None:
            return
        for path, saved in self._preserved.items():
            current = self._workspace / path
            if (current.is_symlink() or not current.is_file() or
                    not current.resolve().is_relative_to(self._workspace) or
                    _file_digest(current) != saved['sha256']):
                raise ValueError('preserved output was removed or changed: ' + path)
        actual = {state['id']: 0 for state in self._plan['strata']}
        for sample in self._samples:
            actual[sample['state_id']] += 1
        if any(actual[state['id']] != state['planned_count'] for state in self._plan['strata']):
            raise ValueError('actual sample counts do not match the production plan')
        all_images = {item['path']: item for item in finished if item.get('type') == 'file'
                      and Path(item['path']).suffix.lower() in _cache.core.IMAGES
                      and item['path'] not in self._preserved}
        if any(not any(_within(path, scope) for scope in self._sample_outputs + self._auxiliary_outputs)
               for path in all_images):
            raise ValueError('output image is outside declared sample and auxiliary paths')
        images = {path: item for path, item in all_images.items()
                  if any(_within(path, scope) for scope in self._sample_outputs)}
        if set(images) != {item['output'] for item in self._samples}:
            raise ValueError('output images do not match recorded samples')
        for sample in self._samples:
            current = images[sample['output']]
            if (sample['bytes'], sample['modified_ns']) != (current['bytes'], current['modified_ns']):
                raise ValueError('sample output changed after recording')

    def set_result(self, value):
        if not self.active:
            raise ValueError('result recording requires an active production run')
        self.value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        self._result_set = True

    def child_env(self):
        if not self.active:
            raise ValueError('a reused or inactive operation cannot start a child')
        env = os.environ.copy()
        env['AEC_PRODUCTION_PARENT_OPERATION'] = self._operation
        env['AEC_PRODUCTION_PARENT_TOKEN'] = self._token
        return env


def _index(root):
    if root is None or Path(root).resolve() == _cache.core.ROOT.resolve():
        return _cache
    # Test workspaces use independent module globals; production index state is untouched.
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    module = load('_production_guard_index_cache', _MEDIA / 'media_index_sqlite.py')
    module.core = load('_production_guard_index', _MEDIA / 'media_index.py')
    module.core.ROOT = Path(root).resolve()
    module.core.INDEX = module.core.ROOT / 'datasets/annotated/media_index'
    module.DATABASE = module.core.INDEX / 'query_cache.sqlite3'
    return module


def receipt_completed(root, operation, token):
    """Read whether a run committed; missing state raises rather than guessing.

    The state store and receipt history must exist. This function never creates,
    migrates, or writes either database, including when the query is negative.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError('operation must be a nonempty string')
    if (not isinstance(token, str) or len(token) != 32 or
            any(c not in '0123456789abcdef' for c in token)):
        raise ValueError('token must be a lowercase 32-character hexadecimal run token')
    state_path = _index(root).DATABASE.parent / 'production_state.sqlite3'
    if not state_path.is_file():
        raise FileNotFoundError('production state is missing: ' + str(state_path))
    with closing(sqlite3.connect(state_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        table = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                           "AND name='production_receipt_history'").fetchone()
        if table is None:
            raise RuntimeError('production receipt history is unavailable')
        return db.execute('SELECT 1 FROM production_receipt_history '
                          'WHERE operation=? AND run_token=? AND origin=?',
                          (operation, token, 'completed_run')).fetchone() is not None


def plan_production(declaration, root=None):
    """Plan before generation using the workspace allocator, including in Blender.

    A plan alone neither reserves inputs nor proves generation followed it;
    generated samples must still use ``generation_guard(production_plan=...)``.
    """
    root = _WORKSPACE if root is None else root
    if importlib.util.find_spec('scipy') is not None:
        from lab.training_common.bayesian_allocation import plan_generation
        return plan_generation(declaration, root)
    interpreter = _WORKSPACE / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not interpreter.is_file():
        raise RuntimeError('SciPy is unavailable and the workspace .venv interpreter is missing')
    try:
        payload = json.dumps(declaration, ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as error:
        raise ValueError('generation declaration must be finite JSON') from error
    if len(payload) > 64 * 1024 * 1024:
        raise ValueError('generation declaration exceeds the input size bound')
    env = os.environ.copy()
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    try:
        completed = subprocess.run(
            [str(interpreter), '-I', '-B', str(Path(__file__).with_name('bayesian_allocation.py')), '--plan-stdin'],
            input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=60,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError('workspace Python allocation timed out') from error
    if completed.returncode:
        raise ValueError('workspace Python allocation failed: ' + completed.stderr.decode('utf-8', errors='replace')[-4000:])
    try:
        return json.loads(completed.stdout.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError('workspace Python returned an invalid allocation') from error


def _paths(root, paths):
    result = []
    for value in paths:
        path = Path(value)
        path = path if path.is_absolute() else root / path
        if path.is_symlink():
            raise ValueError('symlink input/output is not supported: ' + str(path))
        path = path.resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError('path must be within the workspace: ' + str(path))
        result.append(path.relative_to(root).as_posix())
    if len(result) != len(set(result)):
        raise ValueError('duplicate paths')
    return sorted(result)


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _inventory(root, paths, require, *, nonempty=False):
    result = []
    for relative in paths:
        path = root / relative
        if not path.exists():
            if require:
                raise ValueError('required path is missing: ' + relative)
            result.append({'path': relative, 'missing': True})
            continue
        if path.is_symlink():
            raise ValueError('symlink is not supported: ' + relative)
        entries = [path]
        if path.is_dir():
            entries += sorted(path.rglob('*'))
            if nonempty and not any(child.is_file() for child in entries[1:]):
                raise ValueError('output directory contains no files: ' + relative)
        for child in entries:
            if child.is_symlink() or not child.resolve().is_relative_to(root):
                raise ValueError('symlink or external path is not supported: ' + str(child))
            stat = child.stat()
            result.append({'path': child.relative_to(root).as_posix(),
                           'type': 'directory' if child.is_dir() else 'file',
                           'bytes': stat.st_size, 'modified_ns': stat.st_mtime_ns})
    return result


def _sources(cache, paths, states):
    refs = {path: [] for path in paths}
    observations = {path: {key: states[path][key] for key in ('path', 'bytes', 'modified_ns')}
                    for path in paths}
    try:
        with closing(cache.connect(readonly=True)) as db:
            cache.validated(db, True)
            refs = _indexed_refs(cache, db, paths, observations)
    except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.Error):
        pass
    return refs, observations


def _indexed_refs(cache, db, paths, current_states):
    core = cache.core
    refs = {path: [] for path in paths}
    by_key = {cache.path_key(path): path for path in paths}
    if len(by_key) != len(paths):
        raise ValueError('media paths differ only by filesystem case')
    current = set()
    keys = list(by_key)
    for offset in range(0, len(keys), 400):
        chunk = keys[offset:offset + 400]
        placeholders = ','.join('?' for _ in chunk)
        for key, raw in db.execute('SELECT path_key,item_json FROM media '
                                   f'WHERE path_key IN ({placeholders})', chunk):
            item = json.loads(raw)
            state = current_states[by_key[key]]
            if all(item.get(field) == state[field] for field in ('bytes', 'modified_ns')):
                current.add(by_key[key])
        rows = db.execute('SELECT e.path_key,e.entry_json,p.descriptor,r.shared_json FROM evidence e '
                          'JOIN partitions p ON p.name=e.partition_name '
                          'LEFT JOIN manifest_rows r ON r.id=e.manifest_row_id '
                          'WHERE e.path_key IN (' + placeholders + ') ORDER BY e.id', chunk)
        for key, raw, descriptor, shared in rows:
            part = json.loads(descriptor)
            if not part['source_present'] or core.lifecycle_reason(part['name']) or by_key[key] not in current:
                continue
            entry = json.loads(raw)
            if shared is not None:
                entry['evidence'][0].update(json.loads(shared))
            for record in entry['evidence']:
                if record.get('relation') == 'batch_context':
                    continue
                refs[by_key[key]].append({k: record[k] for k in
                    ('manifest', 'record_index', 'relation', 'reference_field', 'subject_paths') if k in record})
    return refs


def _overlap(a, b):
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')



def _within(path, root):
    return path == root or path.startswith(root + '/')


def _outside_inplace(inventory, inplace):
    return [item for item in inventory if not any(
        _within(item['path'], path) or
        (item.get('type') == 'directory' and _within(path, item['path']))
        for path in inplace)]


def _alive(cache, reservation):
    if reservation['host'] != socket.gethostname():
        return True
    return cache.core.process_exists(reservation['pid']) is not False


def _legacy_receipts(cache):
    try:
        with closing(cache.connect(readonly=True)) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                          "AND name='production_operations'").fetchone():
                return list(db.execute('SELECT operation,receipt_json,running_json '
                                       'FROM production_operations ORDER BY operation'))
    except (OSError, ValueError, sqlite3.Error):
        pass
    return None


def _prepare_state(db, cache):
    """Import available legacy receipts once, leaving their source untouched."""
    try:
        db.execute('CREATE TABLE IF NOT EXISTS production_operations ('
                   'operation TEXT PRIMARY KEY, receipt_json TEXT, running_json TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS migrations '
                   '(name TEXT PRIMARY KEY, legacy_rows_json TEXT NOT NULL)')
        history_exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                    "AND name='production_receipt_history'").fetchone()
        db.execute('CREATE TABLE IF NOT EXISTS production_receipt_history ('
                   'id INTEGER PRIMARY KEY, operation TEXT NOT NULL, run_token TEXT UNIQUE, '
                   'origin TEXT NOT NULL, receipt_json TEXT NOT NULL)')
        marker = db.execute("SELECT 1 FROM migrations WHERE name='cache_operations_v1'").fetchone()
        rows = _legacy_receipts(cache) if marker is None else None
        if rows is not None:
            encoded = json.dumps(rows, ensure_ascii=False, separators=(',', ':'))
            for operation, receipt, running in rows:
                inserted = db.execute('INSERT OR IGNORE INTO production_operations '
                                      '(operation,receipt_json,running_json) VALUES(?,?,?)',
                                      (operation, receipt, running)).rowcount
                if history_exists and inserted and receipt is not None:
                    db.execute('INSERT INTO production_receipt_history '
                               '(operation,run_token,origin,receipt_json) VALUES(?,?,?,?)',
                               (operation, None, 'existing_snapshot', receipt))
            db.execute('INSERT INTO migrations VALUES(?,?)', ('cache_operations_v1', encoded))
        if not history_exists:
            # The old schema contains only the latest receipt. Preserve exactly
            # that known snapshot without inventing its original run or date.
            db.execute('INSERT INTO production_receipt_history '
                       '(operation,run_token,origin,receipt_json) '
                       "SELECT operation,NULL,'existing_snapshot',receipt_json "
                       'FROM production_operations WHERE receipt_json IS NOT NULL')
        db.commit()
    except BaseException:
        db.rollback()
        raise


@contextmanager
def _transaction(cache):
    state_path = cache.DATABASE.parent / 'production_state.sqlite3'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state_path, timeout=60)) as db:
        db.execute('BEGIN IMMEDIATE')
        _prepare_state(db, cache)
        db.execute('BEGIN IMMEDIATE')
        with db:
            yield db


@contextmanager
def _generation_guard(operation, input_paths, output_paths, parameters, *, code_paths=(),
                      procedural_inputs=(), inplace_paths=(), adopt_outputs=(), dependency_paths=(), root=None,
                      plan=None, declaration=None, sample_output_paths=(), auxiliary_output_paths=(),
                      replace_outputs=False, replace_verifier=None):
    """Yield a run; honor ``run.reused`` before invoking the wrapped operation.

    Procedural/configuration inputs must be explicitly named and are not given
    invented media claims. Completion is recorded only after successful exit.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError('operation must be a nonempty string')
    cache = _index(root)
    workspace = cache.core.ROOT.resolve()
    inputs = _paths(workspace, input_paths)
    outputs = _paths(workspace, output_paths)
    code = _paths(workspace, code_paths)
    procedural = _paths(workspace, procedural_inputs)
    from datasets.scripts.common.paths import validate_guarded_outputs
    task = validate_guarded_outputs(code, outputs, procedural, allow_outputless=plan is None and not outputs)
    if task:
        common_paths = 'datasets/scripts/common/paths.py'
        if common_paths not in code:
            code.append(common_paths)
            code.sort()
    inputs = sorted(set(inputs) | set(procedural))
    inplace = _paths(workspace, inplace_paths)
    adopted = _paths(workspace, adopt_outputs)
    dependencies = _paths(workspace, dependency_paths)
    samples = _paths(workspace, sample_output_paths)
    auxiliary = _paths(workspace, auxiliary_output_paths)
    if plan is not None and (not outputs or inplace or adopted):
        raise ValueError('planned production requires outputs and does not support inplace or adopted outputs')
    if replace_verifier is not None and (not replace_outputs or not callable(replace_verifier)):
        raise ValueError('replace_verifier requires an explicit replacement')
    if replace_outputs and plan is None and replace_verifier is None:
        raise ValueError('output replacement requires a production plan')
    if replace_verifier is not None and (not outputs or inplace or adopted):
        raise ValueError('verified replacement requires separate declared outputs')
    if plan is not None:
        if not samples or any(not any(_within(path, out) for out in outputs) for path in samples + auxiliary):
            raise ValueError('sample and auxiliary paths must be inside declared outputs; sample paths are required')
        if any(_overlap(sample, aux) for sample in samples for aux in auxiliary):
            raise ValueError('sample and auxiliary output paths must not overlap')
    elif samples or auxiliary:
        raise ValueError('sample and auxiliary output paths require a production plan')
    if not inputs or not code:
        raise ValueError('inputs and code_paths are required')
    if not set(adopted).issubset(outputs):
        raise ValueError('adopt_outputs must be explicitly listed output paths')
    for dep in dependencies:
        pieces = Path(dep).parts
        if (len(pieces) != 6 or tuple(pieces[:4]) != ('lab', 'training_common', 'portable_env', 'bundles') or
                pieces[5] != 'site-packages' or not (workspace / dep).is_dir() or
                not any(_within(dep, path) for path in inputs)):
            raise ValueError('dependency_paths must name a declared portable_env bundle site-packages directory')
    if any(not any(_overlap(p, x) for x in inputs) or not any(_overlap(p, x) for x in outputs)
           for p in inplace):
        raise ValueError('inplace_paths must lie in both input and output scopes')
    if any(_overlap(a, b) for a in code for b in outputs):
        raise ValueError('code and output paths must not overlap')
    for a in inputs:
        for b in outputs:
            if _overlap(a, b):
                intersection = a if _within(a, b) else b
                if not any(_within(intersection, p) for p in inplace):
                    raise ValueError('input/output overlap requires an explicit inplace_paths entry')
    params = json.loads(json.dumps(parameters, ensure_ascii=False, sort_keys=True, allow_nan=False))
    child_bindings = params.get('child_production_plans') if isinstance(params, dict) else None
    if child_bindings is not None and (not isinstance(child_bindings, dict) or
            any(not isinstance(key, str) or not key.strip() or not isinstance(value, dict)
                for key, value in child_bindings.items())):
        raise ValueError('child_production_plans must map operation names to declarations')
    inputs_state = _inventory(workspace, inputs, True)
    code_state = _inventory(workspace, code, True)
    guard_stat = Path(__file__).stat()
    guard_code = {'path': 'lab/training_common/production_guard.py', 'bytes': guard_stat.st_size,
                  'modified_ns': guard_stat.st_mtime_ns}
    media_suffixes = cache.core.IMAGES | cache.core.VIDEOS | {'.npz'}
    undeclared = [p for p in inputs if p not in procedural]
    if any(not (workspace / p).is_file() for p in undeclared):
        raise ValueError('media inputs must be explicit files; declare configuration inputs separately')
    states = {item['path']: item for item in inputs_state if item['type'] == 'file'}
    media = sorted(set(undeclared) | {path for path in states if Path(path).suffix.lower() in media_suffixes and
                                    not any(_within(path, dep) for dep in dependencies)})
    token = uuid.uuid4().hex
    with _transaction(cache) as db:
        rows = list(db.execute('SELECT operation,running_json FROM production_operations '
                               'WHERE running_json IS NOT NULL'))
        parent_name = os.environ.get('AEC_PRODUCTION_PARENT_OPERATION')
        parent_token = os.environ.get('AEC_PRODUCTION_PARENT_TOKEN')
        if bool(parent_name) != bool(parent_token):
            raise ValueError('incomplete parent production token')
        parent = None
        if parent_name:
            row = next((row for row in rows if row[0] == parent_name), None)
            parent = json.loads(row[1]) if row and row[1] else None
            if (not parent or parent.get('token') != parent_token or not _alive(cache, parent) or
                    parent.get('parent_token') or parent_name == operation or 'input_state' not in parent):
                raise ValueError('invalid or inactive parent production operation')
            if not set(code).issubset(parent['code_paths']):
                raise ValueError('child code is not declared by parent')
            parent_code = {item['path']: item for item in parent['code_state']}
            if any(parent_code.get(item['path']) != item for item in code_state):
                raise ValueError('child code changed since the parent operation started')
            if not outputs or any(not any(_within(p, parent_out) for parent_out in parent['outputs'])
                                  for p in outputs):
                raise ValueError('child outputs must be within parent outputs')
        prior = {item['path']: item for item in parent['prior_outputs']} if parent else {}
        input_inventory = {item['path']: item for item in inputs_state}
        staged = [p for p in inputs if parent and any(_within(p, out) for out in parent['outputs']) and
                  input_inventory.get(p) != prior.get(p)]
        if parent:
            if replace_outputs:
                raise ValueError('output replacement is only supported by a top-level production run')
            if plan is not None:
                planner = 'lab/training_common/bayesian_allocation.py'
                if planner not in parent['code_paths']:
                    raise ValueError('parent must declare planner code before starting a planned child')
                parent_params = parent.get('parameters')
                bindings = parent_params.get('child_production_plans') if isinstance(parent_params, dict) else None
                if not isinstance(bindings, dict) or operation not in bindings:
                    raise ValueError('parent must bind the child production declaration in parameters')
                bound = json.dumps(bindings[operation], ensure_ascii=False, sort_keys=True, allow_nan=False)
                actual = json.dumps(declaration, ensure_ascii=False, sort_keys=True, allow_nan=False)
                if bound != actual:
                    raise ValueError('child production declaration differs from the parent binding')
            parent_inputs = {item['path']: item for item in parent['input_state']}
            for path, state in input_inventory.items():
                is_staged = any(_within(path, root) for root in staged)
                if is_staged:
                    if state['type'] == 'file' and state == prior.get(path):
                        raise ValueError('child staged input was not created or changed by the parent: ' + path)
                elif parent_inputs.get(path) != state:
                    raise ValueError('child input is not in the parent input closure or changed: ' + path)
        staged_media = [p for p in media if any(_within(p, root) for root in staged)]
        staged_info = {p: {'parent_operation': parent_name, 'parent_source_refs': parent['source_refs']}
                       for p in staged}
        previous_row = db.execute('SELECT receipt_json FROM production_operations '
                                  'WHERE operation=?', (operation,)).fetchone()
        previous = json.loads(previous_row[0]) if previous_row and previous_row[0] else None
        for name, running_json in rows:
            running = json.loads(running_json) if running_json else None
            if running and _alive(cache, running) and name != parent_name:
                if name == operation or any(_overlap(a, b) for a in outputs for b in running['outputs']):
                    raise RuntimeError('production operation or output is already in use: ' + name)
        output_state = _inventory(workspace, outputs, False)
        preserved = {}
        if replace_verifier is not None:
            # Verifier is trusted adapter code: it checks ownership, collisions
            # and recovery boundaries; this guard does not remove or move files.
            # Copies prevent it from changing admission data by mutating inputs.
            snapshot = json.loads(json.dumps(output_state, ensure_ascii=False))
            receipt_copy = json.loads(json.dumps(previous, ensure_ascii=False)) if previous is not None else None
            verification = replace_verifier(workspace, snapshot, receipt_copy)
            if plan is None:
                if verification is not True:
                    raise ValueError('replacement verifier did not approve output ownership')
            else:
                if (not isinstance(verification, dict) or set(verification) != {'preserved_paths'} or
                        not isinstance(verification['preserved_paths'], list) or
                        any(not isinstance(p, str) for p in verification['preserved_paths'])):
                    raise ValueError('planned replacement verifier must return preserved_paths list')
                paths = _paths(workspace, verification['preserved_paths'])
                if len({p.casefold() for p in paths}) != len(paths):
                    raise ValueError('preserved output paths differ only by case')
                observed = {item['path']: item for item in output_state if item.get('type') == 'file'}
                for path in paths:
                    if path not in observed or not any(_within(path, out) for out in outputs):
                        raise ValueError('preserved output must be a file in declared outputs: ' + path)
                    preserved[path] = {'path': path, 'sha256': _file_digest(workspace / path)}
            if _inventory(workspace, outputs, False) != output_state:
                raise ValueError('replacement verifier changed output state')
        any_output = any(not x.get('missing') for x in output_state)
        if all(
                item.get('missing') or (item.get('type') == 'directory' and item['path'] in outputs)
                or (plan is not None and item.get('type') == 'file' and
                    any(item['path'] == out + '/.gitkeep' for out in outputs))
                for item in output_state):
            any_output = False
        if any_output and not parent and not replace_outputs:
            expected = previous['outputs'] if previous else []
            allowed = inplace + (adopted if not previous else [])
            if _outside_inplace(output_state, allowed) != _outside_inplace(expected, allowed):
                raise ValueError('existing outputs have no matching complete receipt; explicit adoption required')
        refs, observations = _sources(cache, [p for p in media if p not in staged_media], states)
        refs.update({p: [] for p in staged_media})
        request = {'inputs': inputs_state, 'code': code_state, 'parameters': params,
                   'guard_code': guard_code, 'procedural_inputs': procedural,
                   'dependency_paths': dependencies, 'output_paths': outputs,
                   'inplace_paths': inplace, 'adopt_outputs': adopted}
        if plan is not None:
            planner_path = Path(__file__).with_name('bayesian_allocation.py')
            planner_stat = planner_path.stat()
            request['production_plan'] = plan
            request['planner_code'] = {'bytes': planner_stat.st_size, 'modified_ns': planner_stat.st_mtime_ns}
            request['sample_output_paths'] = samples
            request['auxiliary_output_paths'] = auxiliary
        comparable = dict(previous['request'], inputs=previous.get('post_inputs', previous['request']['inputs'])) if previous else None
        if not replace_outputs and not parent and previous and comparable == request and previous['outputs'] == output_state:
            run = ProductionRun(True, refs, observations, previous.get('result'))
        else:
            reservation = {'token': token, 'pid': os.getpid(), 'host': socket.gethostname(), 'outputs': outputs,
                           'code_paths': code, 'code_state': code_state, 'input_state': inputs_state, 'source_refs': refs,
                           'prior_outputs': output_state, 'parent_token': parent_token, 'stages': [],
                           'parameters': params}
            db.execute('INSERT INTO production_operations(operation,running_json) VALUES(?,?) '
                       'ON CONFLICT(operation) DO UPDATE SET running_json=excluded.running_json',
                       (operation, json.dumps(reservation, ensure_ascii=False)))
            run = ProductionRun(False, refs, observations, staged_inputs=staged_info,
                                _operation=operation, _token=token)
        run._plan = plan
        run._workspace = workspace
        run._outputs = outputs
        run._sample_outputs = samples
        run._auxiliary_outputs = auxiliary
        run._preserved = preserved
        if plan is not None:
            run._states = {row['id']: row for row in plan['strata']}
            run._state_counts = {state_id: 0 for state_id in run._states}
    if run.reused:
        try:
            yield run
        finally:
            run._invalidate()
        return
    run._active = True
    try:
        yield run
        if not outputs and not run._result_set:
            raise ValueError('an outputless operation must call run.set_result(value)')
        post_inputs = _inventory(workspace, inputs, True)
        if (_outside_inplace(post_inputs, inplace) != _outside_inplace(inputs_state, inplace) or
                _inventory(workspace, code, True) != code_state):
            raise ValueError('inputs or code changed during production; completion not recorded')
        new_guard_stat = Path(__file__).stat()
        if (new_guard_stat.st_size, new_guard_stat.st_mtime_ns) != (guard_stat.st_size, guard_stat.st_mtime_ns):
            raise ValueError('production guard code changed; completion not recorded')
        finished = _inventory(workspace, outputs, True, nonempty=True)
        run._check_samples(finished)
        if plan is not None:
            planner_stat = Path(__file__).with_name('bayesian_allocation.py').stat()
            if request['planner_code'] != {'bytes': planner_stat.st_size, 'modified_ns': planner_stat.st_mtime_ns}:
                raise ValueError('production planner code changed; completion not recorded')
        with _transaction(cache) as db:
            current = db.execute('SELECT running_json FROM production_operations WHERE operation=?', (operation,)).fetchone()
            if not current or not current[0] or json.loads(current[0])['token'] != token:
                raise RuntimeError('production reservation changed; completion not recorded')
            current_run = json.loads(current[0])
            if current_run.get('stage_failed'):
                raise RuntimeError('a child production stage failed; completion not recorded')
            if child_bindings is not None:
                planned_children = [stage['operation'] for stage in current_run['stages']
                                    if 'production_plan' in stage['request']]
                if sorted(planned_children) != sorted(child_bindings):
                    raise ValueError('completed planned child operations differ from parent bindings')
            receipt = {'request': request, 'post_inputs': post_inputs, 'outputs': finished,
                       'source_refs': refs, 'media_observations': observations,
                       'result': run.value if run._result_set else None, 'stages': current_run['stages'],
                       'adopted_output_state': current_run['prior_outputs'] if adopted else []}
            if plan is not None:
                receipt['planned_samples'] = run._samples
                if run._preserved:
                    receipt['preserved_outputs'] = list(run._preserved.values())
            if parent:
                parent_row = db.execute('SELECT running_json FROM production_operations WHERE operation=?', (parent_name,)).fetchone()
                parent_running = json.loads(parent_row[0]) if parent_row and parent_row[0] else None
                if not parent_running or parent_running['token'] != parent_token or not _alive(cache, parent_running):
                    raise RuntimeError('parent production ended; child completion not recorded')
                stage = {'operation': operation, 'staged_inputs': staged_info,
                         'source_refs': refs, 'request': request, 'outputs': finished}
                if plan is not None:
                    stage['planned_samples'] = run._samples
                parent_running['stages'].append(stage)
                db.execute('UPDATE production_operations SET running_json=? WHERE operation=?',
                           (json.dumps(parent_running, ensure_ascii=False), parent_name))
                db.execute('UPDATE production_operations SET running_json=NULL WHERE operation=?', (operation,))
            else:
                encoded_receipt = json.dumps(receipt, ensure_ascii=False)
                db.execute('INSERT INTO production_receipt_history '
                           '(operation,run_token,origin,receipt_json) VALUES(?,?,?,?)',
                           (operation, token, 'completed_run', encoded_receipt))
                db.execute('UPDATE production_operations SET receipt_json=?,running_json=NULL WHERE operation=?',
                           (encoded_receipt, operation))
    except BaseException:
        # Failure is never a completion; preserve any prior receipt for audit.
        try:
            with _transaction(cache) as db:
                current = db.execute('SELECT running_json FROM production_operations WHERE operation=?', (operation,)).fetchone()
                if current and current[0] and json.loads(current[0])['token'] == token:
                    db.execute('UPDATE production_operations SET running_json=NULL WHERE operation=?', (operation,))
                if parent:
                    row = db.execute('SELECT running_json FROM production_operations WHERE operation=?', (parent_name,)).fetchone()
                    if row and row[0]:
                        parent_running = json.loads(row[0])
                        if parent_running['token'] == parent_token:
                            parent_running['stage_failed'] = True
                            db.execute('UPDATE production_operations SET running_json=? WHERE operation=?',
                                       (json.dumps(parent_running, ensure_ascii=False), parent_name))
        except Exception:
            pass
        raise
    finally:
        run._invalidate()


def _mark_parent_failed(cache, name, token):
    """Record child admission failures after the admission transaction rolls back."""
    with _transaction(cache) as db:
        row = db.execute('SELECT running_json FROM production_operations WHERE operation=?',
                         (name,)).fetchone()
        parent = json.loads(row[0]) if row and row[0] else None
        if parent and parent.get('token') == token and _alive(cache, parent):
            parent['stage_failed'] = True
            db.execute('UPDATE production_operations SET running_json=? WHERE operation=?',
                       (json.dumps(parent, ensure_ascii=False), name))


@contextmanager
def generation_guard(operation, input_paths, output_paths, parameters, *, code_paths=(),
                     procedural_inputs=(), inplace_paths=(), adopt_outputs=(), dependency_paths=(), root=None,
                     production_plan=None, sample_output_paths=(), auxiliary_output_paths=(),
                     replace_outputs=False, replace_verifier=None):
    """Guard an operation; failures under a valid parent also fail that parent.

    Before entering a parent guard, adapters must read and validate each child
    declaration and include it in parameters['child_production_plans'], keyed
    by the child's operation. They must declare the plan file as an input and
    the allocator in code_paths. Parent reuse happens before a child starts;
    a child cannot detect a declaration omitted by a reused parent.

    For an unplanned top-level replacement, a trusted replace_verifier must
    return True. For a planned replacement it must return a dictionary with
    only ``preserved_paths``: a list of existing output file paths to retain.
    The verifier receives (workspace, output_state, previous_receipt) without
    modifying outputs. The guard checks retained file hashes at completion,
    but cannot prevent adapter writes: the caller must preflight name clashes
    and use staging/recovery. No verifier preserves the planned default checks.
    """
    try:
        plan = declaration = None
        if production_plan is not None:
            declaration = json.loads(json.dumps(production_plan, ensure_ascii=False, allow_nan=False))
            plan = plan_production(declaration, root)
        with _generation_guard(operation, input_paths, output_paths, parameters, code_paths=code_paths,
                               procedural_inputs=procedural_inputs, inplace_paths=inplace_paths,
                               adopt_outputs=adopt_outputs, dependency_paths=dependency_paths, root=root,
                               plan=plan, declaration=declaration, sample_output_paths=sample_output_paths,
                               auxiliary_output_paths=auxiliary_output_paths, replace_outputs=replace_outputs,
                               replace_verifier=replace_verifier) as run:
            yield run
    except BaseException:
        name = os.environ.get('AEC_PRODUCTION_PARENT_OPERATION')
        token = os.environ.get('AEC_PRODUCTION_PARENT_TOKEN')
        if name and token:
            try:
                _mark_parent_failed(_index(root), name, token)
            except Exception as failure:
                raise RuntimeError('could not record child failure in parent production state') from failure
        raise
