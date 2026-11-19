"""Bind materialized working copies to sources a task plan already selected."""

from __future__ import annotations

import filecmp
import json
from pathlib import Path

from .config import load_task_config
from .paths import WORKSPACE


CODE = Path(__file__).resolve()


class WorkingInputs:
    """Resolve byte-identical working copies for sources qualified by the caller."""

    def __init__(self, task, *, backgrounds=(), targets=(), split=None):
        _, config = load_task_config(task)
        self._by_role = {'background': {}, 'target': {}}
        self.evidence = set()
        for role, key, requested in (('background', 'background_pool_manifest', backgrounds),
                                     ('target', 'target_working_manifest', targets)):
            aliases = self._by_role[role]
            paths = {self.relative(path) for path in requested}
            if not paths or key not in config:
                continue
            manifest = WORKSPACE / config[key]
            if not manifest.resolve().is_relative_to(WORKSPACE):
                raise ValueError(f'Working input manifest outside workspace: {config[key]}')
            if not manifest.exists():
                continue  # A legacy plan can run before the generated working view has been built.
            manifest = self.workspace_file(config[key])
            catalog = json.loads(manifest.read_text(encoding='utf-8-sig'))
            if catalog.get('schema_version') != 1 or not isinstance(catalog.get('records'), list):
                raise ValueError(f'Unsupported working input manifest: {config[key]}')
            records = catalog['records']
            for row in records:
                source = row.get('source_image')
                if source not in paths:
                    continue
                alias = self.workspace_file(row['image'])
                permitted = (WORKSPACE / 'datasets/raw' /
                             ('shared/backgrounds' if key == 'background_pool_manifest' else f'{task}/targets'))
                if not alias.is_relative_to(permitted):
                    raise ValueError(f'Working copy is outside its role directory: {row["image"]}')
                if not filecmp.cmp(self.workspace_file(source), alias, shallow=False):
                    raise ValueError(f'Working copy differs from its declared input: {row["image"]}')
                if source in aliases and aliases[source] != alias:
                    raise ValueError(f'Multiple {role} working copies declared for {source}')
                aliases[source] = alias
                self.evidence.add(manifest)
        self.aliases = {**self._by_role['background'], **self._by_role['target']}

    @staticmethod
    def relative(path):
        return (WORKSPACE / path).resolve().relative_to(WORKSPACE).as_posix()

    @staticmethod
    def workspace_file(value):
        path = WORKSPACE / value
        if path.is_symlink():
            raise ValueError(f'Working input must be a regular file: {value}')
        path = path.resolve()
        if not path.is_relative_to(WORKSPACE) or not path.is_file():
            raise ValueError(f'Working input missing or outside workspace: {value}')
        return path

    @property
    def media(self):
        return {alias for aliases in self._by_role.values() for alias in aliases.values()}

    def path(self, source, *, role=None):
        key = self.relative(source)
        if role is not None:
            return self._by_role[role].get(key, WORKSPACE / key)
        if key in self._by_role['background'] and key in self._by_role['target']:
            # An unqualified lookup of a dual-role frame uses its canonical source.
            return WORKSPACE / key
        return self.aliases.get(key, WORKSPACE / key)

    def actual(self, source, *, role=None):
        return self.relative(self.path(source, role=role))
