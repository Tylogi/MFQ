from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
from contextlib import suppress
from pathlib import Path

import psutil

from mfq.quantize.workbench_candidates import WORKBENCH_CANDIDATES, CANDIDATE_GROUPS
from mfq.server.protocol.models import ErrorDetail
from mfq.server.protocol.quantization import (
    FitPayload, QuantizationDirectories, QuantizationFile, QuantizationFiles,
    QuantizationSource, QuantizationWorkspace, RecipePayload, WorkbenchImatrixPayload, WorkbenchImportPayload,
)
from mfq.server.services.jobs import JobExecutionError, TypedJobHandler


def failure(code, message, details=None):
    return JobExecutionError(ErrorDetail(code=code, message=message, details=details or {}))


class QuantizationWorkbench:
    def __init__(self, tools):
        self.tools = tools
        self._lock = asyncio.Lock()
        self._quiet_since = None
        self.settings_path = tools.root / 'quantization' / 'directories.json'

    def handlers(self):
        return {
            'quantization.recipe': TypedJobHandler(self.generate, RecipePayload),
            'quantization.fit': TypedJobHandler(self.fit, FitPayload),
            'quantization.import': TypedJobHandler(self.import_artifact, WorkbenchImportPayload),
            'quantization.imatrix': TypedJobHandler(self.imatrix, WorkbenchImatrixPayload),
        }

    def workspace(self):
        defaults = {'import_directory': str(self.tools.model_root), 'export_directory': str(self.tools.root / 'quantization' / 'outputs')}
        if self.settings_path.is_file():
            saved = QuantizationDirectories.model_validate_json(self.settings_path.read_text())
            defaults.update(saved.model_dump())
        return QuantizationWorkspace(**defaults, candidates=list(WORKBENCH_CANDIDATES),
            candidate_groups={group: list(names) for group, names in CANDIDATE_GROUPS})

    def configure(self, request):
        values = {name: str(self.resolve(value)) for name, value in request.model_dump().items()}
        for value in values.values():
            path = Path(value)
            if path.exists() and not path.is_dir():
                raise failure('not_a_directory', f'not a directory: {path}')
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(values, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, self.settings_path)
        return self.workspace()

    def resolve(self, value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else self.tools.root / path).resolve()

    def files(self, value=None):
        path = self.resolve(value or self.workspace().import_directory)
        if not path.is_dir():
            raise failure('directory_not_found', f'directory not found: {path}')
        items = []
        for item in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
            if item.name.startswith('.'):
                continue
            try:
                directory = item.is_dir()
                items.append(QuantizationFile(name=item.name, path=str(item), directory=directory,
                    byte_size=None if directory else item.stat().st_size))
            except OSError:
                continue
            if len(items) >= 4096:
                break
        return QuantizationFiles(path=str(path), parent=str(path.parent) if path.parent != path else None, data=items)

    def source(self, value):
        from mfq.quantize.workbench import source_summary
        return QuantizationSource(**source_summary(self.resolve(value)))

    def loading_plan(self, request):
        from mfq.calibration.loading import loading_plan
        return loading_plan(self.resolve(request.path), **request.model_dump(exclude={'path'}))

    def validate_recipe(self, payload):
        from mfq.quantize.workbench import RecipeValidationError, recipe_arguments, validate_recipe
        try:
            recipe_arguments(payload.get('target_bpw'), payload.get('candidates', []))
            request = RecipePayload.model_validate(payload)
            return validate_recipe(self.resolve(request.input), request.target_bpw, request.candidates)
        except RecipeValidationError as error:
            raise failure(error.code, str(error), error.details) from error

    def _output(self, value, suffix):
        path = self.resolve(value)
        if path.suffix.lower() != suffix:
            raise failure('invalid_output_extension', f'output must end in {suffix}')
        if path.exists():
            raise failure('output_exists', f'output already exists: {path}')
        return path

    async def _record(self, context, path, kind, result):
        await context.artifact(name=path.name, uri=path.as_uri(), media_type=f'application/x-mfq-{kind}',
            metadata={'kind': kind, 'path': str(path), 'total_bytes': path.stat().st_size})
        return {**result, 'output': str(path), 'artifact': path.as_uri(), 'artifact_kind': kind}

    async def import_artifact(self, context, payload):
        request = WorkbenchImportPayload.model_validate(payload)
        if request.media_id is not None:
            _, source = await asyncio.to_thread(context.store.get_media_path, request.media_id)
        else:
            source = self.resolve(request.source)
        if request.kind == 'recipe':
            from mfq.quantize.workbench import load_recipe
            item = await asyncio.to_thread(load_recipe, source)
            metadata = {'method': item.target_profile, 'tensors': len(item.selections) + len(item.expert_selections)}
        else:
            from mfq.quantize.imatrix import load_importance_matrix
            item = await asyncio.to_thread(load_importance_matrix, source)
            metadata = {'entries': len(item.entries)}
        target = self._output(request.output, '.json' if request.kind == 'recipe' else '.imatrix') if request.output else source
        if target != source:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output, source.open('rb') as input_file:
                await asyncio.to_thread(shutil.copyfileobj, input_file, output)
        return await self._record(context, target, request.kind, metadata)

    async def imatrix(self, context, payload):
        request = WorkbenchImatrixPayload.model_validate(payload)
        source = await asyncio.to_thread(self.source, request.model)
        if source.format != 'hf':
            raise failure('imatrix_hf_required', 'imatrix generation requires an original HF model directory, including supported native low-precision/QAT sources')
        if not source.imatrix_supported:
            raise failure('imatrix_architecture_unsupported', 'imatrix collection currently supports Qwen3.5 and Gemma4 architectures; other sources support data-free quantization and imported imatrix files')
        output = self._output(request.output, '.imatrix')
        corpus = self.resolve(request.corpus)
        if not corpus.is_file() and not (corpus.is_dir() and (corpus / 'manifest.json').is_file()):
            raise failure('corpus_not_found', 'calibration corpus file or prepared corpus directory was not found')
        async with self._lock:
            result = await self.tools._collect_imatrix(context, model=Path(source.path), corpus=corpus, output=output,
                backend=request.backend, device='', attention='sdpa', objective='naq', window_length=request.window_length,
                batch_size=1, train_tokens=request.train_tokens, seed=20260810, accumulation_dtype='auto',
                apply_chat_template=request.apply_chat_template, layerwise=request.layerwise)
        return await self._record(context, output, 'imatrix', result)

    async def generate(self, context, payload):
        request = RecipePayload.model_validate(payload)
        if len(set(request.candidates)) != len(request.candidates) or set(request.candidates) - set(WORKBENCH_CANDIDATES):
            raise failure('invalid_candidates', 'select distinct canonical NVQ/NINT/native-SQ candidates')
        output = self._output(request.output, '.json')
        await asyncio.to_thread(self.validate_recipe, payload)
        arguments = ['recipe', '--input', str(self.resolve(request.input)), '--output', str(output),
            '--target-bpw', str(request.target_bpw), '--candidates', ','.join(request.candidates), '--backend', request.backend]
        async with self._lock:
            result = await self._worker(context, arguments, request.service_policy)
        return await self._record(context, output, 'recipe', result)

    async def fit(self, context, payload):
        request = FitPayload.model_validate(payload)
        output = self._output(request.output, '.mfq')
        arguments = ['fit', '--input', str(self.resolve(request.input)), '--recipe', str(self.resolve(request.recipe)),
            '--output', str(output), '--backend', request.backend, '--row-chunk', str(request.row_chunk)]
        if request.imatrix:
            arguments.extend(['--imatrix', str(self.resolve(request.imatrix))])
        async with self._lock:
            result = await self._worker(context, arguments, request.service_policy)
        await self.tools.catalog.register_directory(path=output.parent)
        return await self._record(context, output, 'model', result)

    async def permit(self, context, requested, policy, progress):
        if policy == 'allow':
            return requested
        announced = 0.
        while True:
            context.raise_if_cancelled()
            pool = self.tools.runtime_manager
            state = await pool.quantization_pressure() if pool is not None else {'busy': False}
            memory = psutil.virtual_memory()
            reserve = max(1 << 30, memory.total // 32)
            busy = state['busy'] or memory.available < reserve
            now = time.monotonic()
            self._quiet_since = None if busy else self._quiet_since or now
            if self._quiet_since is not None and now - self._quiet_since >= 1.0:
                memory_rows = 128 if memory.available < reserve * 2 else 256 if memory.available < reserve * 4 else 512
                return min(requested, memory_rows)
            if now - announced >= 2:
                await context.progress(progress, message='等待服务低谷或可用资源', data={'phase': 'waiting', 'reason': 'service_busy' if state['busy'] else 'memory_pressure' if busy else 'quiet_window'})
                announced = now
            try:
                await asyncio.wait_for(context.cancel_event.wait(), timeout=.25)
            except TimeoutError:
                pass

    async def _worker(self, context, arguments, policy):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[3]), environment.get('PYTHONPATH')]))
        process = await asyncio.create_subprocess_exec(*self.tools._mfq_command('_quantization-worker', *arguments, '--interactive'),
            cwd=self.tools.root, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True, env=environment)
        async def stop():
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        context.add_cleanup(stop)
        result, progress = None, .01
        try:
            while raw := await process.stdout.readline():
                context.raise_if_cancelled()
                line = raw.decode(errors='replace').strip()
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    await context.log(line[:4096]); continue
                if not isinstance(event, dict):
                    continue
                if event.get('event') == 'workbench_batch':
                    rows = await self.permit(context, int(event['requested_rows']), policy, progress)
                    process.stdin.write((json.dumps({'rows': rows}) + '\n').encode())
                    await process.stdin.drain()
                elif event.get('event') == 'workbench_progress':
                    fraction = event['completed'] / max(1, event['total'])
                    progress = .01 + .97 * fraction if event['phase'] != 'assembly' else .99
                    await context.progress(progress, message=event.get('tensor', event['phase']), data=event)
                elif event.get('event') == 'workbench_result':
                    result = event
            if await process.wait() != 0 or result is None:
                raise failure('quantization_failed', 'quantization worker failed; see task log for the cause')
            result.pop('event', None)
            return result
        finally:
            await stop()
