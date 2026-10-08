from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


class CalibrationChatRenderer:
    def __init__(self, tokenizer, *, enabled=True, template_kwargs=None):
        self.tokenizer = tokenizer
        self.enabled = enabled
        self.kwargs = dict(template_kwargs or {})
        self.counts = dict(wrapped=0, preformatted=0, plain=0, conversations=0)
        template = getattr(tokenizer, 'chat_template', None)
        self.has_template = bool(template)
        self.prefixes = ()
        self.affixes = None
        if enabled and self.has_template:
            marker = 'MFQ_CALIBRATION_CONTENT_7b38e420'
            rendered = self._apply([{'role': 'user', 'content': marker}], tokenize=False)
            if not isinstance(rendered, str) or rendered.count(marker) != 1:
                raise ValueError('model chat template must preserve calibration content')
            self.affixes = tuple(rendered.split(marker))
            openings = [self.affixes[0]]
            try:
                system = self._apply([{'role': 'system', 'content': marker}, {'role': 'user', 'content': 'probe'}], tokenize=False)
                if isinstance(system, str) and system.count(marker) == 1:
                    openings.append(system.split(marker)[0])
            except Exception:
                pass
            excluded = {getattr(tokenizer, name, None) for name in ('bos_token', 'eos_token', 'pad_token', 'unk_token')}
            special = [str(value) for value in getattr(tokenizer, 'all_special_tokens', ()) if str(value) not in excluded]
            prefixes = set()
            bos = str(getattr(tokenizer, 'bos_token', '') or '')
            for opening in openings:
                prefix = opening.removeprefix(bos).lstrip() if bos else opening.lstrip()
                if prefix:
                    prefixes.add(prefix)
                    first = next((value for value in sorted(special, key=len, reverse=True) if prefix.startswith(value)), None)
                    if first:
                        prefixes.add(first)
            self.prefixes = tuple(prefixes)

    def _apply(self, messages, *, tokenize):
        kwargs = dict(self.kwargs)
        generation = kwargs.pop('add_generation_prompt', messages[-1]['role'] == 'user')
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=generation, **kwargs)

    def is_preformatted(self, text):
        value = text.lstrip('\ufeff \t\r\n')
        bos = str(getattr(self.tokenizer, 'bos_token', '') or '')
        if bos:
            value = value.removeprefix(bos).lstrip()
        return any(value.startswith(prefix) for prefix in self.prefixes)

    def _finish(self, value):
        if isinstance(value, Mapping):
            value = value['input_ids']
        ids = [int(token) for token in value]
        eos = getattr(self.tokenizer, 'eos_token_id', None)
        if eos is not None and (not ids or ids[-1] != int(eos)):
            ids.append(int(eos))
        return ids

    def encode(self, record):
        if isinstance(record, list):
            self.counts['conversations'] += 1
            if self.enabled and self.has_template:
                return self._finish(self._apply(record, tokenize=True))
            record = '\n'.join(message['content'] for message in record)
        if not isinstance(record, str):
            raise TypeError('calibration record must be text or a conversation')
        if self.enabled and self.has_template:
            if self.is_preformatted(record):
                self.counts['preformatted'] += 1
            else:
                self.counts['wrapped'] += 1
                return self._finish(self._apply([{'role': 'user', 'content': record}], tokenize=True))
        else:
            self.counts['plain'] += 1
        return self._finish(self.tokenizer.encode(record, add_special_tokens=False))

    def wrap_ids(self, tokens):
        ids = [int(token) for token in tokens]
        opening = self.tokenizer.decode(ids[:256], skip_special_tokens=False)
        if self.is_preformatted(opening):
            self.counts['preformatted'] += 1
            return ids
        eos = getattr(self.tokenizer, 'eos_token_id', None)
        if eos is not None and ids and ids[-1] == int(eos):
            ids = ids[:-1]
        prefix, suffix = self.affixes
        self.counts['wrapped'] += 1
        return self._finish([
            *self.tokenizer.encode(prefix, add_special_tokens=False),
            *ids,
            *self.tokenizer.encode(suffix, add_special_tokens=False),
        ])

    def summary(self):
        return {'enabled': self.enabled, 'model_has_template': self.has_template, **self.counts}


def _record(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise ValueError('calibration JSON records must contain text or messages')
    messages = value.get('messages', value.get('conversations'))
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise ValueError('calibration messages must be a nonempty array')
        normalized = []
        aliases = {'human': 'user', 'gpt': 'assistant'}
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError('calibration messages must be objects')
            role = message.get('role', message.get('from'))
            role = aliases.get(role, role)
            content = message.get('content', message.get('value'))
            if role not in {'system', 'user', 'assistant', 'tool'} or not isinstance(content, str):
                raise ValueError('calibration messages require a valid role and text content')
            normalized.append({**message, 'role': role, 'content': content})
        return normalized
    for name in ('text', 'content'):
        if isinstance(value.get(name), str):
            return value[name]
    raise ValueError('calibration JSON records must contain text or messages')


def _file_records(path):
    with path.open(encoding='utf-8-sig') as stream:
        if path.suffix.lower() == '.jsonl':
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield _record(json.loads(line))
                    except (ValueError, TypeError) as error:
                        raise ValueError(f'invalid calibration record at line {number}: {error}') from error
        elif path.suffix.lower() == '.json':
            data = json.load(stream)
            if isinstance(data, Mapping):
                data = data.get('data', data.get('records', [data]))
            if not isinstance(data, list):
                raise ValueError('calibration JSON must be a record or an array of records')
            for value in data:
                yield _record(value)
        elif path.suffix.lower() == '.txt':
            paragraph = []
            for line in stream:
                if line.strip():
                    paragraph.append(line)
                elif paragraph:
                    yield ''.join(paragraph).strip()
                    paragraph = []
            if paragraph:
                yield ''.join(paragraph).strip()
        else:
            raise ValueError('calibration input must be .txt, .json, .jsonl or a prepared MFQ corpus directory')


@contextmanager
def prepare_calibration_corpus(path, model, *, apply_chat_template=True, window_length=16384, train_tokens=1572864, tokenizer=None):
    from mfq.calibration.dataset import (
        _FORMAT,
        _file_identity,
        _pack_stream,
        _tokenizer_metadata,
        _write_corpus_artifact,
        load_corpus,
    )

    if window_length < 2 or train_tokens < 2:
        raise ValueError('calibration window length and train token limit must be at least two')
    path = Path(path).resolve()
    original = load_corpus(path) if path.is_dir() else None
    try:
        if original is not None:
            mode = original.manifest.get('tokenizer', {}).get('render_mode')
            rendering = original.manifest.get('chat_rendering', {})
            if not apply_chat_template or mode in {'chat', 'trace-chat'} or (mode == 'auto' and rendering.get('model_has_template')):
                print(json.dumps({'event': 'calibration_corpus_prepared', 'chat_rendering': {
                    'enabled': apply_chat_template, 'status': 'already_prepared' if apply_chat_template else 'disabled',
                }}), flush=True)
                yield original
                return
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
        renderer = CalibrationChatRenderer(tokenizer, enabled=apply_chat_template)
        if original is not None and not renderer.has_template:
            print(json.dumps({'event': 'calibration_corpus_prepared', 'chat_rendering': renderer.summary()}), flush=True)
            yield original
            return
        chunks, splits, domains = [], [], []
        if original is not None:
            if original.loss_mask is not None:
                raise ValueError('cannot automatically wrap an assistant-masked plain corpus; use chat-formatted records or disable automatic chat templates')
            for index in range(original.chunks):
                parts = _pack_stream(renderer.wrap_ids(original.chunk_tokens(index)), window_length)
                chunks.extend(parts)
                splits.extend([int(original.split_ids[index])] * len(parts))
                domains.extend([int(original.domain_ids[index])] * len(parts))
            manifest = dict(original.manifest)
            manifest['sources'] = {'prepared_corpus': _file_identity(path / 'manifest.json')}
        else:
            used = 0
            for record in _file_records(path):
                if isinstance(record, str) and not record.strip():
                    continue
                ids = renderer.encode(record)
                remaining = train_tokens - used
                if remaining < 2:
                    break
                parts = _pack_stream(ids[:remaining], window_length)
                chunks.extend(parts)
                used += sum(part.size for part in parts)
                if used >= train_tokens:
                    break
            splits, domains = [0] * len(chunks), [0] * len(chunks)
            manifest = {'format': _FORMAT, 'domains': ['local'], 'sources': {'file': _file_identity(path)}}
        manifest.update(sequence_length=window_length, tokenizer=_tokenizer_metadata(tokenizer, 'auto' if apply_chat_template else 'plain'),
            chat_rendering=renderer.summary(), chat_template_kwargs=renderer.kwargs,
            actual_tokens={name: sum(chunk.size for chunk, split in zip(chunks, splits, strict=True) if split == index)
                for index, name in enumerate(('train', 'validation'))})
        manifest['target_tokens'] = dict(manifest['actual_tokens'])
        print(json.dumps({'event': 'calibration_corpus_prepared', 'chat_rendering': renderer.summary(), 'tokens': manifest['actual_tokens']}), flush=True)
        with (
            TemporaryDirectory(prefix='mfq-calibration-') as temporary,
            _write_corpus_artifact(Path(temporary) / path.stem, chunks, splits, domains, manifest) as corpus,
        ):
            yield corpus
    finally:
        if original is not None:
            original.close()
