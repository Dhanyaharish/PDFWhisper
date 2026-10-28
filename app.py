"""Local single-user teaching server. Run: python app.py
The left side of the page builds the index; the right side is a chat that uses it.
Images are turned into text by a local Ollama vision model, then indexed like any document.
"""
import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from rag import Index, build_prompt, generate, citation_check, ollama_post

ROOT = Path(__file__).resolve().parent
VISION_MODEL = os.environ.get('VISION_MODEL', 'qwen2.5vl:3b')  # any Ollama vision model
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.webp')
HISTORY_TURNS = 4
STATE = {'index': None}


def describe_image(raw):
    """Ask the local vision model to describe an image and transcribe any visible text."""
    payload = {
        'model': VISION_MODEL, 'stream': False, 'keep_alive': 0,
        'messages': [{
            'role': 'user',
            'content': 'Describe this image in detail. Transcribe ALL visible text exactly, '
                       'and describe any tables, charts or diagrams.',
            'images': [base64.b64encode(raw).decode()],
        }],
    }
    try:
        return ollama_post('/api/chat', payload)['message']['content']
    except Exception as exc:
        raise RuntimeError('Image reading failed. Is Ollama running and is "%s" pulled? (%s)'
                           % (VISION_MODEL, exc)) from exc


def read_uploads(files):
    docs, warnings = [], []
    for f in files:
        source = str(f['name']).replace('\\', '/').split('/')[-1][:120]
        raw = base64.b64decode(f['data'], validate=True)
        suffix = Path(source).suffix.lower()
        if suffix in ('.txt', '.md'):
            docs.append({'source': source, 'page': 1, 'text': raw.decode('utf-8-sig')})
        elif suffix == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError('PDF needs pypdf: python -m pip install pypdf') from exc
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise ValueError('Use an unencrypted PDF.')
            if len(reader.pages) > 50:
                raise ValueError('Workshop limit: 50 pages per PDF.')
            for number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ''
                if text.strip():
                    docs.append({'source': source, 'page': number, 'text': text})
                else:
                    warnings.append('%s page %s has no text; OCR is not included.' % (source, number))
        elif suffix in IMAGE_SUFFIXES:
            text = describe_image(raw)
            if text.strip():
                docs.append({'source': source, 'page': 1, 'text': text})
            else:
                warnings.append('%s: the vision model returned no text.' % source)
        else:
            raise ValueError('Supported files: TXT, MD, text-based PDF, PNG, JPG, WEBP.')
    if sum(len(d['text']) for d in docs) > 500000:
        raise ValueError('Workshop limit: 500,000 extracted characters.')
    return docs, warnings


def clean_history(raw):
    """Accept only the last few {q, a} pairs sent by the page, trimmed to a safe size."""
    turns = []
    if isinstance(raw, list):
        for turn in raw[-HISTORY_TURNS:]:
            if isinstance(turn, dict):
                q, a = str(turn.get('q', ''))[:500], str(turn.get('a', ''))[:500]
                if q and a:
                    turns.append((q, a))
    return turns


class Handler(BaseHTTPRequestHandler):
    def send(self, data, status=200, content_type='application/json'):
        raw = data if isinstance(data, bytes) else json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == '/':
            return self.send((ROOT/'index.html').read_bytes(), content_type='text/html')
        if self.path == '/api/status':
            idx = STATE['index']
            return self.send({'indexed': idx is not None, 'chunks': len(idx.chunks) if idx else 0})
        self.send({'error': 'Not found'}, 404)

    def do_POST(self):
        # Same-origin JSON requests only; server is bound to loopback.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send({'error': 'Cross-origin requests are not allowed.'}, 403)
        if self.headers.get_content_type() != 'application/json':
            return self.send({'error': 'Use application/json.'}, 415)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 8_000_000:
                raise ValueError('Request must be under 8 MB.')
            body = json.loads(self.rfile.read(length))
            start = time.perf_counter()
            if self.path == '/api/index':
                STATE['index'] = None
                if body.get('sample', False):
                    documents = [{'source': p.name, 'page': 1, 'text': p.read_text(encoding='utf-8')}
                                 for p in sorted((ROOT/'data').glob('*.txt'))]
                    warnings = []
                else:
                    documents, warnings = read_uploads(body.get('files', []))
                # Failed rebuilds invalidate old state, avoiding answers from stale documents.
                idx = Index(documents, body.get('mode', 'keyword'),
                            int(body.get('size', 100)), int(body.get('overlap', 20)))
                STATE['index'] = idx
                result = {'chunks': len(idx.chunks), 'mode': idx.mode, 'warnings': warnings,
                          'preview': [vars(c) for c in idx.chunks[:5]]}
            elif self.path == '/api/ask':
                idx = STATE['index']
                if idx is None:
                    raise ValueError('Build an index first.')
                question = str(body.get('question', '')).strip()
                if not question or len(question) > 1500:
                    raise ValueError('Question must contain 1..1500 characters.')
                mode = body.get('answer_mode', 'rag')
                if mode not in ('evidence', 'rag', 'plain'):
                    raise ValueError('Unknown answer mode.')
                history = clean_history(body.get('history'))
                # Chat memory: the previous question helps retrieval for follow-ups like "and for 2nd year?"
                search_text = (history[-1][0] + ' ' if history else '') + question
                hits = idx.search(search_text, int(body.get('k', 3)))
                chat_question = question
                if history:
                    chat_question = ('Previous conversation:\n' +
                                     '\n'.join('User: %s\nAssistant: %s' % t for t in history) +
                                     '\n\nCurrent question: ' + question)
                # Zero keyword overlap is a useful baseline rejection; semantic scores have no universal cutoff.
                no_overlap = idx.mode == 'keyword' and hits[0]['score'] <= 0
                if mode == 'evidence':
                    answer = ('No matching words. I do not know from these documents.' if no_overlap
                              else 'Retrieved passages below. This mode does not generate an answer.')
                elif no_overlap and mode == 'rag':
                    answer = "I don't know from these documents. No keyword overlap was found."
                else:
                    model = body.get('model', 'qwen2.5:1.5b')
                    if model not in ('qwen2.5:1.5b', 'qwen2.5:0.5b'):
                        raise ValueError('Choose one of the provided local generation models.')
                    answer = generate(chat_question, hits, model, use_rag=mode == 'rag')
                result = {'answer': answer, 'hits': hits, 'answer_mode': mode,
                          'prompt': chat_question if mode == 'plain' else build_prompt(chat_question, hits),
                          'citations': citation_check(answer, hits) if mode == 'rag' else None}
            else:
                return self.send({'error': 'Not found'}, 404)
            result['seconds'] = round(time.perf_counter()-start, 3)
            self.send(result)
        except (ValueError, KeyError, TypeError, UnicodeError) as exc:
            self.send({'error': str(exc)}, 400)
        except Exception as exc:
            self.send({'error': str(exc)}, 502)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print('Campus Handbook Assistant: http://127.0.0.1:%s' % args.port)
    print('Single-user classroom demo. Ctrl+C to stop. Documents stay in memory.')
    try:
        HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')