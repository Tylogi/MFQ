"""验证真实服务入口的 SPA 回退、静态资源和接口错误隔离。"""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

from mfq.runtime.minicpmo45_realtime import build_app
from mfq.server.api import create_app
from mfq.server.api.static import SPAStaticFiles

INDEX = '<!doctype html><title>MFQ Studio</title><div id="root"></div>'


@pytest.fixture
def web_root(tmp_path: Path) -> Path:
  """创建最小构建产物，包含入口、真实资源和独立静态页面。"""
  root = tmp_path / 'web'
  root.mkdir()
  (root / 'index.html').write_text(INDEX, encoding='utf-8')
  (root / 'assets').mkdir()
  (root / 'assets' / 'app.js').write_text('console.log("loaded");', encoding='utf-8')
  (root / 'guide').mkdir()
  (root / 'guide' / 'index.html').write_text('static guide', encoding='utf-8')
  (tmp_path / 'secret.txt').write_text('private content', encoding='utf-8')
  return root


@pytest.fixture(params=['server', 'realtime'])
def client(request: pytest.FixtureRequest, web_root: Path, monkeypatch):
  """对主服务与实时音频服务运行相同的页面和资源契约测试。"""
  if request.param == 'server':
    app = create_app(web_root=web_root)
  else:
    # 代理连接真实 API 应用，避免依赖外部推理服务或网络。
    transport = httpx.ASGITransport(app=create_app())
    monkeypatch.setattr(httpx, 'AsyncClient', partial(httpx.AsyncClient, transport=transport))
    gateway = SimpleNamespace(api_key='', backend_url='http://127.0.0.1:1')
    app = build_app(gateway, web_root=web_root)
  with TestClient(app, headers={'Accept': 'text/html'}) as connection:
    yield connection


@pytest.mark.parametrize('path', [
  '/', '/chat', '/models', '/settings', '/settings/profile',
  '/future-page', '/chat/?session=123', '/unknown/nested/page',
])
def test_navigation_and_refresh_return_entry(client: TestClient, path: str) -> None:
  """直接访问、重复刷新和未来页面均返回同一入口且不重定向。"""
  for _ in range(2):
    response = client.get(path, follow_redirects=False)
    assert response.status_code == 200
    assert response.text == INDEX
    assert response.headers['content-type'].startswith('text/html')
    assert 'location' not in response.headers
  head = client.head(path, follow_redirects=False)
  assert head.status_code == 200
  assert head.content == b''
  assert int(head.headers['content-length']) == len(INDEX.encode())


@pytest.mark.parametrize('path', [
  '/assets/missing.js', '/assets/missing', '/static/missing',
  '/missing.css', '/missing.png', '/missing.js/', '/favicon.ico',
  '/openapi-missing.json', '/health/missing', '/docs/missing',
  '/realtime/missing', '/%2e%2e/secret.txt',
])
def test_missing_resources_stay_404(client: TestClient, path: str) -> None:
  """缺失资源、保留前缀和目录穿越请求不会被伪装成成功页面。"""
  response = client.get(path)
  assert response.status_code == 404
  assert response.text != INDEX
  assert 'private content' not in response.text


def test_real_files_and_health_keep_original_responses(client: TestClient) -> None:
  """真实静态文件及健康检查优先于 SPA 回退。"""
  assert client.get('/assets/app.js').text == 'console.log("loaded");'
  assert client.get('/guide/').text == 'static guide'
  response = client.get('/health')
  assert response.status_code == 200
  assert response.headers['content-type'].startswith('application/json')


@pytest.mark.parametrize('headers', [
  {'Accept': 'application/json'}, {'Accept': '*/*'},
  {'Accept': 'text/html', 'Sec-Fetch-Dest': 'script'},
])
def test_non_navigation_does_not_fallback(client: TestClient, headers: dict) -> None:
  """数据请求和脚本加载不应收到应用入口。"""
  assert client.get('/chat', headers=headers).status_code == 404


def test_post_does_not_fallback(client: TestClient) -> None:
  """提交请求不能被页面回退误报为成功。"""
  assert client.post('/chat').status_code == 405


def test_api_and_proxy_errors_stay_json(client: TestClient) -> None:
  """主服务及实时代理都保留接口 404 和业务错误。"""
  for path, status in [('/api/v1/missing', 404), ('/v1/missing', 404), ('/api/v1/models', 501)]:
    response = client.get(path)
    assert response.status_code == status
    assert response.headers['content-type'].startswith('application/json')


def test_existing_404_html_does_not_disable_fallback(client: TestClient, web_root: Path) -> None:
  """构建目录存在 404.html 时，页面回退仍然有效。"""
  (web_root / '404.html').write_text('static not found', encoding='utf-8')
  assert client.get('/chat').text == INDEX
  response = client.get('/missing.js')
  assert response.status_code == 404
  assert response.text == 'static not found'


@pytest.mark.parametrize('api_key', ['', 'test-secret'])
def test_api_errors_and_auth_are_not_masked(web_root: Path, api_key: str) -> None:
  """接口不存在、鉴权失败及业务错误保留原有状态和 JSON。"""
  with TestClient(create_app(web_root=web_root, api_key=api_key)) as connection:
    for path in ['/api/v1/missing', '/v1/missing']:
      response = connection.get(path, headers={'Accept': 'text/html'})
      assert response.status_code == (401 if api_key else 404)
      assert response.headers['content-type'].startswith('application/json')
    headers = {'Accept': 'text/html', 'Authorization': f'Bearer {api_key}'}
    assert connection.get('/api/v1/models', headers=headers).status_code == 501
    assert connection.get('/chat', headers={'Accept': 'text/html'}).text == INDEX


@pytest.mark.parametrize('status', [403, 500])
def test_other_static_errors_are_not_masked(web_root: Path, monkeypatch, status: int) -> None:
  """回退只处理 404，不吞掉权限错误和内部错误。"""
  async def fail(self, path, scope):
    """模拟静态服务发生非 404 错误。"""
    raise HTTPException(status_code=status)

  monkeypatch.setattr(StaticFiles, 'get_response', fail)
  app = FastAPI()
  app.mount('/', SPAStaticFiles(directory=web_root, html=True))
  with TestClient(app) as connection:
    assert connection.get('/chat', headers={'Accept': 'text/html'}).status_code == status
