"""提供前端静态文件，并为浏览器 History 路由统一回退到应用入口。"""

from pathlib import PurePosixPath

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class SPAStaticFiles(StaticFiles):
  """在静态文件未命中时处理页面导航，保留接口及资源的错误响应。"""

  async def get_response(self, path: str, scope: Scope) -> Response:
    """优先返回实际文件，仅将符合页面导航约定的 404 回退到 index.html。"""
    headers = Headers(scope=scope)
    # StaticFiles 在 Windows 上传入反斜杠路径，先统一后再判断前缀。
    normalized = path.replace('\\', '/').strip('/')
    reserved = {'api', 'v1', 'realtime', 'health', 'docs', 'redoc', 'assets', 'static'}
    is_navigation = (
      scope['method'] in {'GET', 'HEAD'}
      and 'text/html' in headers.get('accept', '').lower()
      and headers.get('sec-fetch-dest', '') in {'', 'document', 'iframe'}
      and normalized.split('/', 1)[0] not in reserved
      and not PurePosixPath(normalized).suffix
      and '\\' not in scope['path']
      and '\x00' not in scope['path']
      and '..' not in scope['path'].split('/')
    )

    try:
      response = await super().get_response(path, scope)
    except HTTPException as error:
      if error.status_code != 404 or not is_navigation:
        raise
    else:
      # HTML 模式可能返回 404.html 响应，而不是抛出异常。
      if response.status_code != 404 or not is_navigation:
        return response

    # 仍通过 StaticFiles 读取入口，保留路径校验、缓存和 HEAD 处理。
    return await super().get_response('index.html', scope)
