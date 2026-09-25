/** 定义 media 领域的服务契约，仅包含类型，不依赖运行时代码。 */

export interface MediaRef {
  id: string;
  sha256: string;
  mime_type: string;
  byte_size: number;
}

export interface MediaResource {
  media: MediaRef;
  created_at: string;
}

export interface DocumentResource {
  media: MediaRef;
  name: string;
  text: string;
  page_count?: number | null;
  extractor: string;
  created_at: string;
}
