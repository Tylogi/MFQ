export type ApplicationId = 'dsh' | 'opencode' | 'claude' | 'openclaw' | 'hermes';

export const APPLICATIONS = [
  { id: 'dsh', name: 'DeepSeek Harness', url: 'https://www.deepseek.com/en/harness/', file: '~/.dsh/profiles/web/cordis.patch.yml', protocol: 'OpenAI Chat Completions' },
  { id: 'opencode', name: 'OpenCode', url: 'https://opencode.ai', file: '~/.config/opencode/opencode.json', protocol: 'OpenAI Chat Completions' },
  { id: 'claude', name: 'Claude Code', url: 'https://code.claude.com', file: '~/.claude/settings.json', protocol: 'Anthropic Messages' },
  { id: 'openclaw', name: 'OpenClaw', url: 'https://openclaw.ai', file: '~/.openclaw/openclaw.json', protocol: 'OpenAI Chat Completions' },
  { id: 'hermes', name: 'Hermes Agent', url: 'https://github.com/NousResearch/hermes-agent', file: '~/.hermes/config.yaml', protocol: 'OpenAI Chat Completions' },
] as const;

export function applicationConfiguration(id: ApplicationId, model: string, openAI: string, anthropic: string | null): string {
  const key = 'YOUR_MFQ_API_KEY';
  const json = (value: unknown) => JSON.stringify(value, null, 2);
  if (id === 'claude') {
    if (!anthropic) return '';
    return json({ env: { ANTHROPIC_BASE_URL: anthropic.replace(/\/v1\/messages$/, ''), ANTHROPIC_AUTH_TOKEN: key,
      ANTHROPIC_MODEL: model, ANTHROPIC_DEFAULT_OPUS_MODEL: model, ANTHROPIC_DEFAULT_SONNET_MODEL: model, ANTHROPIC_DEFAULT_HAIKU_MODEL: model } });
  }
  if (id === 'opencode') return json({ provider: { mfq: { npm: '@ai-sdk/openai-compatible', name: 'MFQ', options: { baseURL: openAI, apiKey: key }, models: { [model]: { name: model } } } }, model: `mfq/${model}` });
  if (id === 'openclaw') return json({ models: { mode: 'merge', providers: { mfq: { baseUrl: openAI, apiKey: key, api: 'openai-completions', models: [{ id: model, name: model }] } } }, agents: { defaults: { model: { primary: `mfq/${model}` } } } });
  if (id === 'hermes') return `providers:\n  mfq:\n    name: MFQ\n    base_url: ${JSON.stringify(openAI)}\n    api_key: ${JSON.stringify(key)}\n    api_mode: chat_completions\n    default_model: ${JSON.stringify(model)}\nmodel:\n  provider: mfq\n  default: ${JSON.stringify(model)}`;
  return json([
    { id: 'llm-pi-ai', config: { providers: { mfq: {
      displayName: 'MFQ', api: 'openai-completions', baseURL: openAI,
      apiKeyEnv: 'MFQ_API_KEY', models: [{ id: model, name: model, input: ['text'] }],
    } } } },
    { id: 'agent-default-model', config: { provider: 'mfq', model } },
  ]);
}
