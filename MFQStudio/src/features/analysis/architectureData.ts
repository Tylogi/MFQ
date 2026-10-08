import type { CheckpointAnalysis } from '../../shared/api/types';
import { activationLabel, parameterLabel, routerWeightLabel, type Translate } from './analysisData';

export interface ArchitectureNode {
  id: string; x: number; y: number; width: number; height: number;
  title: string; detail?: string; action?: string; accent?: boolean;
}
export interface ArchitectureEdge { from: string; to: string; path?: string; residual?: boolean }
export interface ArchitectureView {
  id: string; title: string; description: string; height: number;
  nodes: ArchitectureNode[]; edges: ArchitectureEdge[];
  frames?: { x: number; y: number; width: number; height: number; title: string }[];
}

export function architectureReference(analysis: CheckpointAnalysis) {
  if (analysis.architecture === 'qwen4_exp') return { href: 'https://huggingface.co/Qwen/Qwen3.8-Flash-Next', name: 'Qwen3.8-Flash-Next' };
  if (/qwen3\.8.*27b/i.test(analysis.name)) return { href: 'https://huggingface.co/Qwen/Qwen3.8-27B', name: 'Qwen3.8-27B' };
  if (analysis.architecture === 'deepseek_v4') return { href: 'https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash', name: 'DeepSeek-V4' };
  return null;
}

export function architectureSchedule(analysis: CheckpointAnalysis): string {
  const kinds = analysis.layers.map(layer => layer.attention_type);
  for (let period = 1; period <= Math.min(16, kinds.length / 2); period++) {
    if (kinds.length % period || !kinds.every((kind, index) => kind === kinds[index % period])) continue;
    const runs: { kind: string; count: number }[] = [];
    for (const kind of kinds.slice(0, period)) {
      if (runs.at(-1)?.kind === kind) runs[runs.length - 1].count++;
      else runs.push({ kind, count: 1 });
    }
    return `${kinds.length / period} × [${runs.map(run => run.count > 1 ? `${run.count} × ${run.kind}` : run.kind).join(' → ')}]`;
  }
  return Object.entries(analysis.attention_distribution).map(([kind, ids]) => `${kind} × ${ids.length}`).join(' · ');
}

export function architectureViews(analysis: CheckpointAnalysis, tr: Translate): ArchitectureView[] {
  const qwenGr = analysis.architecture === 'qwen4_exp';
  const hasMhc = Number(analysis.graph.hc_count) > 1;
  const residual = hasMhc ? qwenGr ? 'GR' : 'mHC' : tr('残差', 'Residual');
  const ff = analysis.expert_count ? 'MoE' : 'FFN';
  const ffnActivation = analysis.advanced?.ffn_activation ? activationLabel(analysis.advanced.ffn_activation) : tr('激活函数', 'Activation');
  const gateLabel = (activation?: string | null) => activation ? `${activationLabel(activation)}(Gate)` : 'Gate';
  const pleLayers = [...new Set(analysis.tensors.filter(item => ['ple', 'ple_projection'].includes(item.category) && item.layer != null).map(item => item.layer!))].sort((a, b) => a - b);
  const hasPle = Boolean(analysis.graph.has_ple) || pleLayers.length > 0;
  const optional = analysis.graph.optional_components as Record<string, boolean> | undefined;
  const hasMtp = Boolean(optional?.predictor);
  const hidden = analysis.hidden_size?.toLocaleString() ?? '—';
  const branchInput = qwenGr ? tr('来自 GR 的隐藏状态', 'Hidden state from GR') : 'RMSNorm';
  const pleNorm = qwenGr ? 'Grouped RMSNorm' : 'Norm';
  const gdnNorm = analysis.advanced?.normalizations?.find(norm => norm.component === 'gdn' && norm.position === 'output')?.kind ?? 'RMSNorm';
  const hasGroupedQk = analysis.advanced?.normalizations?.some(norm => norm.component === 'attention' && ['query', 'key'].includes(norm.position) && norm.kind === 'GroupedRMSNorm');
  const node = (id: string, y: number, title: string, detail?: string, action?: string, x = 60, width = 240): ArchitectureNode =>
    ({ id, x, y, width, height: detail ? 50 : 36, title, detail, action, accent: Boolean(action) });
  const edge = (from: string, to: string, path?: string, residual = false): ArchitectureEdge => ({ from, to, path, residual });
  const read = hasMhc ? `${residual} · ${tr('读取', 'Read')}` : 'RMSNorm';
  const write = hasMhc ? `${residual} · ${tr('写回', 'Write')}` : tr('残差相加', 'Residual add');
  const kinds = Object.entries(analysis.attention_distribution);
  const fullAttention = kinds.find(([kind]) => ['GQA', 'MQA', 'MHA', 'MLA'].includes(kind))?.[0];
  const offset = hasPle ? 62 : 0;
  const nodes = [
    node('ids', 8, 'Token IDs'),
    node('embedding', 66, optional?.vision ? tr('词嵌入 + 视觉特征', 'Embedding + vision features') : tr('词嵌入', 'Token embedding'), `d = ${hidden}`),
    node('streams', 138, hasMhc ? `${analysis.graph.hc_count} ${tr('路残差流', 'residual streams')}` : tr('隐藏状态', 'Hidden states')),
    ...(hasPle ? [node('ple', 224, 'N-gram / PLE', pleLayers.map(layer => `L${layer}`).join(' · ') || tr('指定层注入', 'Selected layers only'), 'ple')] : []),
    node('read-attn', 224 + offset, read, undefined, hasMhc ? 'residual' : undefined),
    node('write-attn', 370 + offset, write, undefined, hasMhc ? 'residual' : undefined),
    node('read-ffn', 430 + offset, read, undefined, hasMhc ? 'residual' : undefined),
    node('ffn', 490 + offset, ff, analysis.expert_count ? `${analysis.expert_count} ${tr('专家', 'experts')} · Top ${analysis.experts_per_token ?? '—'}${analysis.graph.has_shared_expert ? ' + Shared' : ''}` : `${ffnActivation}(Gate) ⊙ Up → Down`, 'ffn'),
    node('write-ffn', 562 + offset, write, undefined, hasMhc ? 'residual' : undefined),
    node('final', 650 + offset, hasMhc ? `${residual} · ${tr('最终合流', 'Final readout')}` : 'Final RMSNorm', undefined, hasMhc ? 'residual' : undefined),
    node('head', 710 + offset, hasMhc ? 'RMSNorm · LM Head' : 'LM Head', tr('输出词表投影', 'Vocabulary projection')),
    node('logits', 786 + offset, 'Logits'),
    ...(hasMtp ? [node('mtp', 856 + offset, 'MTP', tr('隐藏状态 + 后续 token → 草稿', 'Hidden state + following token → draft'), 'mtp')] : []),
  ];
  const edges = [edge('ids', 'embedding'), edge('embedding', 'streams'), edge('streams', hasPle ? 'ple' : 'read-attn'),
    ...(hasPle ? [edge('ple', 'read-attn'), edge('ids', 'ple', 'M60 26H16V249H60', true)] : []),
    edge('write-attn', 'read-ffn'), edge('read-ffn', 'ffn'), edge('ffn', 'write-ffn'), edge('write-ffn', 'final'),
    edge('final', 'head'), edge('head', 'logits'),
    edge(hasMhc ? 'read-attn' : hasPle ? 'ple' : 'streams', 'write-attn', `M300 ${hasMhc ? 242 + offset : hasPle ? 249 : 156}H336V${388 + offset}H300`, true),
    edge(hasMhc ? 'read-ffn' : 'write-attn', 'write-ffn', `M300 ${(hasMhc ? 448 : 388) + offset}H336V${580 + offset}H300`, true),
    ...(hasMtp ? [edge('write-ffn', 'mtp', `M300 ${580 + offset}H348V${881 + offset}H300`, true)] : [])];
  if (kinds.length <= 2) {
    kinds.forEach(([kind, ids], index) => {
      const id = `attention-${index}`;
      const action = kind === 'GDN' ? 'gdn' : kind === 'QSA' && qwenGr ? 'qsa' : 'attention';
      nodes.push(node(id, 296 + offset, kind === 'QSA' && !qwenGr ? 'Sparse Attention' : kind, `${ids.length} ${tr('层', 'layers')}`, action, kinds.length === 1 ? 60 : 20 + index * 170, kinds.length === 1 ? 240 : 150));
      edges.push(edge('read-attn', id), edge(id, 'write-attn'));
    });
  } else {
    nodes.push(node('attention', 296 + offset, tr('注意力分派', 'Attention dispatch'), architectureSchedule(analysis), 'attention'));
    edges.push(edge('read-attn', 'attention'), edge('attention', 'write-attn'));
  }
  const views: ArchitectureView[] = [{
    id: 'overview', title: tr('总览', 'Overview'), height: (hasMtp ? 930 : 844) + offset,
    description: tr('实线为计算路径，虚线为残差或条件分支；注意力按层类型切换，不在同一层并行执行。层号从 0 开始。', 'Solid lines show computation; dashed lines show residual or conditional branches. Attention types alternate by layer, not within one layer. Layer IDs are zero-based.'),
    nodes, edges, frames: [{ x: 8, y: 194, width: 344, height: 430 + offset, title: `${tr('解码器层', 'Decoder layers')} × ${analysis.layer_count}` }],
  }];
  if (analysis.attention_distribution.GDN) views.push({
    id: 'gdn', title: 'GDN', height: 626,
    description: tr('Q/K/V 经过因果卷积；α、β 调控递推状态，输出再做归一化、门控与下投影。', 'Q/K/V pass through causal convolution; α and β control the recurrent state, followed by normalized, gated output projection.'),
    nodes: [node('in', 10, branchInput, `d = ${hidden}`), node('qkv', 94, 'Q · K · V', undefined, undefined, 18, 152),
      node('gates', 94, 'Gate · α · β', undefined, undefined, 190, 152),
      node('conv', 164, 'Causal Conv', activationLabel(analysis.advanced?.gdn_conv_activation), undefined, 18, 152),
      node('qknorm', 234, 'Q / K L2Norm', undefined, undefined, 18, 152),
      node('state', 316, 'Gated DeltaNet', `QK ${analysis.linear_key_heads ?? '—'} · V ${analysis.linear_value_heads ?? '—'} ${tr('头', 'heads')}`),
      node('norm', 404, gdnNorm, `⊙ ${gateLabel(analysis.advanced?.gdn_gate_activation)}`), node('out', 482, 'Output projection', `→ d = ${hidden}`), node('result', 570, tr('注意力分支输出', 'Attention branch output'))],
    edges: [edge('in', 'qkv'), edge('in', 'gates'), edge('qkv', 'conv'), edge('conv', 'qknorm'), edge('qknorm', 'state'),
      edge('gates', 'state'), edge('gates', 'norm', 'M342 112H354V429H300'),
      edge('state', 'state', 'M60 340H10V390H44V352H60', true), edge('state', 'norm'), edge('norm', 'out'), edge('out', 'result')],
  });
  if (qwenGr && analysis.attention_distribution.QSA) views.push({
    id: 'qsa', title: 'QSA', height: 690,
    description: tr('轻量索引器按微块选上下文；主 Q/K/V 在选定 KV 上做注意力，之后施加输出门控。', 'The lightweight indexer selects context micro-blocks; main Q/K/V attend to selected KV, followed by output gating.'),
    nodes: [node('in', 10, branchInput, `d = ${hidden}`),
      node('qkv', 100, 'Q · K · V', `Q ${analysis.attention_heads ?? '—'} / KV ${analysis.kv_heads ?? '—'}`, undefined, 18, 152),
      node('index', 100, 'Indexer Q / K', undefined, undefined, 190, 152),
      node('rope', 190, 'Q/K Norm · RoPE', 'GroupedRMSNorm', undefined, 18, 152),
      node('blocks', 190, tr('微块压缩与打分', 'Micro-block scores'), 'Q/K GroupedRMSNorm', undefined, 190, 152),
      node('cache', 274, 'KV Cache', undefined, undefined, 18, 152), node('topk', 274, 'Top-k blocks', undefined, undefined, 190, 152),
      node('attn', 366, 'Selected KV Attention', 'Scale · Mask · Softmax'),
      node('gate', 458, `⊙ ${gateLabel(analysis.advanced?.qsa_gate_activation)}`), node('out', 536, 'Output projection', `→ d = ${hidden}`), node('result', 626, tr('注意力分支输出', 'Attention branch output'))],
    edges: [edge('in', 'qkv'), edge('in', 'index'), edge('qkv', 'rope'), edge('index', 'blocks'), edge('rope', 'cache'), edge('blocks', 'topk'),
      edge('cache', 'attn'), edge('topk', 'attn'), edge('rope', 'attn', 'M18 208H8V391H60'),
      edge('in', 'gate', 'M300 35H354V476H300'), edge('attn', 'gate'), edge('gate', 'out'), edge('out', 'result')],
  });
  if (kinds.some(([kind]) => kind !== 'GDN' && !(kind === 'QSA' && qwenGr))) {
    const latent = analysis.tensors.some(item => item.category === 'attention' && /lora|latent|compressor/.test(item.name));
    const sparse = kinds.some(([kind]) => kind === 'QSA') || analysis.tensors.some(item => /indexer/.test(item.name));
    const query = analysis.tensors.find(item => item.category === 'attention' && item.projection === 'query');
    const gated = analysis.tensors.some(item => item.category === 'attention' && item.projection === 'gate') ||
      (/qwen/.test(analysis.architecture) && query && analysis.attention_heads && analysis.head_dim &&
        query.shape.at(-2) === 2 * analysis.attention_heads * analysis.head_dim);
    views.push({
      id: 'attention', title: fullAttention ?? tr('注意力', 'Attention'), height: gated ? 536 : 490,
      description: tr('按检查点中实际的注意力投影与索引器绘制；不存在的稀疏索引或低秩投影不会添加。', 'Drawn from the checkpoint attention projections and indexer; absent sparse or low-rank branches are not added.'),
      nodes: [node('in', 10, 'RMSNorm', `d = ${hidden}`), node('qkv', 94, latent ? tr('低秩 / 压缩 QKV', 'Low-rank / compressed QKV') : 'Q · K · V', `Q ${analysis.attention_heads ?? '—'} / KV ${analysis.kv_heads ?? '—'}`),
        node('cache', 186, hasGroupedQk ? 'Q/K Norm · RoPE' : 'RoPE · KV Cache', hasGroupedQk ? 'GroupedRMSNorm · KV Cache' : undefined), node('attn', 270, sparse ? 'Sparse KV Attention' : fullAttention ?? 'Scaled Dot-Product', sparse ? 'Indexer → Select KV → Softmax' : 'Scale · Mask · Softmax'),
        ...(gated ? [node('gate', 352, `⊙ ${gateLabel(analysis.advanced?.full_attention_gate_activation)}`)] : []),
        node('out', gated ? 430 : 366, 'Output projection', `→ d = ${hidden}`)],
      edges: [edge('in', 'qkv'), edge('qkv', 'cache'), edge('cache', 'attn'), edge('attn', gated ? 'gate' : 'out'),
        ...(gated ? [edge('gate', 'out'), edge('in', 'gate', 'M300 35H348V370H300')] : [])],
    });
  }
  views.push({
    id: 'ffn', title: ff, height: analysis.expert_count ? 636 : 500,
    description: analysis.expert_count ? tr('路由选择专家；每个专家内部独立执行 Gate/Up/Down，再按路由权重归并。共享专家为并行分支。', 'Routing selects experts; each expert computes Gate/Up/Down and outputs are route-weighted. Shared experts form a parallel branch.')
      : tr(`稠密前馈网络：Gate 经 ${ffnActivation} 后与 Up 逐元素相乘，最后由 Down 投影回隐藏维度。`, `Dense FFN: ${ffnActivation}(Gate) multiplies Up elementwise, then Down projects back to the hidden dimension.`),
    nodes: analysis.expert_count ? [
      node('in', 10, branchInput, `d = ${hidden}`), node('router', 94, 'Router · Top-k', `${routerWeightLabel(analysis.advanced)} · ${analysis.experts_per_token ?? '—'} / ${analysis.expert_count}`, undefined, 18, 152),
      ...(analysis.graph.has_shared_expert ? [node('shared', 94, 'Shared expert', 'Gate · Up · Down', undefined, 190, 152)] : []),
      node('gate', 208, 'Gate', ffnActivation, undefined, 18, 152), node('up', 208, 'Up', undefined, undefined, 190, 152),
      node('multiply', 302, `${ffnActivation}(Gate) ⊙ Up`), node('down', 376, 'Down projection'),
      node('reduce', 448, tr('路由权重归并', 'Route-weighted reduce')),
      node('sum', 524, analysis.graph.has_shared_expert ? tr('+ 门控共享专家输出', '+ Gated shared output') : tr('专家输出', 'Expert output'), `→ d = ${hidden}`),
    ] : [node('in', 10, branchInput, `d = ${hidden}`), node('gate', 104, 'Gate', ffnActivation, undefined, 18, 152),
      node('up', 104, 'Up', undefined, undefined, 190, 152), node('multiply', 206, `${ffnActivation}(Gate) ⊙ Up`),
      node('down', 290, 'Down projection'), node('sum', 376, tr('前馈分支输出', 'FFN branch output'), `→ d = ${hidden}`)],
    edges: analysis.expert_count ? [edge('in', 'router'), edge('router', 'gate'), edge('router', 'up'), edge('gate', 'multiply'),
      edge('up', 'multiply'), edge('multiply', 'down'), edge('down', 'reduce'), edge('reduce', 'sum'),
      edge('router', 'reduce', 'M18 119H8V466H60', true),
      ...(analysis.graph.has_shared_expert ? [edge('in', 'shared'), edge('shared', 'sum', 'M342 119H354V549H300')] : [])]
      : [edge('in', 'gate'), edge('in', 'up'), edge('gate', 'multiply'), edge('up', 'multiply'), edge('multiply', 'down'), edge('down', 'sum')],
  });
  if (hasMhc) views.push({
    id: 'residual', title: residual, height: 626,
    description: qwenGr ? tr('GR 在每个 Attention / MoE 子层前读取多路残差流，子层后按写入门控注入；最终单独合流。', 'GR reads the residual streams before each Attention/MoE sublayer and injects gated writes afterward; final readout merges the streams.')
      : tr('mHC 对多路残差进行读取、混合与写回，包裹注意力及前馈子层。', 'mHC reads, mixes and writes multiple residual streams around attention and feed-forward sublayers.'),
    nodes: [node('in', 10, tr('多路残差状态', 'Residual streams'), `${analysis.graph.hc_count} × ${hidden}`),
      node('gate', 104, qwenGr ? 'Down → SiLU → Up' : tr('残差混合与门控', 'Residual mix & gates'), qwenGr ? 'Grouped RMSNorm' : tr('根据隐藏状态生成门控', 'Data-dependent gates')),
      node('read', 208, qwenGr ? tr('逐元素读门控 · 合流', 'Read gate · stream reduction') : 'Pre gate · stream reduction', qwenGr ? 'Sigmoid' : undefined),
      node('branch', 300, 'Attention / MoE / FFN', tr('执行当前子层', 'Current sublayer')),
      node('write', 400, qwenGr ? tr('逐路写门控 · 注入', 'Per-stream write gate') : 'Post gate · injection', qwenGr ? '2 × Sigmoid' : undefined),
      node('out', 508, tr('新残差状态', 'Updated residual streams'), `${analysis.graph.hc_count} × ${hidden}`)],
    edges: [edge('in', 'gate'), edge('gate', 'read'), edge('read', 'branch'), edge('branch', 'write'), edge('write', 'out'),
      edge('in', 'out', 'M60 35H12V533H60', true), edge('gate', 'write', 'M300 129H348V418H300')],
  });
  if (hasPle) {
    const parameters = analysis.tensors.filter(item => item.category === 'ple').reduce((sum, item) => sum + item.parameters, 0);
    views.push({
      id: 'ple', title: 'PLE', height: 692,
      description: tr('PLE 从局部 token 历史哈希查表，通过当前隐藏状态门控，再经卷积注入指定层；不是每层都执行。', 'PLE hashes local token history for lookup, gates it using the current hidden state, then injects through convolution at selected layers, not every layer.'),
      nodes: [node('ids', 10, tr('局部 token 历史', 'Local token history')), node('hash', 76, 'N-gram Hash → Lookup', parameters ? `${parameterLabel(parameters)} ${tr('参数', 'parameters')}` : 'PLE table'),
        node('key', 184, 'Key projection', pleNorm, undefined, 18, 152), node('value', 184, 'Value projection', undefined, undefined, 190, 152),
        node('hidden', 276, tr('残差流 · Query', 'Residual Query'), pleNorm, undefined, 190, 152),
        node('gate', 370, tr('Query · Key 匹配门控', 'Query · Key matching gate'), 'Sigmoid → Gate ⊙ Value'),
        node('conv', 464, 'Dilated Conv · SiLU', pleNorm), node('sum', 546, tr('+ 门控嵌入 · 注入残差', '+ Gated embedding · residual injection'), pleLayers.map(layer => `L${layer}`).join(' · ') || tr('指定层', 'Selected layers')),
      ], edges: [edge('ids', 'hash'), edge('hash', 'key'), edge('hash', 'value'), edge('key', 'gate'), edge('hidden', 'gate'),
        edge('value', 'gate', 'M342 202H354V395H300'), edge('gate', 'conv'), edge('conv', 'sum'),
        edge('gate', 'sum', 'M60 395H8V571H60', true)],
    });
  }
  if (hasMtp) {
    const topology = analysis.graph.topology as Record<string, number> | undefined;
    views.push({
      id: 'mtp', title: 'MTP', height: 510,
      description: tr('MTP 预测分支读取主模型隐藏状态与后续 token 嵌入，生成草稿，再由主模型验证；不是额外的主干层。', 'MTP reads backbone hidden states and subsequent token embeddings to draft candidates verified by the main model; it is not an extra backbone layer.'),
      nodes: [node('hidden', 10, tr('主干隐藏状态', 'Backbone hidden state'), undefined, undefined, 18, 152),
        node('embedding', 10, tr('后续 token 嵌入', 'Token embedding'), undefined, undefined, 190, 152),
        node('merge', 104, tr('归一化 · 合并投影', 'Normalize · merge projection')),
        node('predictor', 188, tr('预测网络', 'Prediction network'), topology?.predictor_layers ? `${topology.predictor_layers} ${tr('层', 'layers')}` : tr('检查点预测分支', 'Checkpoint predictor')),
        node('head', 282, 'Norm · LM Head'), node('draft', 356, tr('草稿 token', 'Draft tokens')),
        node('verify', 432, tr('主模型验证 · 接受 / 拒绝', 'Target verification · accept / reject'))],
      edges: [edge('hidden', 'merge'), edge('embedding', 'merge'), edge('merge', 'predictor'), edge('predictor', 'head'), edge('head', 'draft'), edge('draft', 'verify')],
    });
  }
  return views;
}
