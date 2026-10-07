import { CaretDownIcon } from '@phosphor-icons/react';
import type { CheckpointAnalysis } from '../../shared/api/types';
import { activationLabel, routerWeightLabel, type Translate } from './analysisData';

export function AdvancedDetails({ analysis, tr }: { analysis: CheckpointAnalysis; tr: Translate }) {
  const info = analysis.advanced;
  const residual = analysis.architecture === 'qwen4_exp' ? 'GR' : 'mHC';
  const kinds = analysis.attention_distribution;
  const fullAttention = Object.keys(kinds).find(kind => ['GQA', 'MQA', 'MHA', 'MLA'].includes(kind)) ?? 'Attention';
  const optional = analysis.graph.optional_components as Record<string, boolean> | undefined;
  const convType = (value?: string | null) => value === 'causal_depthwise_1d' ? tr('因果一维逐通道卷积', 'Causal depthwise 1D') : value;
  const normPosition = (value: string) => {
    const predictor = /^block\.(\d+)\.(attention(?:\.indexer)?|mlp)\.(norm|query_norm|key_norm)$/.exec(value);
    if (predictor) return `L${predictor[1]} · ${predictor[2].includes('indexer') ? 'Indexer' : predictor[2] === 'mlp' ? 'FFN' : fullAttention} · ${predictor[3] === 'query_norm' ? 'Q' : predictor[3] === 'key_norm' ? 'K' : tr('输入', 'Input')}`;
    return ({ query: 'Q', key: 'K', output: tr('输出', 'Output'), input: tr('输入', 'Input'),
      conv: tr('卷积输入', 'Convolution input'), attention_input: tr('注意力读取', 'Attention read'), ffn_input: tr('FFN 读取', 'FFN read'),
      final_readout: tr('最终合流', 'Final readout'), embedding_norm: tr('词嵌入', 'Embedding'), hidden_norm: tr('隐藏状态', 'Hidden states'),
      output_norm: tr('输出', 'Output') } as Record<string, string>)[value] ?? value;
  };
  const normsFor = (id: string) => (info?.normalizations ?? []).filter(norm =>
    (id === 'sparse' ? ['qsa', 'indexer'] : id === 'base' ? ['output'] : [id]).includes(norm.component));
  const groups: { id: string; title: string; rows: [string, string | number | null | undefined][] }[] = info ? [
    { id: 'ffn', title: tr('FFN 与专家', 'FFN & experts'), rows: [
      ...(info.ffn_hidden_size || !analysis.expert_count ? [[tr('稠密 FFN 隐藏维度', 'Dense FFN hidden size'), info.ffn_hidden_size] as [string, number | null | undefined]] : []),
      ...(analysis.expert_count ? [
        [tr('专家隐藏维度', 'Expert hidden size'), info.expert_hidden_size],
        [tr('共享专家数量', 'Shared expert count'), info.shared_expert_count],
        ...(info.shared_expert_count ? [[tr('共享专家隐藏维度', 'Shared expert hidden size'), info.shared_expert_hidden_size]] : []),
      ] as [string, number | null | undefined][] : []),
      [tr('FFN 激活函数', 'FFN activation'), activationLabel(info.ffn_activation)],
      ...(analysis.expert_count ? [[tr('Router 激活函数', 'Router activation'), activationLabel(info.router_activation)] as [string, string]] : []),
      ...(analysis.expert_count && info.router_activation && info.router_activation !== 'softmax' ? [[tr('Router 权重归一化', 'Router weight normalization'), info.router_normalization === 'l1_topk' ? routerWeightLabel(info) : info.router_normalization === 'none' ? tr('无额外归一化', 'No extra normalization') : null]] as [string, string | null][] : []),
    ] },
    ...(info.hc_count && info.hc_count > 1 ? [{ id: 'residual', title: tr('残差与门控', 'Residual streams & gates'), rows: [
      [`${residual} ${tr('残差流数量', 'stream count')}`, info.hc_count],
      [`${residual} ${tr('单流维度', 'stream dimension')}`, info.hc_dim],
      ...(info.hc_lowrank ? [[`${residual} ${tr('瓶颈维度', 'bottleneck dimension')}`, info.hc_lowrank]] : []),
      [`${residual} ${tr('门控激活函数', 'gate activation')}`, activationLabel(info.residual_gate_activation)],
    ] as [string, string | number | null | undefined][] }] : []),
    ...(info.indexer_query_heads || info.indexer_head_dim || kinds.QSA ? [{ id: 'sparse', title: tr('稀疏注意力与 Indexer', 'Sparse attention & indexer'), rows: [
      [tr('压缩块步长', 'Compression block stride'), info.compression_stride ?? (info.compression_strides?.length
        ? [...new Set(info.compression_strides)].join(' / ') + tr('（按层）', ' (by layer)') : null)],
      [tr('激活块上限', 'Active block limit'), info.active_blocks],
      ...(info.active_tokens ? [[tr('激活 token 上限', 'Active token limit'), info.active_tokens]] : []),
      [tr('Indexer Q 头数', 'Indexer Q heads'), info.indexer_query_heads],
      ...(info.indexer_kv_heads ? [[tr('Indexer KV 头数', 'Indexer KV heads'), info.indexer_kv_heads]] : []),
      [tr('Indexer 头维度', 'Indexer head dimension'), info.indexer_head_dim],
      ...(kinds.QSA && analysis.architecture === 'qwen4_exp' ? [[tr('QSA 输出门控激活函数', 'QSA output gate activation'), activationLabel(info.qsa_gate_activation)]] : []),
    ] as [string, string | number | null | undefined][] }] : []),
    ...(kinds.GDN ? [{ id: 'gdn', title: 'GDN', rows: [
      [tr('K 头维度', 'K head dimension'), info.linear_key_head_dim],
      [tr('V 头维度', 'V head dimension'), info.linear_value_head_dim],
      [tr('因果卷积核大小', 'Causal convolution kernel'), info.linear_conv_kernel],
      [tr('卷积类型', 'Convolution type'), convType(info.gdn_conv_type)],
      [tr('卷积步长', 'Convolution stride'), info.gdn_conv_stride],
      [tr('卷积膨胀率', 'Convolution dilation'), info.gdn_conv_dilation],
      [tr('卷积激活函数', 'Convolution activation'), activationLabel(info.gdn_conv_activation)],
      [tr('GDN 输出门控激活函数', 'GDN output gate activation'), activationLabel(info.gdn_gate_activation)],
    ] as [string, string | number | null | undefined][] }] : []),
    ...(info.full_attention_gate_activation || normsFor('attention').length ? [{ id: 'attention', title: fullAttention, rows: [
      ...(info.full_attention_gate_activation ? [[tr('输出门控激活函数', 'Output gate activation'), activationLabel(info.full_attention_gate_activation)]] : []),
    ] as [string, string][] }] : []),
    ...(analysis.graph.has_ple ? [{ id: 'ple', title: 'PLE', rows: [
      ['N-gram', info.ple_ngram], [tr('每个 N-gram 头数', 'Heads per n-gram'), info.ple_heads],
      [tr('卷积核大小', 'Convolution kernel'), info.ple_conv_kernel],
      [tr('卷积类型', 'Convolution type'), convType(info.ple_conv_type)],
      [tr('卷积步长', 'Convolution stride'), info.ple_conv_stride],
      [tr('卷积膨胀率', 'Convolution dilation'), info.ple_conv_dilation],
      [tr('卷积激活函数', 'Convolution activation'), activationLabel(info.ple_conv_activation)],
      [tr('注入层（从 0 开始）', 'Injection layers (zero-based)'), info.ple_layer_ids?.map(layer => `L${layer}`).join(' · ') || null],
    ] as [string, string | number | null | undefined][] }] : []),
    ...(optional?.predictor ? [{ id: 'predictor', title: 'MTP', rows: [[tr('预测层数', 'Predictor layers'), info.predictor_layers]] as [string, number | null | undefined][] }] : []),
    { id: 'base', title: tr('基础配置', 'Base configuration'), rows: [
      [tr('模型最大上下文', 'Model maximum context'), info.max_context], [tr('词表大小', 'Vocabulary size'), info.vocab_size],
      ['RoPE θ', info.rope_theta], [tr('RoPE 旋转比例', 'RoPE rotary fraction'), info.rope_partial_factor == null ? null : `${info.rope_partial_factor * 100}%`],
      ['RMSNorm ε', info.rms_norm_eps == null ? null : info.rms_norm_eps.toExponential()],
    ] },
  ] : [];
  return <details className="analysis-advanced" key={analysis.model_id}>
    <summary>{tr('高级', 'Advanced')}<CaretDownIcon size={14} aria-hidden="true" /></summary>
    <div className="analysis-advanced-body">{groups.length ? groups.map(group => <section key={group.title}>
      <h3>{group.title}</h3><dl>{group.rows.map(([label, value]) => <div key={label}><dt>{label}</dt>
        <dd>{value == null ? '—' : typeof value === 'number' ? value.toLocaleString() : value}</dd></div>)}</dl>
      {normsFor(group.id).length > 0 && <div className="analysis-normalizations"><h4>Norm</h4><dl>{normsFor(group.id).map(norm =>
        <div key={[norm.component, norm.position, norm.kind, norm.dimension, norm.epsilon, norm.groups].join(':')}>
          <dt>{norm.component === 'indexer' ? 'Indexer · ' : ''}{normPosition(norm.position)}
            {normsFor(group.id).filter(item => item.component === norm.component && item.position === norm.position).length > 1 && ` · ${norm.layers.map(layer => `L${layer}`).join(', ')}`}</dt>
          <dd>{norm.kind ?? '—'}<small>{[
            norm.dimension == null ? null : `d = ${norm.dimension.toLocaleString()}`,
            norm.epsilon == null ? null : `ε = ${norm.epsilon.toExponential()}`,
            norm.groups == null ? null : tr(`${norm.groups} 组`, `${norm.groups} groups`),
          ].filter(Boolean).join(' · ')}</small></dd>
        </div>)}</dl></div>}
    </section>) : <p>{tr('模型未提供高级元数据。', 'Advanced model metadata is unavailable.')}</p>}</div>
  </details>;
}
