// Run with: node docs/figures/generate-expert-allocation-comparison.mjs
// The checked-in snapshot contains real layer-0 allocation metadata; no weights
// or network access are needed to regenerate the English and Chinese SVGs.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';

const source = new URL('./data/qwen38-flash-next-layer0-allocation.json', import.meta.url);
const data = JSON.parse(fs.readFileSync(source, 'utf8'));
const formatData = JSON.parse(fs.readFileSync(new URL('./data/qwen38-flash-next-base-format-sse.json', import.meta.url), 'utf8'));
const recipeText = fs.readFileSync(new URL('../../mfq/calibration/recipes/IQ3_XXS.json', import.meta.url), 'utf8');
const recipe = JSON.parse(recipeText);
const expertIds = Array.from({length: 12}, (_, id) => id);
const projections = ['gate', 'up', 'down'];
const nExperts = data.mfq.experts_per_layer;
const nLayers = data.mfq.layer_count;
const repositoryUrl = 'https://github.com/Tylogi/MFQ';
assert.equal(nExperts, 512);
assert.equal(nLayers, 48);
assert.equal(formatData.model, data.mfq.model);
assert.equal(formatData.evaluation.layer, data.mfq.layer);
assert.equal(formatData.evaluation.experts, nExperts);
assert.equal(formatData.evaluation.matrices, nExperts * projections.length);
assert.equal(formatData.measurements.length, 7);
for (const row of formatData.measurements) {
  assert(Math.abs(100 * (1 - row.mfq_sse / row.gguf_sse) - row.sse_reduction_percent) < 0.005);
}
assert.equal(recipe.revision, data.gguf.revision);
assert.equal(createHash('sha256').update(recipeText).digest('hex'), data.gguf.recipe_sha256);

const theme = JSON.parse(fs.readFileSync(new URL('./studio-figure-theme.json', import.meta.url), 'utf8'));
const c = theme.colors;
const colors = theme.profiles;
for (const projection of data.mfq.projections) {
  assert.equal(projection.profiles.length, nExperts);
  const covered = new Set();
  for (const pool of projection.pools) {
    for (const id of pool.expert_ids) {
      assert(!covered.has(id));
      covered.add(id);
      assert.equal(projection.profiles[id], pool.profile);
    }
  }
  assert.equal(covered.size, nExperts);
  for (const profile of projection.profiles) assert(colors[profile], `Unknown profile: ${profile}`);
  const tensor = `blk.${data.mfq.layer}.ffn_${projection.projection}_exps.weight`;
  assert.equal(recipe.tensor_types[tensor], data.gguf.tensor_types[projection.projection]);
}

const language = {
  en: {
    title: 'Fine-grained precision & format fidelity',
    context: `${data.mfq.model} · Layer ${data.mfq.layer} · ${nExperts} experts`,
    description: 'Qwen3.8-Flash-Next, layer 0. The allocation panels show experts 0–11: this packed GGUF recipe uses one format per projection tensor; MFQ selects profiles per expert projection. Colors distinguish profiles. GGUF block-level parameters can still vary. The base-format panel shows mean reconstruction SSE reductions over all 512 experts, with actual bpw for each of seven format pairs.',
    slots: 'allocation slots / layer',
    expert: 'Expert',
    formatTitle: 'Base-format reconstruction',
    formatScope: 'Layer 0 · 512 experts × Gate / Up / Down · 1,536 matrices · equal matrix weighting',
    mfqFormat: 'MFQ format · actual bpw',
    ggufFormat: 'GGUF format · actual bpw',
    reduction: 'SSE reduction',
    maximum: value => `Up to ${value.toFixed(2)}% lower SSE`,
  },
  'zh-CN': {
    title: '细粒度精度与基础格式保真度',
    context: `${data.mfq.model} · 第 ${data.mfq.layer} 层 · ${nExperts} 个专家`,
    description: 'Qwen3.8-Flash-Next 第 0 层。分配面板展示专家 0–11 的真实档位：该 GGUF 配方的每个打包投影张量共用一种格式，MFQ 则按专家投影分配档位。颜色区分档位，GGUF 块内参数仍可变化。基础格式面板展示全部 512 个专家的平均重构 SSE 降低幅度，并列出 7 组格式的实际 bpw。',
    slots: '个分配位置／层',
    expert: '专家',
    formatTitle: '基础格式的重构误差',
    formatScope: '第 0 层 · 512 个专家 × Gate / Up / Down · 1,536 个矩阵 · 各矩阵等权平均',
    mfqFormat: 'MFQ 格式 · 实际 bpw',
    ggufFormat: 'GGUF 格式 · 实际 bpw',
    reduction: 'SSE 降低',
    maximum: value => `最高降低 ${value.toFixed(2)}% SSE`,
  },
};

const xml = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&apos;'}[c]));
for (const [locale, t] of Object.entries(language)) {
  const parts = [];
  const text = (x, y, value, size = 18, weight = 400, fill = c.ink, anchor = 'start') =>
    parts.push(`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" fill="${fill}" text-anchor="${anchor}">${xml(value)}</text>`);
  const rect = (x, y, width, height, fill, stroke = 'none', radius = 8) =>
    parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="${radius}" fill="${fill}" stroke="${stroke}"/>`);

  const line = (x1, y, x2, stroke = c.line_soft) =>
    parts.push(`<path d="M${x1} ${y}H${x2}" fill="none" stroke="${stroke}"/>`);

  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1280" viewBox="0 0 1600 1280" role="img" aria-labelledby="title desc" xml:lang="${locale}">`);
  parts.push(`<title id="title">${xml(t.title)}</title>`);
  parts.push(`<desc id="desc">${xml(t.description)}</desc>`);
  parts.push(`<metadata>${xml(JSON.stringify({model:data.mfq.model, layer:data.mfq.layer, expert_ids:expertIds, mfq_revision:data.mfq.revision, gguf_revision:data.gguf.revision, repository:repositoryUrl, source:'data/qwen38-flash-next-layer0-allocation.json', base_format_source:'data/qwen38-flash-next-base-format-sse.json', base_format_evaluation:formatData.evaluation, theme:'studio-figure-theme.json'}))}</metadata>`);
  rect(0, 0, 1600, 1280, c.canvas, 'none', 0);
  parts.push('<g font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Noto Sans CJK SC, Helvetica Neue, Arial, sans-serif">');
  text(48, 63, t.title, 30, 700, c.strong);
  text(48, 96, t.context, 17, 400, c.muted);

  // The small brand mark and neutral panels follow MFQ Studio's UI.
  parts.push('<g transform="translate(1434 38) scale(0.5)">');
  rect(1, 1, 62, 62, '#0e1010', 'none', 17);
  parts.push('<path d="M16 44V20L32 37 48 20V44" fill="none" stroke="#f1f2ef" stroke-width="5.5" stroke-linecap="round" stroke-linejoin="round"/>');
  parts.push('<circle cx="48" cy="44" r="3" fill="#d6aa73"/></g>');
  text(1480, 62, 'MFQ', 21, 700, c.strong);

  const gridX = 190, cellWidth = 108, gridWidth = 12*cellWidth;
  const heading = (y, name, recipeName, slots, isMfq) => {
    rect(48, y, 1504, 250, c.surface, c.line, 12);
    text(76, y+41, name, 24, 700, c.strong);
    text(name === 'MFQ' ? 151 : 164, y+40, recipeName, 17, 400, c.muted);
    text(1312, y+42, slots, 30, 600, isMfq ? c.accent : c.ink, 'end');
    text(1332, y+40, t.slots, 15, 400, c.muted);
    line(76, y+64, 1524);
    text(76, y+93, t.expert, 15, 400, c.muted_light);
    for (const id of expertIds) text(gridX + id*cellWidth + cellWidth/2, y+93, id, 15, 400, c.muted_light, 'middle');
  };
  heading(128, 'GGUF', 'Unsloth UD · IQ3_XXS', '3', false);
  for (const [row, projection] of projections.entries()) {
    const y = 238 + row*44;
    const label = data.gguf.tensor_types[projection];
    const [fill, ink] = colors[label];
    text(76, y+25, projection[0].toUpperCase()+projection.slice(1), 18, 500, c.soft);
    rect(gridX+2, y, gridWidth-5, 36, fill, 'none', 4);
    text(gridX + gridWidth/2, y+25, label, 17, 500, ink, 'middle');
  }
  line(48, 409, 734, c.line);
  text(800, 417, '512×', 23, 600, c.accent, 'middle');
  line(866, 409, 1552, c.line);

  heading(440, 'MFQ', 'V3-XXS', '1,536', true);
  for (const [row, projection] of projections.entries()) {
    const y = 550 + row*44;
    const tensor = data.mfq.projections.find(p => p.projection === projection);
    text(76, y+25, projection[0].toUpperCase()+projection.slice(1), 18, 500, c.soft);
    for (const id of expertIds) {
      const x = gridX + id*cellWidth;
      const profile = tensor.profiles[id];
      const [fill, ink] = colors[profile];
      parts.push(`<g data-projection="${projection}" data-expert="${id}" data-profile="${profile}">`);
      rect(x+2, y, cellWidth-5, 36, fill, 'none', 4);
      text(x+cellWidth/2, y+25, profile, 14.5, 500, ink, 'middle');
      parts.push('</g>');
    }
  }
  rect(48, 716, 1504, 496, c.surface, c.line, 12);
  text(76, 760, t.formatTitle, 25, 700, c.strong);
  text(1524, 760, t.maximum(Math.max(...formatData.measurements.map(row => row.sse_reduction_percent))), 23, 700, c.accent_ink, 'end');
  text(76, 790, t.formatScope, 16, 400, c.muted);
  text(76, 828, t.mfqFormat, 16, 400, c.muted);
  text(430, 828, t.ggufFormat, 16, 400, c.muted);
  text(800, 828, t.reduction, 16, 400, c.muted);
  for (const value of [0, 10, 20, 30]) {
    const x = 800 + value / 30 * 640;
    parts.push(`<path d="M${x} 842V1152" fill="none" stroke="${c.line_soft}"/>`);
    text(x, 1186, `${value}%`, 14, 400, c.muted_light, 'middle');
  }
  for (const [index, row] of formatData.measurements.entries()) {
    const y = 864 + index * 44;
    text(76, y-3, row.mfq_format, 19, 700, c.accent_ink);
    text(76, y+17, row.mfq_bpw.toFixed(4), 16, 400, c.muted);
    text(430, y-3, row.gguf_format, 19, 400, c.soft);
    text(430, y+17, row.gguf_bpw.toFixed(4), 16, 400, c.muted);
    parts.push(`<g id="base-format-sse-${row.gguf_format}" data-reduction-percent="${row.sse_reduction_percent}" data-mfq-bpw="${row.mfq_bpw}" data-gguf-bpw="${row.gguf_bpw}" data-mfq-sse="${row.mfq_sse}" data-gguf-sse="${row.gguf_sse}">`);
    parts.push(`<title>${xml(`${row.mfq_format} (${row.mfq_bpw.toFixed(4)} bpw), SSE ${row.mfq_sse}; ${row.gguf_format} (${row.gguf_bpw.toFixed(4)} bpw), SSE ${row.gguf_sse}; reduction ${row.sse_reduction_percent.toFixed(2)}%`)}</title>`);
    const width = row.sse_reduction_percent / 30 * 640;
    rect(800, y-9, width, 18, c.accent, 'none', 3);
    text(808 + width, y+6, `${row.sse_reduction_percent.toFixed(2)}%`, 19, 700, c.accent_ink);
    parts.push('</g>');
  }
  parts.push(`<a id="repository-url" href="${repositoryUrl}">`);
  text(1552, 1254, repositoryUrl, 14, 400, c.muted_light, 'end');
  parts.push('</a></g></svg>');
  const suffix = locale === 'en' ? '' : '.zh-CN';
  const output = new URL(`./qwen38-flash-next-allocation-comparison${suffix}.svg`, import.meta.url);
  fs.writeFileSync(output, parts.join('\n')+'\n');
  console.log(`Wrote ${output.pathname}`);
}
