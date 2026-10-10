"""Frozen released-profile geometry: canonical codec + independent FP64 oracle."""
import math
from pathlib import Path
import sys
import numpy as np

SOURCE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE))
from mfq.formats import io
from mfq.formats.header import FileHeader
from mfq.formats.nint import NintSpec, NintTensor
from mfq.formats.nvq import (NvqJscTensor, NVQ2_E8, NVQ2_E8_1024,
    NVQ2_E8_4096, NVQ3_D4, NVQ3_D4_512, NVQ3_D4_1024, codebook_for)
from mfq.formats.nvq1_l import Nvq1LTensor, NVQ1_L_T8_S4
from mfq.formats.nvq1_s import Nvq1STensor, Nvq1SSpec
from mfq.quantize.nvq1_l_quant import dequantize as decode_l
from mfq.quantize.nvq1_s_quant import dequantize as decode_s
from mfq.quantize.nvq_jsc import dequantize_nvq_jsc

if len(sys.argv) != 2:
    raise SystemExit('usage: make_cpu_quant_fixtures.py OUTPUT_DIRECTORY')
ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
random = np.random.default_rng(20261005)

def nint(n, k, bits, gs, sub_bits):
    groups = math.ceil(k/gs)
    return NintTensor(spec=NintSpec(bits, gs, sub_bits), shape=(n,k), axis=0,
        q=random.integers(0, 1<<bits, (n,groups,gs), dtype=np.uint8),
        neuron_scale=random.uniform(.002,.02,n).astype(np.float16).astype(np.float32)/(1<<bits),
        neuron_min=random.uniform(.0001,.002,n).astype(np.float16).astype(np.float32),
        sub_scale=random.integers(0, 1<<sub_bits, (n,groups), dtype=np.uint8),
        sub_min=random.integers(0, 1<<sub_bits, (n,groups), dtype=np.uint8), neuron_len=k)

def onebit(n, k, small):
    cls, spec = (Nvq1STensor, Nvq1SSpec()) if small else (Nvq1LTensor, NVQ1_L_T8_S4)
    groups=math.ceil(k/24)
    return cls(spec=spec, shape=(n,k),axis=0,neuron_len=k,
        neuron_scale=random.uniform(.002,.02,n).astype(np.float16).astype(np.float32),
        sub_scale=random.integers(0,16,(n,groups),dtype=np.uint8),
        indices=random.integers(0,512 if small else 2048,(n,k//8),dtype=np.uint16),
        delta_sign=random.integers(0,2,(n,groups),dtype=np.uint8))

def jsc(n, k, spec):
    book=(codebook_for(spec).astype(np.int16)*8).astype(np.int8)
    return NvqJscTensor(shape=(n,k),axis=0,neuron_len=k,
        neuron_scale=random.uniform(.002,.02,n).astype(np.float16).astype(np.float32),
        scale_lut=np.linspace(.5,1.5,16,dtype=np.float16).astype(np.float32),
        bank_for_state=np.arange(16,dtype=np.uint8)&1,
        state=random.integers(0,16,(n,math.ceil(k/24)),dtype=np.uint8),
        indices=random.integers(0,spec.codebook_entries,(n,math.ceil(k/spec.vector_size)),dtype=np.uint16),
        signs=random.integers(0,128,(n,math.ceil(k/8)),dtype=np.uint8),
        codebooks=np.stack((book,book),axis=0),base_spec=spec)

def decoded(tensor):
    if isinstance(tensor,NintTensor):
        scale=tensor.neuron_scale[:,None]*tensor.sub_scale.astype(np.float32)
        minimum=tensor.neuron_min[:,None]*tensor.sub_min.astype(np.float32)
        dense=scale[:,:,None]*tensor.q.astype(np.float32)-minimum[:,:,None]
        return dense.reshape(tensor.q.shape[0],-1)[:,:tensor.neuron_len]
    if isinstance(tensor,Nvq1LTensor): return decode_l(tensor)
    if isinstance(tensor,Nvq1STensor): return decode_s(tensor)
    return dequantize_nvq_jsc(tensor)

profiles=[('nint4',lambda n,k:nint(n,k,4,24,6)),
    ('nint5',lambda n,k:nint(n,k,5,28,7)),
    ('nint6',lambda n,k:nint(n,k,6,32,8)),
    ('nint8',lambda n,k:nint(n,k,8,32,8)),
    ('nvq1-s',lambda n,k:onebit(n,k,True)),
    ('nvq1-l',lambda n,k:onebit(n,k,False))]
for label,spec in [('nvq2j',NVQ2_E8),('nvq2j-l',NVQ2_E8_1024),('nvq2j-xl',NVQ2_E8_4096),
    ('nvq3j',NVQ3_D4),('nvq3j-512',NVQ3_D4_512),('nvq3j-l',NVQ3_D4_1024)]:
    profiles.append((label,lambda n,k,spec=spec:jsc(n,k,spec)))
cases=[]
for label,create in profiles:
    for n,k in [(640,2560),(2560,640)]:
        name=f'{label}-{n}-{k}'
        tensor=create(n,k)
        dtype,blob=io.pack_tensor_payload(tensor)
        frozen=io.unpack_tensor_payload(dtype,blob)
        io.save(ROOT/(name+'.mfq'),FileHeader(version=2,model_arch='cpu-quant-oracle'),{'linear.weight':frozen})
        input=random.uniform(-.15,.15,(3,k)).astype(np.float32)
        input[1]=0
        input[2,k//2]=.00003
        reference=(input.astype(np.float64)@decoded(frozen).astype(np.float64).T).astype(np.float32)
        input.tofile(ROOT/(name+'.input.f32'))
        reference.tofile(ROOT/(name+'.expected.f32'))
        cases.append(f'{name} {n} {k} 3')
        print(name,flush=True)
(ROOT/'cases.txt').write_text('\n'.join(cases)+'\n',encoding='utf-8')
