"""Read the generic device body after MFE host/device source factoring."""
import re

def read_device_body(root):
    source=root/'mfq/kernels/cuda/mfe_ffn.cu'
    text=source.read_text()
    if '#include "mfe_ffn_device.cuh"' not in text:
        return text.split('bool mfe_group_dot_enabled() {')[0]
    text=(root/'mfq/kernels/cuda/mfe_ffn_device.cuh').read_text()
    start=text.index('#include "mfe_ffn.h"')
    text=text[start:].removesuffix('\n} // anonymous namespace\n} // namespace mfq::cuda\n')
    text,count=re.subn(r'#if MFQ_MFE_COMPACT_NVQ\n.*?#else\n(.*?)#endif\n',r'\1',text,flags=re.S)
    assert count==2
    return text.replace('if(MFQ_MFE_COMPACT_NVQ || d.','if(d.')
