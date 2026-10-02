import io
import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from mfq.calibration.alphaq_output import AlphaQOutput, model_files
from mfq.formats.assets import ASSET_DTYPE, ASSET_PREFIX, RuntimeAsset
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap
from mfq.formats.shards import _write_header_and_table
from mfq.formats.nint import NintSpec
from mfq.tools import quantize_hf_to_mfq as q


@pytest.mark.parametrize('staged', [False, True])
def test_normal_writer_enforces_frozen_metadata_and_complete_file_cap(tmp_path, staged):
    root=tmp_path/'model';root.mkdir()
    save_file({'model.language_model.layers.0.mlp.down_proj.weight':torch.ones((4,640),dtype=torch.bfloat16)},root/'model.safetensors')
    (root/'config.json').write_text(json.dumps({'model_type':'qwen3_5'}))
    output=tmp_path/'model.mfq'
    args=q.build_parser().parse_args(['--input',str(root),'--output',str(output),
        '--quant-backend','cpu','--device','cpu','--row-chunk','4'])
    args.staged_blobs=staged
    plans=tuple(q._normalize_hf_expert_storage(q._plan(root,False,None,args.dense_dtype.upper())))
    asset=RuntimeAsset(ASSET_PREFIX+'proof.bin','application/octet-stream',b'proof')
    header=FileHeader(version=2,model_arch='test-alphaq',extra={'contract':'frozen before allocation'})
    records=[q.BlobRecord(p.name,'NINT',q._plan_blob_nbytes(p,NintSpec()),tmp_path) for p in plans]
    records.append(q.BlobRecord(asset.name,ASSET_DTYPE,len(asset.data),tmp_path))
    buffer=io.BytesIO();_write_header_and_table(buffer,header,records)
    size=buffer.tell()+sum(r.nbytes for r in records)
    contract=AlphaQOutput(root,plans,model_files(root),header,(asset,),size)
    args._alphaq_output=replace(contract,maximum_file_bytes=size-1)
    with pytest.raises(ValueError,match='exceeds file budget'):
        q.convert(args)
    assert not output.exists()
    args._alphaq_output=contract
    q.convert(args)
    assert output.stat().st_size==size
    store=open_mmap(output)
    try:
        assert store.header.extra==header.extra
        assert store.read_blob(asset.name)==asset.data
    finally:
        store.close()
    with pytest.raises(ValueError,match='plan differs'):
        contract.validate_plan([replace(plans[0],target_dtype='BF16')])
    (root/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='source changed'):
        contract.validate_plan(plans)
