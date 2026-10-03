import json
from pathlib import Path

import pytest

from mfq.cli import main
from mfq.calibration.alphaq_cache import source_identity
from mfq.calibration.alphaq_output import file_sha256
from mfq.calibration.alphaq_payloads import EncodedRecord,plan_identity,precision_identity
from mfq.calibration.alphaq_workflow import planned
from mfq.calibration.artifact import load_scheme
from mfq.formats.header import FileHeader
from mfq.formats.io import open_mmap,unpack_mfe,pack_tensor_payload
from mfq.formats.shards import write_blob_record_shards
from mfq.tools import quantize_hf_to_mfq as q
from tests.test_alphaq_workflow import fixture


def file_record(path):
    return dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                sha256=file_sha256(path))


def test_normal_command_reuses_encoded_bytes_without_any_weight_fitting(tmp_path,monkeypatch):
    root,imat,_=fixture(tmp_path)
    first=tmp_path/'first.mfq';cache=tmp_path/'cache'
    argv=['quantize',str(root),str(first),'--target-bpw','16','--imatrix',str(imat),
          '--alphaq-cache',str(cache),'--alphaq-profiles','NINT8','--backend','cpu']
    assert main(argv)==0
    folder=next((cache/'allocations').iterdir());scheme=load_scheme(folder/'scheme.json')
    overrides=json.loads((folder/'native-overrides.json').read_text())
    plans=planned(root,None,scheme,overrides,'recipe')
    doc=dict(format='mfq.alphaq-encoded-candidates.v1',source=source_identity(root),
             imatrix_sha256=file_sha256(imat),experts={},fixed={})
    with open_mmap(first) as store:
        for item in plans:
            if item.target_dtype=='MFE':
                tensor=unpack_mfe(store.read_blob(item.name));assert len(tensor.pools)==1
                dtype,blob=pack_tensor_payload(tensor.pools[0].tensor)
                path=cache/(item.name+'.bin');path.write_bytes(blob)
                doc['experts'][item.name]=dict(shape=item.shape,profiles={'NINT8':dict(
                    precision=precision_identity(item.expert_precisions[0]),
                    parts=[dict(**file_record(path),dtype=dtype,first=0,stop=item.shape[0])])})
            else:
                record=store.records[item.name]
                doc['fixed'][item.name]=[dict(identity=plan_identity(item),file=file_record(first),
                    offset=record.offset,nbytes=record.nbytes,dtype=record.dtype)]
    manifest=cache/'encoded-candidates.json';manifest.write_text(json.dumps(doc))
    def forbidden(*a,**kw):raise AssertionError('cached output attempted weight fitting')
    monkeypatch.setattr(q,'_write_mixed_moe_axis0_blob',forbidden)
    monkeypatch.setattr(q,'_write_nint_axis0_blob',forbidden)
    second=tmp_path/'second.mfq';argv[2]=str(second)
    assert main(argv)==0
    with open_mmap(first) as old,open_mmap(second) as new:
        assert set(old.records)==set(new.records)
        for name in old.records:
            assert old.read_blob(name)==new.read_blob(name)
        assert old.header.extra==new.header.extra
    assert first.exists() and all(Path(p['parts'][0]['path']).exists()
        for t in doc['experts'].values() for p in t['profiles'].values())
    # Weighted and unweighted NINT8 may never share an encoded cache entry.
    target=next(iter(doc['experts'].values()))['profiles']['NINT8']
    target['precision']['options']['imatrix_weighted']=False
    manifest.write_text(json.dumps(doc));argv[2]=str(tmp_path/'wrong.mfq')
    with pytest.raises(ValueError,match='precision/tables differ'):
        main(argv)
    assert not (tmp_path/'wrong.mfq').exists()


def test_generated_record_size_and_final_validation_are_checked_before_publish(tmp_path):
    path=tmp_path/'bad.mfq'
    with pytest.raises(ValueError,match='blob size mismatch'):
        write_blob_record_shards(path,FileHeader(),[EncodedRecord('a','BF16',3,lambda f:f.write(b'ab'))])
    assert not path.exists()
    def invalid(paths):raise ValueError('source changed')
    with pytest.raises(ValueError,match='source changed'):
        write_blob_record_shards(path,FileHeader(),[EncodedRecord('a','BF16',2,lambda f:f.write(b'ab'))],
                                 before_publish=invalid)
    assert not path.exists()
