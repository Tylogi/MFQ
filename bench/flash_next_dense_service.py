"""Verify the current dense-operator changes through the native stdio service."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'workspace/local-server'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile',action='store_true')
parser.add_argument('--mtp-only',action='store_true')
parser.add_argument('--ordinary-profile',action='store_true')
parser.add_argument('--tag')
args=parser.parse_args()
if args.ordinary_profile:
    args.profile=True
tag=args.tag or ('dense-service-profile' if args.profile else 'dense-service')
cases=[{'requests':1,'tokens':128,'tag':tag+'-cold128'},
    {'requests':1,'tokens':1024,'tag':tag+'-ordinary1024'},
    {'requests':1,'tokens':1024,'mtp':True,'tag':tag+'-mtp1024'},
    {'requests':2,'tokens':512,'tag':tag+'-b2-512'},
    {'requests':1,'tokens':1024,'tag':tag+'-ordinary-repeat1024'}]
if args.mtp_only:
    cases=[{'requests':1,'tokens':128,'tag':tag+'-cold128'},
        {'requests':1,'tokens':512,'tag':tag+'-ordinary512'},
        {'requests':1,'tokens':1024,'mtp':True,'tag':tag+'-mtp1024'},
        {'requests':1,'tokens':1024,'mtp':True,'tag':tag+'-mtp-repeat1024'},
        {'requests':2,'tokens':512,'tag':tag+'-b2-512'},
        {'requests':1,'tokens':512,'tag':tag+'-ordinary-repeat512'}]
if args.ordinary_profile:
    cases=[{'requests':1,'tokens':1024,'tag':tag+'-warm1024'},
        {'requests':1,'tokens':1024,'tag':tag+'-ordinary-repeat1024'}]
if not args.profile and not args.mtp_only:
    prompt='请用中文写一篇2000字以上的技术分析，解释操作系统中的虚拟内存、分页、缺页异常、页缓存、内存映射和多进程的关系，包含具体例子、实现细节及性能影响。不要省略分析。'
    cases.extend([{'requests':1,'tokens':256,'prompt':prompt,'tag':tag+'-general256'},
        {'requests':1,'tokens':256,'prompt':prompt,'mtp':True,
         'reference_tag':tag+'-general256','tag':tag+'-general-mtp256'}])
env={**os.environ,'MFQ_BENCH_CONCURRENCY':'2','MFQ_BENCH_ALLOW_MTP_OUTPUT_DIFFERENCE':'1',
    'MFQ_TRACE_LAYER_TIMINGS':'1' if args.profile else '0',
    'MFQ_TRACE_MHC_TIMINGS':'1' if args.profile else '0','MFQ_TRACE_LAYER_TIMINGS_CLOCK':'1',
    'MFQ_NINT_GROUP_LOAD':'0','MFQ_NINT_Q8_DECODE':'2','MFQ_TENSOR_VIEW_RESHAPE':'1',
    'MFQ_NINT_SHORT_WARPS':'1','MFQ_GR_NORM_CACHE':'1',
    'MFQ_NINT_PACKED_STAGE':'0','MFQ_GR_GROUP_READ':'0','MFQ_GR_UP_METADATA':'0',
    'PYTHONPATH':str(ROOT)+os.pathsep+str(ROOT/'workspace/python-deps'),
    'TEMP':str(ROOT/'workspace/tmp'),'TMP':str(ROOT/'workspace/tmp')}
commands=''.join(json.dumps(case,ensure_ascii=False)+'\n' for case in cases)+json.dumps({'stop':True})+'\n'
(OUT/(tag+'-cases.jsonl')).write_text(commands,encoding='utf-8')
started=time.perf_counter()
with (OUT/(tag+'-console.log')).open('w',encoding='utf-8') as log:
    result=subprocess.run([sys.executable,'bench/flash_next_request_throughput.py'],input=commands,
        encoding='utf-8',env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=360)
for name in ['worker.log','worker-config.json']:
    shutil.copyfile(OUT/'continuous-throughput'/name,OUT/(tag+'-'+name))
reports=[]
for case in cases:
    path=OUT/'continuous-throughput'/(case['tag']+'.json')
    if not path.exists():
        reports.append({'case':case,'errors':['case did not run; inspect console log']});continue
    data=json.loads(path.read_text(encoding='utf-8'))
    outputs=data.get('outputs',[])
    errors=data.get('errors',[data.get('error')]) if 'errors' not in data else list(data['errors'])
    if case.get('mtp') and not all(out.get('generation_metrics',{}).get('mtp_used') and
        out.get('generation_metrics',{}).get('mtp_cycles',0)>0 for out in outputs):
        errors.append('requested MTP was not executed')
    reports.append({'case':case,'errors':errors,
        'decode_tps':(data.get('tokens',0)-len(outputs))/max((out.get('decode_s') or 0 for out in outputs),default=1),
        'output_differences':data.get('output_differences',[]),
        'generation_metrics':[out.get('generation_metrics',{}) for out in outputs]})
report={'process_exit_code':result.returncode,'elapsed_s':time.perf_counter()-started,'reports':reports}
(OUT/(tag+'-results.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
raise SystemExit(result.returncode or int(any(row['errors'] for row in reports)))
