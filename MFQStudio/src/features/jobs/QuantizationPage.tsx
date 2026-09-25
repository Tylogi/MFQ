/** 量化路由入口：在页面范围组织任务状态及职责独立的业务面板。 */
import { PanelDeck } from '../../app/PanelDeck';
import { QuantizationContext } from './QuantizationContext';
import { useQuantizationWorkspace } from './useQuantizationWorkspace';
import { ImatrixPanel } from './ImatrixPanel';
import { LineagePanel } from './LineagePanel';
import { JobBuilder } from './JobBuilder';
import { JobHistory } from './JobHistory';
import { JobDetail } from './JobDetail';
/** 挂载量化工作台，离开页面时释放页面自己的日志订阅。 */
export function QuantizationPage() {
  const workspace = useQuantizationWorkspace();
  return (
    <QuantizationContext.Provider value={workspace}>
      <PanelDeck labels={workspace.panelLabels} page="lab-quantization">
        <ImatrixPanel key="imatrix" />
        <LineagePanel key="lineage" />
        <JobBuilder key="builder" />
        <JobHistory key="history" />
        {workspace.selectedJob && <JobDetail key="detail" />}
      </PanelDeck>
    </QuantizationContext.Provider>
  );
}
