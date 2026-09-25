/** 展示量化工作台的动态任务参数表单。 */
import { useQuantization } from './QuantizationContext';
import { Icon } from '../../app/display';
import { schemaType } from './jobSchema';
/** 从页面状态读取本面板所需数据与业务操作。 */
export function JobBuilder() {
  const {
    tr,
    busy,
    selectedJobKind,
    setSelectedJobKind,
    jobPayload,
    genericJobKinds,
    selectedKind,
    imatrixArtifacts,
    updateJobPayload,
    submitJob,
  } = useQuantization();
  return (
    <>
      <form className="dashboard-panel job-builder" key="builder" onSubmit={submitJob}>
        <div className="panel-heading">
          <div>
            <h2>{tr('新任务', 'New job')}</h2>
          </div>
        </div>
        <label>
          <span>{tr('任务类型', 'Job type')}</span>
          <select
            onChange={(event) => setSelectedJobKind(event.target.value)}
            value={selectedJobKind}
          >
            {genericJobKinds.map((item) => (
              <option key={item.kind} value={item.kind}>
                {item.kind}
              </option>
            ))}
          </select>
        </label>
        {Object.entries(selectedKind?.payload_schema.properties ?? {}).map(([name, property]) => {
          if (
            selectedJobKind === 'model.quantize' &&
            name.startsWith('imatrix_') &&
            name !== 'imatrix' &&
            !Boolean(jobPayload.calibrate_imatrix)
          )
            return null;
          if (
            selectedJobKind === 'model.quantize' &&
            name === 'imatrix' &&
            Boolean(jobPayload.calibrate_imatrix)
          )
            return null;
          const required = selectedKind?.payload_schema.required?.includes(name);
          const type = schemaType(property);
          const value = jobPayload[name];
          const isImatrix = selectedJobKind === 'model.quantize' && name === 'imatrix';
          return (
            <label key={name}>
              <span>
                {property.title || name}
                {required ? ' *' : ''}
              </span>
              {isImatrix && imatrixArtifacts.length > 0 ? (
                <select
                  onChange={(event) => updateJobPayload(name, property, event.target.value)}
                  value={String(value ?? '')}
                >
                  <option value="">{tr('不使用', 'None')}</option>
                  {imatrixArtifacts.map((item) => (
                    <option key={item.id} value={item.artifact_uri.replace(/^workspace:\/\//, '')}>
                      {item.artifact_name}
                    </option>
                  ))}
                </select>
              ) : property.enum ? (
                <select
                  onChange={(event) => updateJobPayload(name, property, event.target.value)}
                  required={required}
                  value={String(value ?? '')}
                >
                  {!required && <option value="" />}
                  {property.enum.map((item) => (
                    <option key={String(item)} value={String(item)}>
                      {String(item)}
                    </option>
                  ))}
                </select>
              ) : type === 'boolean' ? (
                <input
                  checked={Boolean(value)}
                  onChange={(event) => updateJobPayload(name, property, event.target.checked)}
                  type="checkbox"
                />
              ) : (
                <input
                  max={property.maximum}
                  min={property.minimum}
                  onChange={(event) => updateJobPayload(name, property, event.target.value)}
                  required={required}
                  type={type === 'integer' || type === 'number' ? 'number' : 'text'}
                  value={Array.isArray(value) ? value.join(', ') : String(value ?? '')}
                />
              )}
              {property.description && <small>{property.description}</small>}
            </label>
          );
        })}
        <div className="job-submit">
          <button className="job-run" disabled={busy || !selectedJobKind} type="submit">
            <Icon name="play" size={12} />
            {tr('运行', 'Run')}
          </button>
        </div>
      </form>
    </>
  );
}
