import React, { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, Bell, ChevronDown,
  CircleGauge, Cpu, Download, Factory, Gauge, LayoutDashboard, LifeBuoy,
  Search, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Thermometer,
  Wind, Zap, Menu, X, Clock3, ArrowRight, CircleHelp, RefreshCw
} from "lucide-react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, ComposedChart,
  Line, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis
} from "recharts";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

const demoEngines = [
  {unit_id:1,predicted_rul:201.4,lower_rul:97.7,upper_rul:305.1,interval_width:207.4,average_sensor_health:81.9,unhealthy_sensor_count:9,maintenance_priority:"High",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Prioritise inspection and verify sensor readings"},
  {unit_id:2,predicted_rul:151.0,lower_rul:47.3,upper_rul:254.8,interval_width:207.4,average_sensor_health:82.3,unhealthy_sensor_count:8,maintenance_priority:"Medium",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Schedule inspection and monitor RUL trend"},
  {unit_id:3,predicted_rul:49.5,lower_rul:0,upper_rul:153.2,interval_width:153.2,average_sensor_health:82.8,unhealthy_sensor_count:8,maintenance_priority:"High",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Prioritise inspection and verify sensor readings"},
  {unit_id:4,predicted_rul:78.6,lower_rul:0,upper_rul:182.3,interval_width:182.3,average_sensor_health:82.5,unhealthy_sensor_count:8,maintenance_priority:"Medium",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Schedule inspection and monitor RUL trend"},
  {unit_id:5,predicted_rul:119.7,lower_rul:16.0,upper_rul:223.4,interval_width:207.4,average_sensor_health:82.4,unhealthy_sensor_count:8,maintenance_priority:"Low",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Continue routine monitoring"},
  {unit_id:15,predicted_rul:192.5,lower_rul:88.7,upper_rul:296.2,interval_width:207.4,average_sensor_health:79.4,unhealthy_sensor_count:10,maintenance_priority:"Medium",sensor_health_status:"Warning",prediction_trust:"Use with caution",recommended_action:"Schedule inspection and monitor RUL trend"},
];

const fallback = {
  engines: demoEngines,
  summary: {total_engines:100, high_priority:19, medium_priority:81, low_priority:0, average_predicted_rul:85.8, average_sensor_health:82.8, mean_interval_width:174.8, test_coverage:99, mae:21.78, rmse:30.34},
  source: "DEMO PREVIEW"
};

const priorityColor = {High:"#ff657a", Medium:"#f3b64b", Low:"#45d6a0"};
const format = (n, digits=1) => Number(n ?? 0).toFixed(digits);


function normalizePayload(payload) {
  if (Array.isArray(payload)) {
    return {
      ...fallback,
      engines: payload,
      source: "LIVE API",
    };
  }

  const apiSummary = payload.summary || {};

  const summary = {
    ...fallback.summary,
    ...apiSummary,

    // Map FastAPI field names to the existing React KPI names.
    average_predicted_rul:
      apiSummary.average_rul ??
      apiSummary.average_predicted_rul ??
      fallback.summary.average_predicted_rul,

    mean_interval_width:
      apiSummary.average_interval_width ??
      apiSummary.mean_interval_width ??
      fallback.summary.mean_interval_width,
  };

  return {
    ...fallback,
    ...payload,
    engines:
      payload.engines ||
      payload.predictions ||
      payload.data ||
      fallback.engines,
    summary,
    source: "LIVE API",
  };
}


function App() {
  const [page, setPage] = useState("Overview");
  const [engines, setEngines] = useState(fallback.engines);
  const [summary, setSummary] = useState(fallback.summary);
  const [source, setSource] = useState("DEMO PREVIEW");
  const [selected, setSelected] = useState(1);
  const [query, setQuery] = useState("");
  const [priority, setPriority] = useState("All priorities");
  const [loading, setLoading] = useState(true);
  const [mobileNav, setMobileNav] = useState(false);
  const [engineDetails, setEngineDetails] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  async function loadData() {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/dashboard`, {signal: AbortSignal.timeout(2500)});
      if (!response.ok) throw new Error("API unavailable");
      const payload = normalizePayload(await response.json());
      setEngines(payload.engines);
      setSummary(payload.summary);
      setSource("LIVE API");
    } catch {
      setEngines(fallback.engines);
      setSummary(fallback.summary);
      setSource("DEMO PREVIEW");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadData(); }, []);
  const current = engines.find(e => Number(e.unit_id) === Number(selected)) || engines[0];
  useEffect(() => {
  if (page !== "Engine analysis" || !selected) return;

  const controller = new AbortController();

  async function loadEngineDetails() {
    setDetailLoading(true);
    setDetailError("");

    try {
      const response = await fetch(
        `${API_BASE}/api/engines/${selected}`,
        { signal: controller.signal }
      );

      if (!response.ok) {
        throw new Error(`Engine API returned ${response.status}`);
      }

      const data = await response.json();
      setEngineDetails(data);
    } catch (error) {
      if (error.name !== "AbortError") {
        setDetailError(error.message);
        setEngineDetails(null);
      }
    } finally {
      if (!controller.signal.aborted) {
        setDetailLoading(false);
      }
    }
  }

  loadEngineDetails();

  return () => controller.abort();
}, [page, selected]);


const filtered = useMemo(() => {
  const normalizedQuery = query
    .toLowerCase()
    .trim()
    .replace(/^(engine|unit)\s*/i, "")
    .replace(/^#/, "")
    .replace(/\s+/g, "");

  return engines.filter((engine) => {
    const matchesPriority =
      priority === "All priorities" ||
      engine.maintenance_priority?.toLowerCase() ===
        priority.toLowerCase();

    const matchesSearch =
      normalizedQuery === "" ||
      String(engine.unit_id).includes(normalizedQuery);

    return matchesPriority && matchesSearch;
  });
}, [engines, priority, query]);

  const nav = [
    {name:"Overview", icon:LayoutDashboard},
    {name:"Fleet", icon:Factory},
    {name:"Engine analysis", icon:Gauge},
  ];

  function downloadCSV() {
    const columns = Object.keys(engines[0] || {});
    const csv = [columns.join(","), ...engines.map(row => columns.map(c => `"${String(row[c] ?? "").replaceAll('"','""')}"`).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
    const a = document.createElement("a"); a.href=url; a.download="predictx-fleet-report.csv"; a.click(); URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
      <div className="brand">
        <div className="brand-mark"><Activity size={21}/></div>
        <div><div className="brand-name">PredictX<span> AI</span></div><div className="brand-caption">ASSET INTELLIGENCE</div></div>
        <button className="mobile-close" onClick={()=>setMobileNav(false)}><X size={18}/></button>
      </div>
      <div className="workspace-label">WORKSPACE</div>
      <div className="nav-list">
        {nav.map(item => <button key={item.name} onClick={()=>{setPage(item.name);setMobileNav(false)}} className={`nav-item ${page===item.name?"active":""}`}><item.icon size={17}/><span>{item.name}</span>{page===item.name && <span className="nav-indicator"/>}</button>)}
      </div>
      <div className="sidebar-bottom">
        <div className="dataset-card">
          <div className="dataset-icon"><Cpu size={17}/></div>
          <div><div className="dataset-title">NASA C-MAPSS</div><div className="dataset-sub">FD001 · Turbofan fleet</div></div>
          <span className="dataset-dot"/>
        </div>
        <div className="sidebar-foot"><div className="avatar">PX</div><div><b>PredictX Workspace</b><small>Research environment</small></div><Settings2 size={16}/></div>
      </div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <button className="mobile-menu" onClick={()=>setMobileNav(true)}><Menu/></button>
        <div className="breadcrumb">Workspace <span>/</span> <b>{page}</b></div>
        <div className="top-actions">
          <div className={`source-pill ${source==="LIVE API"?"live":""}`}><span/> {source}</div>
          <button className="icon-button" title="Refresh data" onClick={loadData}><RefreshCw size={17}/></button>
          <button className="icon-button" title="Notifications"><Bell size={17}/><i/></button>
          <div className="user-avatar">D</div>
        </div>
      </header>

      <div className="page-wrap">
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span className="eyebrow-line"/> PREDICTIVE INTELLIGENCE PLATFORM</div>
            <h1>{page==="Overview"?"Fleet command center":page==="Fleet"?"Fleet intelligence":"Engine inspection"}</h1>
            <p>Understand asset health, anticipate degradation, and make maintenance decisions with confidence.</p>
          </div>
          <div className="hero-visual">
            <img src="/images/turbofan-hero.jpg" alt="Photorealistic aerospace turbofan engine" />
            <div className="hero-visual-shade" />
            <div className="hero-visual-label"><span>ASSET CLASS</span><b>HIGH-BYPASS TURBOFAN</b></div>
          </div>
        </section>

        {page==="Overview" && <>
          <div className="section-heading"><div><span className="section-kicker">LIVE SNAPSHOT</span><h2>Operational overview</h2></div><div className="updated"><Clock3 size={14}/> {loading?"Refreshing…":"Latest report loaded"}</div></div>
          <div className="kpi-grid">
            <KPI icon={Factory} label="Monitored engines" value={summary.total_engines} note="Fleet population" accent="cyan"/>
            <KPI icon={AlertTriangle} label="High priority" value={summary.high_priority} note={`${format(summary.high_priority/Math.max(summary.total_engines,1)*100,0)}% of fleet`} accent="red"/>
            <KPI icon={Activity} label="Average RUL" value={`${format(summary.average_predicted_rul)} `} suffix="cycles" note="Predicted remaining life" accent="blue"/>
            <KPI icon={ShieldCheck} label="Sensor health" value={format(summary.average_sensor_health)} suffix="/ 100" note="Rule-based health score" accent="green"/>
            <KPI icon={CircleGauge} label="Mean interval width" value={format(summary.mean_interval_width)} suffix="cycles" note="Prediction uncertainty" accent="violet"/>
          </div>

          <div className="section-heading section-gap"><div><span className="section-kicker">FLEET RISK</span><h2>Reliability at a glance</h2></div><span className="subtle-tag">ENGINE-LEVEL VIEW</span></div>
          <div className="chart-grid two">
            <Panel title="Maintenance priority" subtitle="Distribution across monitored engines" tag="RISK PROFILE">
              <div className="priority-layout">
                <div className="donut-wrap"><ResponsiveContainer width="100%" height={210}><PieChart><Pie data={[{name:"High",value:summary.high_priority},{name:"Medium",value:summary.medium_priority},{name:"Low",value:summary.low_priority}]} dataKey="value" nameKey="name" innerRadius={63} outerRadius={83} paddingAngle={4} stroke="none">{["High","Medium","Low"].map(k=><Cell key={k} fill={priorityColor[k]}/>)}</Pie><Tooltip contentStyle={tooltipStyle}/></PieChart></ResponsiveContainer><div className="donut-center"><b>{summary.total_engines}</b><span>ENGINES</span></div></div>
                <div className="legend-list">{["High","Medium","Low"].map(k=><div className="legend-row" key={k}><span className="legend-dot" style={{background:priorityColor[k]}}/><span>{k} priority</span><b>{summary[`${k.toLowerCase()}_priority`]}</b></div>)}</div>
              </div>
            </Panel>
            <Panel title="RUL distribution" subtitle="Predicted remaining useful life in cycles" tag="MODEL OUTPUT">
              <ResponsiveContainer width="100%" height={230}><BarChart data={histogram(engines)}><CartesianGrid stroke="#223247" vertical={false}/><XAxis dataKey="range" tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={tooltipStyle}/><Bar dataKey="count" fill="#3fcbe8" radius={[5,5,0,0]} maxBarSize={27}/></BarChart></ResponsiveContainer>
            </Panel>
          </div>

          <div className="chart-grid two section-gap">
            <Panel title="RUL vs uncertainty" subtitle="Identify engines with wide prediction intervals" tag="CONFIDENCE MAP">
              <ResponsiveContainer width="100%" height={245}><ComposedChart data={engines}><CartesianGrid stroke="#223247" vertical={false}/><XAxis dataKey="predicted_rul" type="number" tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false} label={{value:"Predicted RUL (cycles)",position:"insideBottom",offset:-7,fill:"#8194aa",fontSize:10}}/><YAxis dataKey="interval_width" tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={tooltipStyle}/><Line dataKey="interval_width" stroke="#9b86ff" strokeWidth={0} dot={{r:4,fill:"#9b86ff",stroke:"#171b2a",strokeWidth:1}}/></ComposedChart></ResponsiveContainer>
            </Panel>
            <Panel title="Fleet health signal" subtitle="Average sensor health by engine" tag="SENSOR QUALITY">
              <ResponsiveContainer width="100%" height={245}><AreaChart data={[...engines].sort((a,b)=>a.unit_id-b.unit_id)}><defs><linearGradient id="healthFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3ed6a0" stopOpacity={0.3}/><stop offset="100%" stopColor="#3ed6a0" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#223247" vertical={false}/><XAxis dataKey="unit_id" tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false}/><YAxis domain={[0,100]} tick={{fill:"#8194aa",fontSize:10}} axisLine={false} tickLine={false}/><Tooltip contentStyle={tooltipStyle}/><Area dataKey="average_sensor_health" stroke="#3ed6a0" fill="url(#healthFill)" strokeWidth={2}/></AreaChart></ResponsiveContainer>
            </Panel>
          </div>

          <div className="section-heading section-gap"><div><span className="section-kicker">ACTION QUEUE</span><h2>Priority attention</h2></div><button className="text-button" onClick={()=>setPage("Fleet")}>View fleet <ArrowRight size={15}/></button></div>
          <EngineTable rows={[...engines].sort((a,b)=>a.predicted_rul-b.predicted_rul).slice(0,5)} onSelect={id=>{setSelected(id);setPage("Engine analysis")}}/>
        </>}

        {page==="Fleet" && <>
          <div className="section-heading"><div><span className="section-kicker">ASSET REGISTER</span><h2>Fleet intelligence</h2></div><button className="secondary-button" onClick={downloadCSV}><Download size={15}/> Export CSV</button></div>
          <div className="filter-bar"><div className="search-box"><Search size={16}/><input placeholder="Search engine ID…" value={query} onChange={e=>setQuery(e.target.value)}/></div><div className="select-box"><SlidersHorizontal size={15}/><select value={priority} onChange={e=>setPriority(e.target.value)}><option>All priorities</option><option>High</option><option>Medium</option><option>Low</option></select><ChevronDown size={14}/></div><span className="result-count">{filtered.length} engines</span></div>
          <EngineTable rows={filtered} onSelect={id=>{setSelected(id);setPage("Engine analysis")}}/>
        </>}

        {page==="Engine analysis" && current && <>
          <div className="section-heading"><div><span className="section-kicker">ASSET DEEP DIVE</span><h2>Engine {current.unit_id} inspection</h2></div><div className="select-box"><Factory size={15}/><select value={selected} onChange={e=>setSelected(Number(e.target.value))}>{engines.map(e=><option key={e.unit_id} value={e.unit_id}>Engine {e.unit_id}</option>)}</select><ChevronDown size={14}/></div></div>
          <div className="engine-hero">
            <div className="engine-digital-twin"><img src="/images/turbofan-digital-twin.jpg" alt="3D digital-twin style cutaway of a turbofan engine"/><span>ENGINE DIGITAL TWIN · CONCEPT VISUAL</span></div>
            <div><div className="section-kicker">REMAINING USEFUL LIFE</div><div className="big-rul">{format(current.predicted_rul)} <small>cycles</small></div><p>Point estimate from the robust Random Forest model.</p></div>
            <div className="interval-box"><span>90% TARGET INTERVAL · EXPLORATORY</span><div><b>{format(current.lower_rul,0)}</b><i>to</i><b>{format(current.upper_rul,0)}</b></div><small>Interval width: {format(current.interval_width)} cycles</small></div>
            <div className="engine-health"><div className="health-ring" style={{"--health":`${current.average_sensor_health}%`}}><b>{format(current.average_sensor_health,0)}</b></div><span>SENSOR HEALTH</span></div>
          </div>
          <div className="chart-grid two section-gap">
            <Panel title="Prediction interval" subtitle="Point estimate and uncertainty bounds" tag="RUL ESTIMATE"><ResponsiveContainer width="100%" height={240}><ComposedChart data={[{name:"Lower",value:current.lower_rul},{name:"Prediction",value:current.predicted_rul},{name:"Upper",value:current.upper_rul}]}><CartesianGrid stroke="#223247" vertical={false}/><XAxis dataKey="name" tick={{fill:"#8194aa"}} axisLine={false} tickLine={false}/><YAxis tick={{fill:"#8194aa"}} axisLine={false} tickLine={false}/><Tooltip contentStyle={tooltipStyle}/><Bar dataKey="value" fill="#3fcbe8" radius={[5,5,0,0]}>{[0,1,2].map((v)=><Cell key={v} fill={v===1?"#9b86ff":"#315a75"}/>)}</Bar></ComposedChart></ResponsiveContainer></Panel>
            <Panel title="Decision assessment" subtitle="Transparent prototype rules" tag="DECISION SUPPORT">
              <div className="decision-list"><Decision label="Maintenance priority" value={current.maintenance_priority} color={priorityColor[current.maintenance_priority]}/><Decision label="Sensor status" value={current.sensor_health_status} color={current.sensor_health_status==="Healthy"?"#45d6a0":"#f3b64b"}/><Decision label="Prediction trust" value={current.prediction_trust} color="#9b86ff"/></div>
              <div className="recommendation"><div><Sparkles size={16}/> Recommended next step</div><p>{current.recommended_action}</p></div>
            </Panel>
          </div>  
<Panel
  title="Sensor intelligence"
  subtitle="Individual sensor reliability diagnostics"
  tag="21 SENSOR CHANNELS"
>
  {detailLoading ? (
    <p className="empty-state">Loading sensor diagnostics...</p>
  ) : detailError ? (
    <p className="empty-state">
      Unable to load sensor data: {detailError}
    </p>
  ) : (
    <>
      <div className="section-heading">
        <div>
          <span className="section-kicker">
            ENGINE {selected}
          </span>
          <h2>
            {engineDetails?.sensors?.length || 0} sensor records
          </h2>
        </div>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>SENSOR</th>
              <th>HEALTH SCORE</th>
              <th>STATUS</th>
              <th>MISSING RATE</th>
              <th>STUCK RATE</th>
              <th>OUTLIER RATE</th>
              <th>NOISE RATIO</th>
            </tr>
          </thead>

          <tbody>
            {(engineDetails?.sensors || []).map((sensor) => (
              <tr key={`${sensor.unit_id}-${sensor.sensor}`}>
                <td><b>{sensor.sensor}</b></td>

                <td>
                  <b>{format(sensor.health_score, 1)}</b>
                </td>

                <td>
                  <span
                    className={`priority-badge ${
                      sensor.status === "Healthy"
                        ? "low"
                        : sensor.status === "Warning"
                        ? "medium"
                        : "high"
                    }`}
                  >
                    {sensor.status}
                  </span>
                </td>

                <td>{format(sensor.missing_rate, 2)}%</td>
                <td>{format(sensor.stuck_rate, 2)}%</td>
                <td>{format(sensor.outlier_rate, 2)}%</td>
                <td>{format(sensor.noise_ratio, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )}
</Panel>

          <div className="notice"><ShieldCheck size={17}/><span>Uncertainty intervals are exploratory. The current calibration set has prior model-selection exposure; do not interpret this as a formal independent coverage guarantee.</span></div>
        </>}
        <footer className="footer"><span>© 2026 PredictX AI · Reliability-aware asset intelligence</span><span>NASA C-MAPSS FD001 <i/> Research prototype</span></footer>
      </div>
    </main>
  </div>;
}

function KPI({icon:Icon,label,value,suffix,note,accent}) {
  return <div className={`kpi-card ${accent}`}><div className="kpi-top"><span>{label}</span><div className="kpi-icon"><Icon size={17}/></div></div><div className="kpi-value">{value}<small>{suffix}</small></div><div className="kpi-note">{note}</div></div>;
}

function Panel({title,subtitle,tag,children}) {
  return <section className="panel"><div className="panel-head"><div><h3>{title}</h3><p>{subtitle}</p></div><span className="panel-tag">{tag}</span></div>{children}</section>;
}

function Decision({label,value,color}) {
  return <div className="decision-row"><span>{label}</span><b style={{color}}>{value}</b></div>;
}

function EngineTable({rows,onSelect}) {
  return <div className="table-wrap"><table><thead><tr><th>ENGINE</th><th>EST. RUL</th><th>SENSOR HEALTH</th><th>INTERVAL WIDTH</th><th>PRIORITY</th><th>INSPECT</th></tr></thead><tbody>{rows.map(e=><tr key={e.unit_id}><td><div className="engine-id"><span className="engine-icon"><Cpu size={15}/></span><b>Engine {e.unit_id}</b></div></td><td><b>{format(e.predicted_rul)} <small>cycles</small></b></td><td><div className="health-cell"><span>{format(e.average_sensor_health,0)}%</span><div><i style={{width:`${e.average_sensor_health}%`}}/></div></div></td><td>{format(e.interval_width)} cycles</td><td><span className={`priority-badge ${e.maintenance_priority.toLowerCase()}`}>{e.maintenance_priority}</span></td><td><button className="inspect-button" onClick={()=>onSelect(e.unit_id)}>Inspect <ArrowUpRight size={13}/></button></td></tr>)}</tbody></table>{rows.length===0 && <div className="empty-state">No engines match these filters.</div>}</div>;
}

function histogram(data) {
  const bins = Array.from({length:8},(_,i)=>({range:`${i*40}-${(i+1)*40}`,count:0}));
  data.forEach(e=>{const idx=Math.min(7,Math.max(0,Math.floor(Number(e.predicted_rul)/40)));bins[idx].count++});
  return bins;
}

const tooltipStyle = {background:"#111d2d",border:"1px solid #2b4058",borderRadius:10,color:"#edf4fc",fontSize:12};
export default App;



