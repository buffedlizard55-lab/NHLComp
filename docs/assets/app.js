
function sortTable(id, col, type){
  const t=document.getElementById(id); if(!t) return;
  const tb=t.tBodies[0]; const rows=[...tb.rows];
  const dir=t.dataset.sortDir==='asc'?'desc':'asc';
  t.dataset.sortDir=dir;
  rows.sort((a,b)=>{
    let x=a.cells[col].dataset.v ?? a.cells[col].textContent.trim();
    let y=b.cells[col].dataset.v ?? b.cells[col].textContent.trim();
    if(type==='n'){x=parseFloat(x)||0;y=parseFloat(y)||0;return dir==='asc'?x-y:y-x;}
    return dir==='asc'?String(x).localeCompare(y):String(y).localeCompare(x);
  });
  rows.forEach(r=>tb.appendChild(r));
}
function filterTable(id, q, cols){
  const t=document.getElementById(id); if(!t) return;
  q=(q||'').toLowerCase();
  for(const r of t.tBodies[0].rows){
    let hay='';
    (cols||[...Array(r.cells.length).keys()]).forEach(i=>{hay+=' '+(r.cells[i].textContent||'');});
    r.style.display = hay.toLowerCase().includes(q)?'':'none';
  }
}
// Multi-field filtering: strategy, market, team, player, goalie, season, month, metrics
const activeFilters = {};
function applyFilters(id){
  const el=document.getElementById(id);
  if(!el) return;
  const f = activeFilters[id] || {};
  if(el.tagName==='TABLE'){
    for(const r of el.tBodies[0].rows){
      let show = true;
      if(f.q){
        let hay='';
        for(let i=0;i<r.cells.length;i++) hay+=' '+(r.cells[i].textContent||'');
        if(!hay.toLowerCase().includes(f.q.toLowerCase())) show=false;
      }
      for(const k of ['strategy','market','team','player','goalie','season','month','category','status','mode']){
        if(f[k] && f[k]!==''){
          const v = (r.dataset[k]||'').toLowerCase();
          if(!v.includes(f[k].toLowerCase())) show=false;
        }
      }
      if(f.min_roi!==undefined && f.min_roi!==''){
        const roi = parseFloat(r.dataset.roi||'');
        if(!isNaN(roi) && roi < parseFloat(f.min_roi)) show=false;
      }
      if(f.min_pnl!==undefined && f.min_pnl!==''){
        const pnl = parseFloat(r.dataset.pnl||'');
        if(!isNaN(pnl) && pnl < parseFloat(f.min_pnl)) show=false;
      }
      if(f.min_edge!==undefined && f.min_edge!==''){
        const ed = parseFloat(r.dataset.edge||'');
        if(!isNaN(ed) && ed < parseFloat(f.min_edge)) show=false;
      }
      r.style.display = show?'':'none';
    }
    return;
  }
  for(const d of el.querySelectorAll('details')){
    let show = true;
    if(f.q){
      const hay = (d.textContent||'').toLowerCase();
      if(!hay.includes(f.q.toLowerCase())) show=false;
    }
    for(const k of ['strategy','market','team','player','goalie','season','month','category','status','mode']){
      if(f[k] && f[k]!==''){
        const v = (d.dataset[k]||'').toLowerCase();
        const hay2 = (d.textContent||'').toLowerCase();
        if(!v.includes(f[k].toLowerCase()) && !hay2.includes(f[k].toLowerCase())) show=false;
      }
    }
    d.style.display = show?'':'none';
  }
}
function setFilter(id, key, val){
  if(!activeFilters[id]) activeFilters[id]={};
  activeFilters[id][key]=val;
  applyFilters(id);
}
function setTextFilter(id, val){
  if(!activeFilters[id]) activeFilters[id]={};
  activeFilters[id].q=val;
  applyFilters(id);
}
function clearFilters(id){
  activeFilters[id]={};
  const c=document.getElementById(id+'_controls');
  if(c){
    for(const el of c.querySelectorAll('input,select')) el.value='';
  }
  applyFilters(id);
}
function filterCategory(id, v){
  setFilter(id,'category',v);
}
