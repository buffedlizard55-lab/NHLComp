
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
