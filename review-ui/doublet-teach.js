const $=id=>document.getElementById(id);
const state={mode:"likely_doublet",items:[],focused:0,busy:false};
const labelNames={
  single:"单细胞",touching_doublet:"相连两个",
  cluster_3plus:"3个以上",not_cell:"非细胞",skip:"跳过"
};

async function api(url,options){
  const response=await fetch(url,options);
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}
function setMessage(text,error=false){
  $("message").textContent=text;
  $("message").style.color=error?"#a43c32":"#176c68";
}
async function refreshStats(){
  const stats=await api("/api/multiplicity-stats");
  $("totalCount").textContent=stats.total;
  $("singleCount").textContent=stats.counts.single||0;
  $("doubletCount").textContent=stats.counts.touching_doublet||0;
  $("clusterCount").textContent=stats.counts.cluster_3plus||0;
  $("notCellCount").textContent=stats.counts.not_cell||0;
  const remaining=Math.max(0,(stats.recommended_minimums.touching_doublet||20)-(stats.counts.touching_doublet||0));
  $("targetText").textContent=remaining?`还建议确认 ${remaining} 个相连双细胞`:"相连双细胞基础数量已达到";
}
function focusTile(index){
  if(!state.items.length)return;
  state.focused=Math.max(0,Math.min(index,state.items.length-1));
  document.querySelectorAll(".tile").forEach((tile,tileIndex)=>
    tile.classList.toggle("focused",tileIndex===state.focused)
  );
  document.querySelector(`.tile[data-index="${state.focused}"]`)?.focus({preventScroll:true});
}
function render(){
  const grid=$("grid");grid.innerHTML="";
  $("empty").hidden=state.items.length>0;
  for(const [index,item] of state.items.entries()){
    const tile=$("tileTemplate").content.firstElementChild.cloneNode(true);
    tile.dataset.index=index;
    tile.querySelector("img").src=`/api/patch?well=${item.well}&timepoint=${item.timepoint}&x=${item.x_px}&y=${item.y_px}&size=160`;
    tile.querySelector(".index-badge").textContent=index+1;
    tile.querySelector(".probability").textContent=`细胞 ${(item.cell_probability*100).toFixed(0)}%`;
    tile.querySelector(".well").textContent=`${item.well} · ${item.timepoint}`;
    tile.querySelector(".candidate").textContent=item.candidate_id;
    tile.querySelector(".measure").innerHTML=`面积 ${Math.round(item.area_px)} px<br>直径 ${Number(item.diameter_px).toFixed(1)} px`;
    const ring=tile.querySelector(".size-ring");
    const ringSize=Math.max(8,Math.min(55,Number(item.diameter_px)/160*100*1.25));
    ring.style.width=`${ringSize}%`;ring.style.height=`${ringSize}%`;
    tile.onclick=()=>focusTile(index);
    tile.querySelectorAll("[data-label]").forEach(button=>{
      button.onclick=event=>{event.stopPropagation();labelItem(index,button.dataset.label)};
    });
    grid.appendChild(tile);
  }
  focusTile(0);
}
async function loadBatch(){
  if(state.busy)return;state.busy=true;setMessage("正在加载候选…");
  try{
    const limit=Number($("pageSize").value);
    state.items=await api(`/api/multiplicity-candidates?mode=${state.mode}&limit=${limit}`);
    state.focused=0;render();
    setMessage(`已载入 ${state.items.length} 个未标注对象`);
  }catch(error){setMessage(`加载失败：${error.message}`,true)}
  finally{state.busy=false}
}
async function labelItem(index,label){
  const item=state.items[index];if(!item||state.busy)return;
  const tile=document.querySelector(`.tile[data-index="${index}"]`);
  tile?.classList.add("saved");
  try{
    await api("/api/multiplicity-labels",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({items:[{
        candidate_id:item.candidate_id,well:item.well,timepoint:item.timepoint,
        x_px:item.x_px,y_px:item.y_px,label,
        source:`quick_multiplicity_${state.mode}`
      }]})
    });
    setMessage(`${item.well} ${item.timepoint} 已标记为${labelNames[label]}`);
    await refreshStats();
    const next=[...document.querySelectorAll(".tile")].findIndex(
      (node,i)=>i>index&&!node.classList.contains("saved")
    );
    if(next>=0)focusTile(next);
  }catch(error){
    tile?.classList.remove("saved");setMessage(`保存失败：${error.message}`,true);
  }
}
document.querySelectorAll("[data-mode]").forEach(button=>button.onclick=()=>{
  state.mode=button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(item=>item.classList.toggle("active",item===button));
  loadBatch();
});
$("reloadButton").onclick=loadBatch;
$("pageSize").onchange=loadBatch;
document.addEventListener("keydown",event=>{
  if(event.target.matches("select,input,textarea"))return;
  const labels={"1":"single","2":"touching_doublet","3":"cluster_3plus","4":"not_cell"," ":"skip"};
  if(labels[event.key]){event.preventDefault();labelItem(state.focused,labels[event.key]);return}
  if(event.key==="ArrowRight"||event.key==="ArrowDown"){event.preventDefault();focusTile(state.focused+1)}
  if(event.key==="ArrowLeft"||event.key==="ArrowUp"){event.preventDefault();focusTile(state.focused-1)}
});
Promise.all([refreshStats(),loadBatch()]);
