const $=id=>document.getElementById(id);
const reviewBaseUrl = document.querySelector('meta[name="review-base-url"]')?.content || "";
const reviewUrl = url => reviewBaseUrl && String(url).startsWith("/")
  ? `${reviewBaseUrl}${url}`
  : url;
const state={mode:"all",items:[],focused:0,busy:false,roundId:""};
const labelNames={
  single:"单个细胞",
  touching_doublet:"相连两个细胞",
  cluster_3plus:"3个以上细胞",
  debris:"杂质",
  invalid:"无效/无关物",
  uncertain:"无法判定"
};

async function api(url,options){
  const response=await fetch(reviewUrl(url),options);
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}
function pct(value){return `${Math.round(100*Number(value||0))}%`}
function setMessage(text,error=false){
  $("message").textContent=text;
  $("message").style.color=error?"#a43c32":"#176c68";
}
async function refreshStats(){
  const [stats,wells]=await Promise.all([
    api("/api/integrated-review-stats"),
    api("/api/well-conclusions?limit=200")
  ]);
  if(stats.status!=="ready")return;
  state.roundId=stats.round_id;
  $("roundId").textContent=stats.round_id;
  $("reviewableCount").textContent=stats.review_queue_count||0;
  const counts=stats.label_counts||{};
  $("cellCount").textContent=(counts.single||0)+(counts.touching_doublet||0)+(counts.cluster_3plus||0);
  $("debrisCount").textContent=counts.debris||0;
  $("uncertainCount").textContent=counts.uncertain||0;
  $("reviewedCount").textContent=stats.reviewed_count||0;
  $("growthCount").textContent=wells.filter(row=>row.growth_status==="confirmed_growth").length;
  $("singleOriginCount").textContent=wells.filter(row=>row.origin_conclusion==="single_cell_origin").length;
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
  state.items.forEach((item,index)=>{
    const tile=$("tileTemplate").content.firstElementChild.cloneNode(true);
    tile.dataset.index=index;
    tile.querySelector("img").src=reviewUrl(`/api/patch?well=${item.well}&timepoint=${item.timepoint}&x=${item.x_px}&y=${item.y_px}&size=192`);
    tile.querySelector(".index-badge").textContent=index+1;
    const shownLabel=item.reviewed_label||item.integrated_label;
    const badge=tile.querySelector(".prediction");
    badge.textContent=`${labelNames[shownLabel]||shownLabel} · ${pct(item.integrated_confidence)}`;
    badge.classList.add(item.integrated_label);
    tile.querySelector(".well").textContent=`${item.well} · ${item.timepoint}`;
    tile.querySelector(".candidate").textContent=item.candidate_id;
    tile.querySelector(".confidence").innerHTML=item.reviewed_label
      ?`<span class="review-note">已审核：${labelNames[item.reviewed_label]}</span>`
      :`面积 ${Math.round(Number(item.area_px))} px`;
    tile.querySelector(".morph-probs").textContent=
      `形态：细胞 ${pct(item.cell_probability)} · 杂质 ${pct(item.debris_probability)} · 无效 ${pct(item.invalid_probability)}`;
    tile.querySelector(".count-probs").textContent=
      `数量：1个 ${pct(item.single_probability)} · 2个 ${pct(item.touching_doublet_probability)} · ≥3 ${pct(item.cluster_3plus_probability)}`;
    const ring=tile.querySelector(".size-ring");
    const ringPercent=Math.max(7,Math.min(66,Number(item.diameter_px||16)/192*100*1.2));
    ring.style.width=`${ringPercent}%`;ring.style.height=`${ringPercent}%`;
    if(item.reviewed_label)tile.classList.add("reviewed",item.decision||"");
    tile.onclick=()=>focusTile(index);
    tile.querySelector("[data-action='approve']").onclick=event=>{
      event.stopPropagation();reviewItem(index,item.integrated_label);
    };
    tile.querySelectorAll("[data-label]").forEach(button=>{
      button.onclick=event=>{
        event.stopPropagation();reviewItem(index,button.dataset.label);
      };
    });
    grid.appendChild(tile);
  });
  focusTile(0);
}
async function loadBatch(){
  if(state.busy)return;
  state.busy=true;setMessage("正在加载联合判定结果…");
  try{
    const limit=Number($("pageSize").value);
    state.items=await api(`/api/integrated-review-candidates?mode=${state.mode}&limit=${limit}`);
    state.focused=0;render();
    setMessage(`已载入 ${state.items.length} 个对象；黄色圆圈按对象实际直径显示`);
  }catch(error){setMessage(`加载失败：${error.message}`,true)}
  finally{state.busy=false}
}
async function reviewItem(index,reviewedLabel){
  const item=state.items[index];
  if(!item||state.busy)return;
  state.busy=true;
  const tile=document.querySelector(`.tile[data-index="${index}"]`);
  try{
    await api("/api/integrated-review-labels",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        round_id:item.integrated_round_id||state.roundId,
        items:[{
          candidate_id:item.candidate_id,
          predicted_label:item.integrated_label,
          reviewed_label:reviewedLabel
        }]
      })
    });
    const same=reviewedLabel===item.integrated_label;
    tile?.classList.add("saved",same?"approved":"corrected");
    setMessage(`${item.well} ${item.timepoint}：${same?"确认正确":"已改判为 "+labelNames[reviewedLabel]}`);
    await refreshStats();
    const next=[...document.querySelectorAll(".tile")].findIndex(
      (node,i)=>i>index&&!node.classList.contains("saved")
    );
    if(next>=0)focusTile(next);
  }catch(error){
    setMessage(`保存失败：${error.message}`,true);
  }finally{state.busy=false}
}
document.querySelectorAll("[data-mode]").forEach(button=>button.onclick=()=>{
  state.mode=button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(item=>item.classList.toggle("active",item===button));
  loadBatch();
});
$("reloadButton").onclick=loadBatch;
$("pageSize").onchange=loadBatch;
$("trainButton").onclick=async()=>{
  if(state.busy)return;
  state.busy=true;
  $("trainButton").disabled=true;
  setMessage("正在用审核结果重训形态与数量模型，并更新孔级筛选结果…");
  try{
    const result=await api("/api/integrated-review-new-round",{method:"POST"});
    setMessage(`新一轮已完成：${result.integrated_round.round_id}`);
    await refreshStats();await loadBatch();
  }catch(error){setMessage(`重训失败：${error.message}`,true)}
  finally{state.busy=false;$("trainButton").disabled=false}
};
document.addEventListener("keydown",event=>{
  if(event.target.matches("select,input,textarea,button"))return;
  const item=state.items[state.focused];
  if(!item)return;
  const labels={
    "1":"single","2":"touching_doublet","3":"cluster_3plus",
    "4":"debris","5":"invalid","6":"uncertain",
    " ":item.integrated_label
  };
  if(labels[event.key]){
    event.preventDefault();reviewItem(state.focused,labels[event.key]);return;
  }
  if(event.key==="ArrowRight"||event.key==="ArrowDown"){
    event.preventDefault();focusTile(state.focused+1);
  }
  if(event.key==="ArrowLeft"||event.key==="ArrowUp"){
    event.preventDefault();focusTile(state.focused-1);
  }
});
Promise.all([refreshStats(),loadBatch()]);
