const $=id=>document.getElementById(id);
const reviewBaseUrl = document.querySelector('meta[name="review-base-url"]')?.content || "";
const reviewUrl = url => reviewBaseUrl && String(url).startsWith("/")
  ? `${reviewBaseUrl}${url}`
  : url;
const state={mode:"seed",items:[],focused:0,anchor:null,busy:false};

async function api(url,options){
  const response=await fetch(reviewUrl(url),options);
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}

function setMessage(text,error=false){
  $("message").textContent=text;
  $("message").style.color=error?"#a43c32":"#176c68";
}

async function refreshStats(){
  const stats=await api("/api/teach-stats");
  $("totalCount").textContent=stats.total;
  $("cellCount").textContent=stats.counts.cell||0;
  $("debrisCount").textContent=stats.counts.debris||0;
  $("invalidCount").textContent=stats.counts.invalid||0;
  if(stats.model?.status==="ready"){
    $("modelState").textContent="增量模型已就绪";
    $("modelDetail").textContent=`${stats.model.training_samples} 个训练样本 · 拟合 ${(stats.model.training_accuracy*100).toFixed(1)}%`;
  }else{
    $("modelState").textContent="尚未训练";
    $("modelDetail").textContent="先确认一批细胞种子";
  }
}

function probabilityText(item){
  if(item.similarity!=null)return `相似 ${(item.similarity*100).toFixed(0)}%`;
  if(item.cell_probability!=null)return `细胞 ${(item.cell_probability*100).toFixed(0)}%`;
  return item.pseudo_label==="cell"?"细胞种子":"待判断";
}

function render(){
  const grid=$("grid");grid.innerHTML="";
  $("empty").hidden=state.items.length>0;
  state.items.forEach((item,index)=>{
    const tile=$("tileTemplate").content.firstElementChild.cloneNode(true);
    tile.dataset.index=index;
    tile.querySelector("img").src=reviewUrl(`/api/patch?well=${item.well}&timepoint=T0&x=${item.x_px}&y=${item.y_px}&size=128`);
    tile.querySelector(".index-badge").textContent=index+1;
    tile.querySelector(".probability").textContent=probabilityText(item);
    tile.querySelector(".well").textContent=item.well;
    tile.querySelector(".candidate").textContent=item.candidate_id;
    tile.onclick=()=>focusTile(index);
    tile.querySelectorAll("[data-label]").forEach(button=>{
      button.onclick=event=>{event.stopPropagation();labelItem(index,button.dataset.label)};
    });
    tile.querySelector(".similar").onclick=event=>{
      event.stopPropagation();showSimilar(index);
    };
    grid.appendChild(tile);
  });
  focusTile(0);
}

function focusTile(index){
  if(!state.items.length)return;
  state.focused=Math.max(0,Math.min(index,state.items.length-1));
  document.querySelectorAll(".tile").forEach((tile,tileIndex)=>{
    tile.classList.toggle("focused",tileIndex===state.focused);
  });
  document.querySelector(`.tile[data-index="${state.focused}"]`)?.focus({preventScroll:true});
}

async function loadBatch(){
  if(state.busy)return;state.busy=true;
  setMessage("正在加载候选…");
  try{
    const size=Number($("pageSize").value);
    const anchor=state.mode==="similar"&&state.anchor?`&anchor_id=${encodeURIComponent(state.anchor)}`:"";
    state.items=await api(`/api/teach-candidates?mode=${state.mode}&limit=${size}${anchor}`);
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
    await api("/api/teach-labels",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({items:[{candidate_id:item.candidate_id,well:item.well,timepoint:"T0",x_px:item.x_px,y_px:item.y_px,label,source:`quick_teaching_${state.mode}`}]})
    });
    setMessage(`${item.well} 已标记为 ${label}`);
    await refreshStats();
    const next=[...document.querySelectorAll(".tile")].findIndex((node,i)=>i>index&&!node.classList.contains("saved"));
    if(next>=0)focusTile(next);
  }catch(error){tile?.classList.remove("saved");setMessage(`保存失败：${error.message}`,true)}
}

function showSimilar(index){
  const item=state.items[index];if(!item)return;
  state.anchor=item.candidate_id;state.mode="similar";
  $("similarMode").disabled=false;
  document.querySelectorAll("[data-mode]").forEach(button=>button.classList.toggle("active",button.dataset.mode==="similar"));
  loadBatch();
}

async function train(){
  if(state.busy)return;state.busy=true;
  $("trainButton").disabled=true;setMessage("正在提取视觉特征并训练；首次运行可能需要数分钟…");
  try{
    const result=await api("/api/teach-train",{method:"POST"});
    setMessage(`训练完成：${result.training_samples} 个样本，训练拟合 ${(result.training_accuracy*100).toFixed(1)}%`);
    await refreshStats();
    state.mode="uncertain";
    document.querySelectorAll("[data-mode]").forEach(button=>button.classList.toggle("active",button.dataset.mode==="uncertain"));
    state.busy=false;
    await loadBatch();
  }catch(error){setMessage(`训练失败：${error.message}`,true)}
  finally{state.busy=false;$("trainButton").disabled=false}
}

document.querySelectorAll("[data-mode]").forEach(button=>button.onclick=()=>{
  if(button.disabled)return;state.mode=button.dataset.mode;
  document.querySelectorAll("[data-mode]").forEach(item=>item.classList.toggle("active",item===button));
  loadBatch();
});
$("reloadButton").onclick=loadBatch;
$("pageSize").onchange=loadBatch;
$("trainButton").onclick=train;

document.addEventListener("keydown",event=>{
  if(event.target.matches("select,input,textarea"))return;
  const labels={"1":"cell","2":"debris","3":"invalid"," ":"skip"};
  if(labels[event.key]){event.preventDefault();labelItem(state.focused,labels[event.key]);return}
  if(event.key.toLowerCase()==="s"){event.preventDefault();showSimilar(state.focused);return}
  if(event.key==="ArrowRight"||event.key==="ArrowDown"){event.preventDefault();focusTile(state.focused+1)}
  if(event.key==="ArrowLeft"||event.key==="ArrowUp"){event.preventDefault();focusTile(state.focused-1)}
});

Promise.all([refreshStats(),loadBatch()]);
