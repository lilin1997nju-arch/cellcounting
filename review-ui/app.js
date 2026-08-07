const $=id=>document.getElementById(id);
const timepoints=["T0","T1","T2"];
const colors={cell:"#16b58d",debris:"#d86b35",irrelevant:"#7b5ca7",uncertain:"#dda52f"};
const modelNames={cell:"细胞",debris:"杂质/碎片",invalid:"无效"};
const temporalNames={live_cell:"活细胞",dead_cell:"死细胞",cell_unknown:"细胞（活死待定）",debris:"杂质/碎片",uncertain:"待定",unmarked:"不标记"};
const state={
  queue:[],filtered:[],reviews:new Map(),linkReviews:new Map(),index:0,item:null,context:null,
  selections:{},images:{},views:{},filter:"unreviewed",search:"",showSuppressed:false,
  contextMode:"densest",searchSize:1024,
  finalLabel:"uncertain",issues:new Set(),links:{"T0-T1":"uncertain","T1-T2":"uncertain"},
  linkScores:{"T0-T1":null,"T1-T2":null},linkReasons:{"T0-T1":"","T1-T2":""}
};

async function apiJson(url,options){
  const response=await fetch(url,options);
  if(!response.ok)throw new Error(await response.text());
  return response.json();
}
function defaultView(){return {zoom:1,panX:0,panY:0,markers:true,mode:"pan",pointer:null,dragged:false}}
function isEdgeItem(item){
  const center=1614.5;
  return Math.hypot(Number(item.x_px)-center,Number(item.y_px)-center)/3229>=0.4;
}
async function loadAll(){
  $("status").textContent="载入审核数据…";
  try{
    const [queue,reviews,links]=await Promise.all([
      apiJson("/api/review-candidates?limit=5000&include_completed=true"),
      apiJson("/api/lineage-reviews?limit=5000"),
      apiJson("/api/link-reviews?limit=5000")
    ]);
    state.queue=queue;state.reviews=new Map(reviews.map(row=>[row.canonical_target_id,row]));
    const latestRound=queue.find(item=>item.round_id)?.round_id||"无模型轮次";
    const pending=queue.filter(item=>!state.reviews.has(item.canonical_target_id)).length;
    $("currentRound").textContent=`${latestRound} · 新待审核 ${pending}`;
    state.linkReviews=new Map();
    for(const link of links){
      if(!state.linkReviews.has(link.canonical_target_id))state.linkReviews.set(link.canonical_target_id,{});
      state.linkReviews.get(link.canonical_target_id)[link.link_id]=link.link_label;
    }
    applyFilters(true);
  }catch(error){$("status").textContent=`载入失败：${error.message}`}
}
function applyFilters(selectFirst=false){
  const query=state.search.trim().toLowerCase();
  state.filtered=state.queue.filter(item=>{
    const reviewed=state.reviews.has(item.canonical_target_id);
    if(state.filter==="unreviewed"&&reviewed)return false;
    if(state.filter==="reviewed"&&!reviewed)return false;
    if(state.filter==="edge"&&!isEdgeItem(item))return false;
    if(state.filter==="model_cell"&&item.auto_label!=="cell")return false;
    if(state.filter==="model_debris"&&item.auto_label!=="debris")return false;
    if(state.filter==="model_uncertain"&&item.auto_status!=="needs_review")return false;
    return !query||`${item.well} ${item.canonical_target_id}`.toLowerCase().includes(query);
  });
  if(selectFirst||state.index>=state.filtered.length)state.index=0;
  renderSidebar();updateProgress();
  if(!state.filtered.length){
    $("emptyState").hidden=false;$("reviewContent").hidden=true;return;
  }
  $("emptyState").hidden=true;$("reviewContent").hidden=false;loadItem();
}
function updateProgress(){
  const reviewed=state.queue.filter(item=>state.reviews.has(item.canonical_target_id)).length;
  const percent=state.queue.length?reviewed/state.queue.length*100:0;
  $("progressText").textContent=`${reviewed} / ${state.queue.length} 已审核`;
  $("progressDetail").textContent=`${Math.round(percent)}% 完成`;
  $("progressBar").style.width=`${percent}%`;
}
function renderSidebar(){
  const groups=new Map();
  for(const item of state.filtered){
    if(!groups.has(item.well))groups.set(item.well,[]);
    groups.get(item.well).push(item);
  }
  $("lineageList").innerHTML=[...groups.entries()].map(([well,items])=>`
    <section class="well-group">
      <div class="well-title"><strong>${well}</strong><span>${items.length} 个目标</span></div>
      ${items.map(item=>{
        const selected=state.item?.canonical_target_id===item.canonical_target_id;
        const reviewed=state.reviews.has(item.canonical_target_id);
        const modelLabel=modelNames[item.auto_label]||"待定";
        const cellProbability=Number(item.cell_probability);
        const modelSummary=item.cell_probability!=null&&Number.isFinite(cellProbability)?
          `模型：${modelLabel} · ${temporalNames[item.temporal_label]||"时序待定"} · 细胞 ${(cellProbability*100).toFixed(1)}%`:
          `${item.candidate_source} · 置信度 ${Number(item.confidence).toFixed(3)}`;
        return `<button class="lineage-item ${selected?"selected":""}" data-target="${item.canonical_target_id}">
          <i class="${reviewed?"reviewed":isEdgeItem(item)?"edge":""}"></i>
          <span><b>${item.canonical_target_id}</b><small>${modelSummary}</small></span>
        </button>`}).join("")}
    </section>`).join("");
  document.querySelectorAll(".lineage-item").forEach(button=>button.onclick=()=>{
    state.index=state.filtered.findIndex(item=>item.canonical_target_id===button.dataset.target);loadItem();
  });
}
function nearestCandidate(tp,x,y,maxDistance=Infinity){
  const info=state.context.timepoints[tp];
  const pool=[...(info?.candidates||[]),...(state.showSuppressed?(info?.suppressed_candidates||[]):[])];
  let best=null,bestDistance=maxDistance;
  for(const candidate of pool){
    const distance=Math.hypot(candidate.x_px-x,candidate.y_px-y);
    if(distance<bestDistance){best=candidate;bestDistance=distance}
  }
  return best;
}
function normalizedPoint(point,fallback="uncertain"){
  const derivedDiameter=point.marker_diameter_px??point.equivalent_diameter_px??
    (point.area_px?Math.sqrt(Number(point.area_px)*4/Math.PI):8);
  return {
    x_px:point.x_px??null,y_px:point.y_px??null,candidate_id:point.candidate_id??null,
    area_px:point.area_px??null,object_label:point.object_label||fallback,
    marker_diameter_px:Math.max(3,Number(derivedDiameter)||8),
    orientation_rad:point.orientation_rad??null
  };
}
function expandProposedPoint(tp,rawPoint,fallback){
  const local=nearestCandidate(tp,Number(rawPoint.x_px),Number(rawPoint.y_px),24);
  const source=normalizedPoint({...local,...rawPoint},fallback);
  const count=Math.max(1,Math.min(3,Number(rawPoint.cell_count||1)));
  if(fallback!=="cell"||count===1)return[source];
  const diameter=Math.max(5,Number(source.marker_diameter_px||8));
  const childDiameter=Math.max(3,diameter*(count===2?.62:.52));
  const orientation=Number(source.orientation_rad||0);
  const points=[];
  for(let index=0;index<count;index++){
    let dx=0,dy=0;
    if(count===2){
      const direction=index===0?-1:1;
      dx=Math.sin(orientation)*diameter*.30*direction;
      dy=Math.cos(orientation)*diameter*.30*direction;
    }else{
      const angle=orientation+index*Math.PI*2/3;
      dx=Math.cos(angle)*diameter*.30;dy=Math.sin(angle)*diameter*.30;
    }
    points.push({...source,x_px:source.x_px+dx,y_px:source.y_px+dy,
      marker_diameter_px:childDiameter,
      candidate_id:`${source.candidate_id}:count:${index+1}`});
  }
  return points;
}
function hydrateModelProposal(item){
  const proposed=JSON.parse(item.proposal_points_json);
  state.selections={};
  for(const tp of timepoints){
    const raw=proposed[tp]||{present:false,additional_points:[]};
    if(!raw.present){
      state.selections[tp]={present:false,x_px:null,y_px:null,object_label:item.auto_label,additional_points:[]};
      continue;
    }
    const rawPoints=[raw,...(raw.additional_points||[])];
    const expanded=rawPoints.flatMap(point=>expandProposedPoint(tp,point,item.auto_label));
    state.selections[tp]={present:true,...expanded[0],additional_points:expanded.slice(1)};
  }
  state.finalLabel=item.proposed_final_label||(item.auto_label==="debris"?"debris":"cell_unknown");
  state.links={...{"T0-T1":"uncertain","T1-T2":"uncertain"},...JSON.parse(item.proposed_links_json||"{}")};
  $("morphology").value=item.proposed_morphology||"uncertain";
  $("division").value=item.proposed_division||"unknown";
  $("lineageStatus").value=item.proposed_lineage_status||"needs_review";
  $("reviewConfidence").value=Number(item.model_confidence)>=.75?"high":"medium";
  $("notes").value="";
}
function mergeDetectedCandidates(tp){
  const selection=state.selections[tp],info=state.context.timepoints[tp];
  if(!selection?.present||!info?.available)return;
  if(state.context.review_overrides?.[tp]?.use_all_detected_candidates===false)return;
  const existing=[selection,...(selection.additional_points||[])];
  for(const candidate of info.candidates||[]){
    const duplicate=existing.some(point=>
      (candidate.candidate_id&&point.candidate_id===candidate.candidate_id)||
      Math.hypot(point.x_px-candidate.x_px,point.y_px-candidate.y_px)<=
        Math.max(5,Number(candidate.equivalent_diameter_px||5))
    );
    if(!duplicate){
      const point=normalizedPoint(candidate);
      selection.additional_points.push(point);existing.push(point);
    }
  }
}
function applyReviewOverrides(){
  const overrides=state.context.review_overrides||{};
  for(const tp of timepoints){
    const requested=Number(overrides[tp]?.primary_count||0);
    if(requested===2&&pointRows(tp).length===1){
      splitPoint(tp,null);
      state.selections[tp].object_label="cell";
      state.selections[tp].additional_points[0].object_label="cell";
    }
  }
}
function hydrateExistingReview(review){
  const saved=JSON.parse(review.timepoint_points_json);
  for(const tp of timepoints){
    const point=saved[tp]||{present:false};
    state.selections[tp]={
      present:!!point.present,...normalizedPoint(point,review.object_type),
      additional_points:(point.additional_points||[]).map(child=>normalizedPoint(child,"uncertain"))
    };
  }
  state.finalLabel={
    "cell|live":"live_cell","cell|dead":"dead_cell","debris|not_applicable":"debris",
    "cell|unknown":"cell_unknown","irrelevant|not_applicable":"irrelevant"
  }[`${review.object_type}|${review.viability}`]||"uncertain";
  state.issues=new Set(JSON.parse(review.issue_tags||"[]"));
  state.links={...{"T0-T1":"uncertain","T1-T2":"uncertain"},...(state.linkReviews.get(review.canonical_target_id)||{})};
  $("morphology").value=review.morphology||"uncertain";$("division").value=review.division_state||"unknown";
  $("lineageStatus").value=review.lineage_status||"needs_review";$("reviewConfidence").value=review.review_confidence||"medium";
  $("reviewer").value=review.reviewer||"local_user";$("notes").value=review.notes||"";
}
function resetReviewFields(){
  state.finalLabel="uncertain";state.issues=new Set();state.links={"T0-T1":"uncertain","T1-T2":"uncertain"};
  state.linkScores={"T0-T1":null,"T1-T2":null};state.linkReasons={"T0-T1":"","T1-T2":""};
  $("morphology").value="uncertain";$("division").value="unknown";$("lineageStatus").value="needs_review";
  $("reviewConfidence").value="medium";$("notes").value="";$("reviewer").value=$("reviewer").value||"local_user";
}
async function loadItem(anchorOverride=null){
  if(!state.filtered.length)return;
  state.item=state.filtered[state.index];state.showSuppressed=false;state.images={};
  state.views=Object.fromEntries(timepoints.map(tp=>[tp,defaultView()]));resetReviewFields();
  state.linkScores={...state.linkScores,...JSON.parse(state.item.proposed_link_scores_json||"{}")};
  state.linkReasons={...state.linkReasons,...JSON.parse(state.item.proposed_link_reasons_json||"{}")};
  $("status").textContent="载入图像…";renderSidebar();
  const x=anchorOverride?.x_px??state.item.x_px,y=anchorOverride?.y_px??state.item.y_px;
  try{
    state.context=await apiJson(`/api/review-context?well=${state.item.well}&x=${x}&y=${y}&search_size=${state.searchSize}&view_mode=${state.contextMode}&target_id=${encodeURIComponent(state.item.candidate_id)}`);
    const review=state.reviews.get(state.item.canonical_target_id);
    let usedProposal=false;
    if(review)hydrateExistingReview(review);
    else if(state.item.proposal_points_json){
      hydrateModelProposal(state.item);usedProposal=true;
    }else{
      state.selections={};
      for(const tp of timepoints){
        const info=state.context.timepoints[tp];
        if(!info?.available){state.selections[tp]={present:false,additional_points:[]};continue}
        let selected=nearestCandidate(tp,info.center_x_px,info.center_y_px,80);
        if(tp==="T0"&&!selected)selected={x_px:x,y_px:y,candidate_id:state.item.candidate_id,area_px:null};
        state.selections[tp]=selected?{present:true,...normalizedPoint(selected),additional_points:[]}:
          {present:false,x_px:null,y_px:null,object_label:"uncertain",additional_points:[]};
        if(tp==="T0"&&state.selections[tp].present&&state.item.auto_label==="cell"){
          state.selections[tp].object_label="cell";
          $("morphology").value="cell_like";
        }
        if(state.selections[tp].present&&state.item.auto_label==="debris"){
          state.selections[tp].object_label="debris";
          $("morphology").value="debris_like";
        }
        mergeDetectedCandidates(tp);
      }
      if(["live_cell","dead_cell","debris"].includes(state.item.temporal_label)){
        state.finalLabel=state.item.temporal_label;
      }
    }
    if(review)for(const tp of timepoints)mergeDetectedCandidates(tp);
    applyReviewOverrides();
    for(const tp of ["T1","T2"])$(`present${tp}`).checked=!!state.selections[tp]?.present;
    await Promise.all(timepoints.map(loadPanelImage));
    renderCase();$("status").textContent=review?"已载入本轮审核结果":usedProposal?"已载入当前模型完整判定":"尚未保存";
  }catch(error){$("status").textContent=`载入失败：${error.message}`}
}
function loadImage(url){
  return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=url});
}
async function loadPanelImage(tp){
  const info=state.context.timepoints[tp];if(!info?.available)return;
  state.images[tp]=await loadImage(`/api/patch?well=${state.item.well}&timepoint=${tp}&x=${info.center_x_px}&y=${info.center_y_px}&size=${state.context.search_size_px}&v=${Date.now()}`);
  $(`overview${tp}`).src=`/api/well-image?well=${state.item.well}&timepoint=${tp}&max_size=500`;
}
function renderCase(){
  $("wellBadge").textContent=state.item.well;$("caseTitle").textContent=state.item.canonical_target_id;
  const probabilities=[
    ["细胞",state.item.cell_probability],
    ["杂质",state.item.debris_probability],
    ["无效",state.item.invalid_probability]
  ];
  const hasModelProbabilities=probabilities.every(([,value])=>
    value!=null&&Number.isFinite(Number(value))
  );
  if(hasModelProbabilities){
    const probabilityText=probabilities.map(([name,value])=>`${name} ${(Number(value)*100).toFixed(1)}%`).join(" · ");
    const status=state.item.auto_status==="needs_review"?"模型待定":"模型高置信";
    const temporal=temporalNames[state.item.temporal_label]||"时序待定";
    $("caseMeta").textContent=`三分类模型 · ${probabilityText} · 时序建议：${temporal} · ${status} · ${state.index+1}/${state.filtered.length}`;
  }else{
    $("caseMeta").textContent=`历史人工审核目标 · ${state.index+1}/${state.filtered.length}`;
  }
  const suppressed=timepoints.reduce((sum,tp)=>sum+(state.context.timepoints[tp]?.suppressed_candidates?.length||0),0);
  $("toggleSuppressed").textContent=`${state.showSuppressed?"隐藏":"显示"}孔壁低优先级候选（${suppressed}）`;
  $("toggleSuppressed").hidden=suppressed===0;
  $("toggleViewMode").textContent=state.contextMode==="densest"?"切换到谱系视野":"切换到最密细胞区";
  for(const tp of ["T1","T2"]){
    $(`heading${tp}`).textContent=`${tp} · ${state.contextMode==="densest"?"谱系内最密区域":"谱系附近视野"}`;
  }
  for(const tp of timepoints){drawPanel(tp);renderObjects(tp);updateOverview(tp);renderMarkTools(tp)}
  renderReviewControls();updateEvidence();
}
function toCanvas(tp,x,y){
  const info=state.context.timepoints[tp],size=state.context.search_size_px;
  return [(x-info.origin_x_px)/size*512,(y-info.origin_y_px)/size*512];
}
function applyView(ctx,tp){
  const view=state.views[tp];ctx.translate(256+view.panX,256+view.panY);ctx.scale(view.zoom,view.zoom);ctx.translate(-256,-256);
}
function drawCandidate(ctx,tp,candidate,suppressed=false){
  const [x,y]=toCanvas(tp,candidate.x_px,candidate.y_px);
  ctx.strokeStyle=suppressed?"#9b7770":"#dda52f";ctx.lineWidth=suppressed?1:1.5;ctx.setLineDash(suppressed?[5,4]:[]);
  ctx.beginPath();ctx.arc(x,y,Math.max(5,Math.min(17,candidate.equivalent_diameter_px*1.15)),0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
}
function drawModelCandidate(ctx,tp,candidate){
  const [x,y]=toCanvas(tp,candidate.x_px,candidate.y_px);
  const cell=["single","touching_doublet","cluster_3plus"].includes(candidate.integrated_label);
  ctx.strokeStyle=cell?"rgba(22,181,141,.72)":"rgba(216,107,53,.62)";
  ctx.lineWidth=1.5;ctx.setLineDash([3,2]);
  const diameter=Number(candidate.equivalent_diameter_px||8);
  const radius=Math.max(3,diameter/state.context.search_size_px*512/2*1.2);
  ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.stroke();ctx.setLineDash([]);
}
function drawSelection(ctx,tp,point,number){
  if(!point||point.x_px==null)return;const [x,y]=toCanvas(tp,point.x_px,point.y_px);
  const diameter=Number(point.marker_diameter_px)||8;
  const radius=Math.max(3,diameter/state.context.search_size_px*512/2*1.25);
  ctx.strokeStyle=colors[point.object_label]||colors.uncertain;ctx.lineWidth=2.4;
  ctx.beginPath();ctx.arc(x,y,radius,0,Math.PI*2);ctx.stroke();
  ctx.fillStyle=colors[point.object_label]||colors.uncertain;ctx.font="bold 8px sans-serif";
  ctx.fillText(String(number),x+radius+2,y-radius-1);
}
function drawPanel(tp){
  const canvas=$(`canvas${tp}`),ctx=canvas.getContext("2d"),info=state.context.timepoints[tp];
  ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,512,512);ctx.fillStyle="#666";ctx.fillRect(0,0,512,512);
  if(!info?.available)return;ctx.save();applyView(ctx,tp);
  if(state.images[tp])ctx.drawImage(state.images[tp],0,0,512,512);
  if(state.views[tp].markers){
    for(const candidate of info.model_candidates||[])drawModelCandidate(ctx,tp,candidate);
    if(!state.item?.proposal_points_json||state.showSuppressed){
      for(const candidate of info.candidates||[])drawCandidate(ctx,tp,candidate,false);
    }
    if(state.showSuppressed)for(const candidate of info.suppressed_candidates||[])drawCandidate(ctx,tp,candidate,true);
    ctx.strokeStyle="#4b9ae8";ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(246,256);ctx.lineTo(266,256);ctx.moveTo(256,246);ctx.lineTo(256,266);ctx.stroke();
    const selection=state.selections[tp];if(selection?.present)drawSelection(ctx,tp,selection,1);
    for(const [index,child] of (selection?.additional_points||[]).entries())drawSelection(ctx,tp,child,index+2);
  }
  ctx.restore();
  const hidden=info.suppressed_candidates?.length||0;
  const dense=state.context.representative_views?.[tp];
  const shift=tp==="T0"?"T0参考":`Δ(${info.align_shift_x_px.toFixed(1)}, ${info.align_shift_y_px.toFixed(1)})`;
  const lineageCount=dense?.whole_lineage_cell_count??0;
  const lineageObjects=dense?.whole_lineage_object_count??0;
  const scopeText=dense?.scope==="lineage_motion_prediction"?"运动预测中心 · 尚未链接":
    state.item.auto_label==="debris"?`谱系内目标 ${lineageObjects}`:`谱系内细胞 ${lineageCount}`;
  $(`meta${tp}`).textContent=`${scopeText}${dense?.scope==="current_lineage"?` · 当前区域 ${dense.cell_count}`:""} · ${shift}`;
  $(`zoom${tp}`).textContent=`${Math.round(state.views[tp].zoom*100)}%`;
}
function updateOverview(tp){
  const info=state.context.timepoints[tp];if(!info?.available)return;
  const marker=document.querySelector(`[data-whole="${tp}"] i`);
  marker.style.left=`${info.origin_x_px/info.image_width_px*100}%`;marker.style.top=`${info.origin_y_px/info.image_height_px*100}%`;
  marker.style.width=`${state.context.search_size_px/info.image_width_px*100}%`;marker.style.height=`${state.context.search_size_px/info.image_height_px*100}%`;
}
function pointRows(tp){
  const selection=state.selections[tp];if(!selection?.present)return[];
  return [{point:selection,index:null},...(selection.additional_points||[]).map((point,index)=>({point,index}))];
}
function renderObjects(tp){
  const rows=pointRows(tp);const container=$(`objects${tp}`);
  if(!rows.length){container.innerHTML="<p>未选择目标；先选择“＋细胞/杂质”等模式，再点击图像添加</p>";return}
  const totals=Object.fromEntries(["cell","debris","irrelevant","uncertain"].map(label=>[
    label,rows.filter(row=>row.point.object_label===label).length
  ]));
  container.innerHTML=`<p>全部标记 ${rows.length}：细胞 ${totals.cell} · 杂质 ${totals.debris} · 无关 ${totals.irrelevant} · 待定 ${totals.uncertain}</p>`+
    rows.map(({point,index},rowIndex)=>`
    <div class="object-row">
      <span>标记 ${rowIndex+1} · (${point.x_px.toFixed(0)}, ${point.y_px.toFixed(0)})</span>
      <div class="object-actions" data-tp="${tp}" data-index="${index==null?"main":index}">
        ${["cell","debris","irrelevant","uncertain"].map(label=>`<button class="${point.object_label===label?`on ${label}`:""}" data-label="${label}">${{cell:"细胞",debris:"杂质",irrelevant:"无关",uncertain:"待定"}[label]}</button>`).join("")}
        <span class="size-tools"><button data-size="-2">−</button><em>${Number(point.marker_diameter_px||8).toFixed(0)}px</em><button data-size="2">＋</button></span>
        <button class="split" data-split="1">拆成2个</button>
        ${index==null?"":'<button class="remove" data-remove="1">×</button>'}
      </div>
    </div>`).join("");
  container.querySelectorAll(".object-actions").forEach(actions=>actions.onclick=event=>{
    const index=actions.dataset.index==="main"?null:Number(actions.dataset.index);
    const point=index==null?state.selections[tp]:state.selections[tp].additional_points[index];
    if(event.target.dataset.label)point.object_label=event.target.dataset.label;
    if(event.target.dataset.size)point.marker_diameter_px=Math.max(3,Math.min(100,Number(point.marker_diameter_px||8)+Number(event.target.dataset.size)));
    if(event.target.dataset.split)splitPoint(tp,index);
    if(event.target.dataset.remove)state.selections[tp].additional_points.splice(index,1);
    renderObjects(tp);drawPanel(tp);
  });
}
function renderMarkTools(tp){
  const tools=document.querySelector(`[data-mark-tools="${tp}"]`),mode=state.views[tp]?.mode||"pan";
  tools.querySelectorAll("button").forEach(button=>button.classList.toggle("on",button.dataset.mode===mode));
  $(`canvas${tp}`).parentElement.classList.toggle("adding",mode!=="pan");
}
function splitPoint(tp,index){
  const selection=state.selections[tp];
  const point=index==null?selection:selection.additional_points[index];
  const diameter=Math.max(4,Number(point.marker_diameter_px||8));
  const orientation=Number(point.orientation_rad||0);
  const dx=Math.sin(orientation)*diameter*.38,dy=Math.cos(orientation)*diameter*.38;
  const childDiameter=Math.max(3,diameter*.55);
  const first={...point,x_px:point.x_px-dx,y_px:point.y_px-dy,marker_diameter_px:childDiameter,
    candidate_id:point.candidate_id?`${point.candidate_id}:split:1`:null};
  const second={...point,x_px:point.x_px+dx,y_px:point.y_px+dy,marker_diameter_px:childDiameter,
    candidate_id:point.candidate_id?`${point.candidate_id}:split:2`:null};
  if(index==null)state.selections[tp]={present:true,...first,additional_points:[second,...(selection.additional_points||[])]};
  else selection.additional_points.splice(index,1,first,second);
}
function canvasToGlobal(tp,event){
  const canvas=$(`canvas${tp}`),rect=canvas.getBoundingClientRect(),view=state.views[tp];
  const displayX=(event.clientX-rect.left)/rect.width*512,displayY=(event.clientY-rect.top)/rect.height*512;
  const baseX=(displayX-256-view.panX)/view.zoom+256,baseY=(displayY-256-view.panY)/view.zoom+256;
  const info=state.context.timepoints[tp],size=state.context.search_size_px;
  return {x:info.origin_x_px+baseX/512*size,y:info.origin_y_px+baseY/512*size,snapRadius:size*18/rect.width/view.zoom};
}
function selectAt(tp,event){
  const info=state.context.timepoints[tp];if(!info?.available)return;
  const mode=state.views[tp].mode;if(mode==="pan")return;
  const location=canvasToGlobal(tp,event),selection=state.selections[tp];
  let snap=nearestCandidate(tp,location.x,location.y,location.snapRadius);
  const existing=selection?.present?[selection,...(selection.additional_points||[])]:[];
  if(snap&&existing.some(point=>point.candidate_id&&point.candidate_id===snap.candidate_id))snap=null;
  const point=normalizedPoint(snap||{x_px:location.x,y_px:location.y,marker_diameter_px:8},mode);
  point.object_label=mode;
  if(selection?.present)selection.additional_points.push(point);
  else state.selections[tp]={present:true,...point,additional_points:[]};
  if(tp!=="T0")$(`present${tp}`).checked=true;renderObjects(tp);drawPanel(tp);updateEvidence();
}
function bindCanvas(tp){
  const wrap=$(`canvas${tp}`).parentElement;
  wrap.onpointerdown=event=>{
    const view=state.views[tp];view.pointer={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY};view.dragged=false;
    wrap.setPointerCapture(event.pointerId);
  };
  wrap.onpointermove=event=>{
    const view=state.views[tp];if(!view.pointer)return;
    const dx=event.clientX-view.pointer.x,dy=event.clientY-view.pointer.y;
    if(Math.hypot(event.clientX-view.pointer.startX,event.clientY-view.pointer.startY)>5){
      view.dragged=true;view.panX+=dx*512/wrap.clientWidth;view.panY+=dy*512/wrap.clientHeight;
      view.pointer.x=event.clientX;view.pointer.y=event.clientY;wrap.classList.add("dragging");drawPanel(tp);
    }
  };
  wrap.onpointerup=async event=>{
    const view=state.views[tp];wrap.classList.remove("dragging");
    if(view.dragged&&view.mode==="pan")await commitPanelPan(tp);
    else if(!view.dragged&&view.mode!=="pan")selectAt(tp,event);
    view.pointer=null;
  };
  wrap.onwheel=event=>{event.preventDefault();const view=state.views[tp];view.zoom=Math.max(1,Math.min(4,view.zoom*(event.deltaY<0?1.2:1/1.2)));drawPanel(tp)};
}
async function commitPanelPan(tp){
  const view=state.views[tp],info=state.context.timepoints[tp];
  if(!info?.available)return;
  const scale=state.context.search_size_px/512/Math.max(view.zoom,1);
  const centerX=info.center_x_px-view.panX*scale;
  const centerY=info.center_y_px-view.panY*scale;
  view.panX=0;view.panY=0;
  $("status").textContent=`正在加载 ${tp} 邻近区域…`;
  try{
    const url=`/api/review-context?well=${state.item.well}&x=${state.item.x_px}&y=${state.item.y_px}&search_size=${state.searchSize}&view_mode=${state.contextMode}&center_tp=${tp}&center_x=${centerX}&center_y=${centerY}&target_id=${encodeURIComponent(state.item.candidate_id)}`;
    const refreshed=await apiJson(url);
    state.context.timepoints[tp]=refreshed.timepoints[tp];
    if(refreshed.representative_views?.[tp])state.context.representative_views[tp]=refreshed.representative_views[tp];
    await loadPanelImage(tp);drawPanel(tp);updateOverview(tp);
    $("status").textContent=`${tp} 已移动到 (${Math.round(centerX)}, ${Math.round(centerY)})`;
  }catch(error){$("status").textContent=`区域加载失败：${error.message}`}
}
function alignedPoint(tp){
  const selection=state.selections[tp],info=state.context.timepoints[tp];if(!selection?.present||selection.x_px==null)return null;
  return{x:selection.x_px+(info.align_shift_x_px||0),y:selection.y_px+(info.align_shift_y_px||0)};
}
function nearbyCount(tp,radius=40){
  const selection=state.selections[tp],info=state.context.timepoints[tp];if(!selection?.present)return 0;
  return Math.max(1,(info.candidates||[]).filter(candidate=>Math.hypot(candidate.x_px-selection.x_px,candidate.y_px-selection.y_px)<=radius).length,1+(selection.additional_points||[]).length);
}
function updateEvidence(){
  if(!state.context)return;const base=alignedPoint("T0"),moves=[];
  for(const tp of ["T1","T2"]){const point=alignedPoint(tp);if(base&&point)moves.push(Math.hypot(point.x-base.x,point.y-base.y))}
  const maxPixels=moves.length?Math.max(...moves):null,res=state.context.resolution_um_per_pixel||2.08;
  $("maxMove").textContent=maxPixels==null?"—":`${(maxPixels*res).toFixed(1)} µm`;
  const a0=state.selections.T0?.area_px,a2=state.selections.T2?.area_px;$("areaChange").textContent=a0&&a2?`${(a2/a0).toFixed(2)}×`:"—";
  const n1=nearbyCount("T1"),n2=nearbyCount("T2");$("nearbyT1").textContent=n1||"—";$("nearbyT2").textContent=n2||"—";
  let text="仪器孔壁结构已从候选和审核队列中硬排除；人工确认的贴壁细胞仍会保留。";
  if(n1>=2||n2>=2)text="附近存在多个细胞样目标，请确认是可信分裂还是邻近目标误连。";
  else if(maxPixels!=null&&maxPixels*res>state.context.significant_displacement_um)text="存在明显配准后位移，更支持活细胞运动，但仍需结合形态排除误链接。";
  else if(maxPixels!=null)text="目标较静止；仅凭静止或单张形态不能判为死细胞，需结合持续不分裂、无明显位移等跨时间点证据。";
  const cellProbability=Number(state.item.cell_probability);
  const modelEvidence=state.item.cell_probability!=null&&Number.isFinite(cellProbability)?
    (state.item.auto_label==="cell"?
      `静态模型支持细胞形态（细胞概率 ${(cellProbability*100).toFixed(1)}%）；活/死仍只依据时间序列。`:
      state.item.auto_label==="debris"?
      `静态模型支持杂质形态（杂质概率 ${(Number(state.item.debris_probability)*100).toFixed(1)}%）；跨时间位置由漂移感知链接给出。`:
      `静态模型尚未达到自动细胞阈值（细胞概率 ${(cellProbability*100).toFixed(1)}%），请优先核对形态。`):"";
  const temporalEvidence=state.item.temporal_label&&state.item.temporal_label!=="unmarked"?
    `时序模型建议“${temporalNames[state.item.temporal_label]||"待定"}”（${state.item.temporal_evidence||"证据不足"}）。`:"";
  $("evidenceText").textContent=`${modelEvidence} ${temporalEvidence} ${text}`.trim();
}
function renderReviewControls(){
  document.querySelectorAll("#finalLabels button").forEach(button=>button.classList.toggle("on",button.dataset.value===state.finalLabel));
  document.querySelectorAll("#issueTags button").forEach(button=>button.classList.toggle("on",state.issues.has(button.dataset.value)));
  document.querySelectorAll(".segmented").forEach(group=>group.querySelectorAll("button").forEach(button=>button.classList.toggle("on",button.dataset.value===state.links[group.dataset.link])));
  for(const linkId of ["T0-T1","T1-T2"]){
    const score=state.linkScores[linkId],label=state.links[linkId],badge=$(`score${linkId.replace("-","")}`);
    badge.className=`link-score ${label==="correct"?"high":"low"}`;
    badge.textContent=score==null?"不建立谱系边":`${label==="correct"?"链接置信":"低置信/未链接"} ${(Number(score)*100).toFixed(0)}%`;
    badge.title=state.linkReasons[linkId]||"";
    $(`reason${linkId.replace("-","")}`).textContent=state.linkReasons[linkId]||"";
  }
}
function labelMapping(){
  return {live_cell:["cell","live"],dead_cell:["cell","dead"],cell_unknown:["cell","unknown"],debris:["debris","not_applicable"],irrelevant:["irrelevant","not_applicable"],uncertain:["uncertain","unknown"]}[state.finalLabel];
}
async function saveReview(){
  if(!state.selections.T0?.present){$("status").textContent="必须先确认T0来源位置";return}
  const [objectType,viability]=labelMapping(),points={};
  for(const tp of timepoints){const selection=state.selections[tp]||{present:false};points[tp]={
    present:!!selection.present,x_px:selection.x_px??null,y_px:selection.y_px??null,candidate_id:selection.candidate_id??null,
    area_px:selection.area_px??null,object_label:selection.object_label||"uncertain",
    marker_diameter_px:selection.marker_diameter_px??null,orientation_rad:selection.orientation_rad??null,
    additional_points:(selection.additional_points||[]).map(point=>normalizedPoint(point))
  }}
  const payload={
    sequence_id:state.item.sequence_id,plate_id:state.item.plate_id,well:state.item.well,
    canonical_target_id:state.item.canonical_target_id,object_type:objectType,viability,
    division_state:$("division").value,morphology:$("morphology").value,points,
    lineage_status:$("lineageStatus").value,review_confidence:$("reviewConfidence").value,
    issue_tags:[...state.issues],links:[
      {link_id:"T0-T1",parent_timepoint:"T0",child_timepoint:"T1",link_label:state.links["T0-T1"]},
      {link_id:"T1-T2",parent_timepoint:"T1",child_timepoint:"T2",link_label:state.links["T1-T2"]}
    ],reviewer:$("reviewer").value.trim()||"local_user",notes:$("notes").value
  };
  $("status").textContent="保存中…";
  try{
    await apiJson("/api/lineage-reviews",{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(payload)});
    state.reviews.set(state.item.canonical_target_id,{...payload,timepoint_points_json:JSON.stringify(points),issue_tags:JSON.stringify(payload.issue_tags)});
    state.linkReviews.set(state.item.canonical_target_id,state.links);updateProgress();renderSidebar();
    $("status").textContent="已保存";moveToNext(true);
  }catch(error){$("status").textContent=`保存失败：${error.message}`}
}
async function approveAllAndNext(){
  $("reviewConfidence").value="high";
  $("status").textContent="确认模型分类、位置与链接全部正确…";
  await saveReview();
}
function moveToNext(preferUnreviewed=false){
  if(!state.filtered.length)return;
  let next=(state.index+1)%state.filtered.length;
  if(preferUnreviewed){
    const found=state.filtered.findIndex((item,index)=>index>state.index&&!state.reviews.has(item.canonical_target_id));
    const wrap=state.filtered.findIndex(item=>!state.reviews.has(item.canonical_target_id));
    next=found>=0?found:wrap>=0?wrap:next;
  }
  state.index=next;loadItem();
}
function openWhole(tp){
  state.wholeTp=tp;$("wholeTitle").textContent=`${state.item.well} · ${tp}`;$("wholeDialog").showModal();
  renderWhole();
}
function renderWhole(){
  const tp=state.wholeTp,info=state.context.timepoints[tp];if(!info?.available)return;
  $("wholeImage").src=`/api/well-image?well=${state.item.well}&timepoint=${tp}&max_size=1400`;
  const crop=$("wholeCrop");crop.style.left=`${info.origin_x_px/info.image_width_px*100}%`;crop.style.top=`${info.origin_y_px/info.image_height_px*100}%`;
  crop.style.width=`${state.context.search_size_px/info.image_width_px*100}%`;crop.style.height=`${state.context.search_size_px/info.image_height_px*100}%`;
  document.querySelectorAll("[data-whole-tab]").forEach(button=>button.classList.toggle("on",button.dataset.wholeTab===tp));
}

for(const tp of timepoints)bindCanvas(tp);
document.querySelectorAll("[data-view]").forEach(button=>button.onclick=()=>{
  const view=state.views[button.dataset.tp],action=button.dataset.view;
  if(action==="zoomIn")view.zoom=Math.min(4,view.zoom*1.25);
  if(action==="zoomOut")view.zoom=Math.max(1,view.zoom/1.25);
  if(action==="reset"){Object.assign(view,defaultView());renderMarkTools(button.dataset.tp)}
  if(action==="markers")view.markers=!view.markers;drawPanel(button.dataset.tp);
});
document.querySelectorAll("[data-mark-tools]").forEach(tools=>tools.onclick=event=>{
  const mode=event.target.dataset.mode;if(!mode)return;
  const tp=tools.dataset.markTools;state.views[tp].mode=mode;renderMarkTools(tp);
});
document.querySelectorAll("[data-clear]").forEach(button=>button.onclick=()=>{
  const tp=button.dataset.clear;state.selections[tp]={present:false,x_px:null,y_px:null,object_label:"uncertain",additional_points:[]};
  if(tp!=="T0")$(`present${tp}`).checked=false;renderObjects(tp);drawPanel(tp);updateEvidence();
});
for(const tp of ["T1","T2"])$(`present${tp}`).onchange=event=>{
  state.selections[tp]=state.selections[tp]||{additional_points:[]};state.selections[tp].present=event.target.checked;renderObjects(tp);drawPanel(tp);updateEvidence();
};
document.querySelectorAll("[data-whole]").forEach(button=>button.onclick=()=>openWhole(button.dataset.whole));
document.querySelectorAll("[data-whole-tab]").forEach(button=>button.onclick=()=>{state.wholeTp=button.dataset.wholeTab;renderWhole()});
$("closeWhole").onclick=()=>$("wholeDialog").close();
$("toggleSuppressed").onclick=()=>{state.showSuppressed=!state.showSuppressed;renderCase()};
$("toggleViewMode").onclick=()=>{state.contextMode=state.contextMode==="densest"?"lineage":"densest";loadItem()};
$("recenter").onclick=()=>loadItem(state.selections.T0);
$("finalLabels").onclick=event=>{if(event.target.dataset.value){state.finalLabel=event.target.dataset.value;renderReviewControls()}};
$("issueTags").onclick=event=>{const value=event.target.dataset.value;if(!value)return;state.issues.has(value)?state.issues.delete(value):state.issues.add(value);renderReviewControls()};
document.querySelectorAll(".segmented").forEach(group=>group.onclick=event=>{if(event.target.dataset.value){state.links[group.dataset.link]=event.target.dataset.value;renderReviewControls()}});
document.querySelectorAll(".filter-row button").forEach(button=>button.onclick=()=>{
  state.filter=button.dataset.filter;document.querySelectorAll(".filter-row button").forEach(item=>item.classList.toggle("active",item===button));applyFilters(true);
});
$("search").oninput=event=>{state.search=event.target.value;applyFilters(true)};
$("reloadQueue").onclick=loadAll;$("skip").onclick=()=>moveToNext(false);$("save").onclick=saveReview;$("approveAll").onclick=approveAllAndNext;
document.addEventListener("keydown",event=>{
  if(["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return;
  if(event.key.toLowerCase()==="j"||event.key==="ArrowRight")moveToNext(false);
  if(event.key.toLowerCase()==="k"||event.key==="ArrowLeft"){state.index=(state.index-1+state.filtered.length)%state.filtered.length;loadItem()}
  if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"){event.preventDefault();saveReview()}
});
loadAll();
