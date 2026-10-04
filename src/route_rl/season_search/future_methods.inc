// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
 // Daily continuation paths chosen by the tile DP, valued in a JOINT market.
 // Wheat/carrot renewal follows enumerate(); other future empty tiles stay empty.
 // All quantities and seed expenses are integral; labour is in fifth-turns.
 // This permits exact delta updates without accumulating floating-point drift.
 struct Projection { int q[30][9]{}; int labour[30]{};int work[30]{}; };
 Projection baseline_projection;
 vector<vector<Projection>> projections;
 vector<vector<array<uint64_t,9>>> projection_flow_fp;
 Projection aggregate_projection;
 // Pad the final SIMD load; entries beyond the task count remain 255.
 array<uint8_t,112> aggregate_choices{};
 // Monotone revision also catches projection updates outside evaluation,
 // including continuation_prices. Returning to an old choice is a cache miss.
 uint64_t projection_revision=0;
 bool aggregate_ready=false;bool labour_cache_valid=false;double labour_cache=0.;array<double,30> labour_day_cost{};
 array<uint64_t,30> flow_salt{};array<uint64_t,9> baseline_flow_fp{},aggregate_flow_fp{};array<array<uint64_t,101>,9> holding_flow_fp{};
 static inline uint64_t flow_mix64(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
 void initialize_flow_fingerprints(){
  for(int d=0;d<30;d++)flow_salt[d]=flow_mix64(0x243f6a8885a308d3ULL+d);
  baseline_flow_fp.fill(0);for(int c=0;c<9;c++)for(int d=day+1;d<30;d++)baseline_flow_fp[c]+=(uint64_t)(int64_t)baseline_projection.q[d][c]*flow_salt[d];
  for(int c=0;c<9;c++){int scheduled=0;for(int d=day+1;d<30;d++)scheduled+=hold_schedule[d][c];for(int h=0;h<=100;h++){uint64_t a=0;int prefix=0,assigned=0;if(!scheduled){if(day+1<30)a+=(uint64_t)(int64_t)h*flow_salt[day+1];}else for(int d=day+1;d<30;d++){prefix+=hold_schedule[d][c];int amount=h*prefix/scheduled;int q=amount-assigned;assigned=amount;a+=(uint64_t)(int64_t)q*flow_salt[d];}holding_flow_fp[c][h]=a;}}
  aggregate_flow_fp=baseline_flow_fp;
 }
 const Projection& update_plan_projection(const Plan&p){
  if(!aggregate_ready){aggregate_projection=baseline_projection;aggregate_flow_fp=baseline_flow_fp;aggregate_choices.fill(255);aggregate_ready=true;}
  uint8_t chosen[112];memset(chosen,255,sizeof(chosen));
  int current_terms=0;
  for(const auto&r:p.routes)for(auto [ti,oi]:r.tasks){current_terms+=chosen[ti]==255;chosen[ti]=(uint8_t)oi;}
  uint8_t changed[100];int nchanged=0,changed_work=0;
  const int nt=(int)tasks.size();
#if defined(__SSE2__)
  for(int base=0;base<nt;base+=16){
   auto now=_mm_loadu_si128(reinterpret_cast<const __m128i*>(chosen+base));
   auto old=_mm_loadu_si128(reinterpret_cast<const __m128i*>(aggregate_choices.data()+base));
   unsigned bits=unsigned(_mm_movemask_epi8(_mm_cmpeq_epi8(now,old)))^0xffffu;
   for(;bits;bits&=bits-1){int ti=base+__builtin_ctz(bits);changed[nchanged++]=(uint8_t)ti;changed_work+=(chosen[ti]!=255)+(aggregate_choices[ti]!=255);}
  }
#else
  for(int ti=0;ti<nt;ti++){
   uint8_t now=chosen[ti],old=aggregate_choices[ti];
   if(now!=old){changed[nchanged++]=(uint8_t)ti;changed_work+=(now!=255)+(old!=255);}
  }
#endif
  if(nchanged){labour_cache_valid=false;projection_revision++;}
  auto add=[&](int ti,uint8_t oi,int sign){
   if(oi==255)return;
   const auto&x=projections[ti][oi];const auto&fp=projection_flow_fp[ti][oi];
   for(int c=0;c<9;c++){if(sign>0)aggregate_flow_fp[c]+=fp[c];else aggregate_flow_fp[c]-=fp[c];}
   for(int d=day+1;d<30;d++){aggregate_projection.labour[d]+=sign*x.labour[d];aggregate_projection.work[d]+=sign*x.work[d];}
  };
  if(changed_work>current_terms){
   aggregate_projection=baseline_projection;aggregate_flow_fp=baseline_flow_fp;aggregate_choices.fill(255);
   nchanged=0;for(int ti=0;ti<nt;ti++)if(chosen[ti]!=255)changed[nchanged++]=(uint8_t)ti;
  }
  for(int z=0;z<nchanged;z++){
   int ti=changed[z];uint8_t old=aggregate_choices[ti],now=chosen[ti];add(ti,old,-1);add(ti,now,1);aggregate_choices[ti]=now;
  }
  return aggregate_projection;
 }
 Projection plan_projection(const Plan&p,const int*carry){
  update_plan_projection(p);
  Projection proj=baseline_projection;
  for(int d=day+1;d<30;d++){proj.labour[d]=aggregate_projection.labour[d];proj.work[d]=aggregate_projection.work[d];}
  for(int ti=0;ti<(int)tasks.size();ti++){
   uint8_t oi=aggregate_choices[ti];if(oi==255)continue;const auto&x=projections[ti][oi];
   for(int d=day+1;d<30;d++)for(int c=0;c<9;c++)proj.q[d][c]+=x.q[d][c];
  }
  for(int c=0;c<9;c++){proj.q[day+1][c]+=carry[c];int scheduled=0;for(int d=day+1;d<30;d++)scheduled+=hold_schedule[d][c];if(!scheduled){proj.q[day+1][c]+=p.holding[c];continue;}int prefix=0,assigned=0;for(int d=day+1;d<30;d++){prefix+=hold_schedule[d][c];int amount=p.holding[c]*prefix/scheduled;proj.q[d][c]+=amount-assigned;assigned=amount;}}
  return proj;
 }
 struct ProjectionKeyHash{size_t operator()(const array<int,12>&k)const{uint64_t h=0xcbf29ce484222325ULL;for(int x:k){h^=uint32_t(x);h*=0x100000001b3ULL;}return h;}};
 unordered_map<array<int,12>,Projection,ProjectionKeyHash> projection_memo;
 Projection project(Tile t,int d){
  if(d>=30||(t.kind!=3&&t.kind!=6))return Projection{};
  array<int,12> k{d,t.kind,t.item,t.age,t.y,t.dry,t.water,t.fert,t.care,t.fa,t.pending,t.decay};
  auto it=projection_memo.find(k);if(it!=projection_memo.end())return it->second;
  auto result=project_uncached(t,d);projection_memo.emplace(k,result);return result;
 }
 Projection project_uncached(Tile t,int d){
  Projection p;
  for(;d<30;d++){
   if(t.kind!=3&&t.kind!=6)break;
   auto opts=enumerate(t,d);int best=0;double bv=-1e100;
   for(int i=0;i<(int)opts.size();i++){
    auto&o=opts[i];double v=o.immediate-future_action_cost(o,d);
    if(d<29)v+=time_discount(d)*value(refresh(o.next,d),d+1);
    if(v>bv){bv=v;best=i;}
   }
   const auto&o=opts[best];
   for(int c=0;c<9;c++)p.q[d][c]+=o.goods[c];
   p.q[d][0]-=o.w;p.q[d][8]-=o.f;
   p.labour[d]=0;for(int c=0;c<5;c++)p.labour[d]+=seedcost[c]*o.seeds[c];p.work[d]=5*o.acts.size()+(o.acts.empty()?0:7)+(o.w>0)+(o.f>0);
   t=refresh(o.next,d);
  }
  return p;
 }
 void make_projections(){
  if(day>=29)return;
  vector<Projection> no(100);
  for(int pos=0;pos<100;pos++){
   no[pos]=project(refresh(tiles[pos],day),day+1);
   for(int d=day+1;d<30;d++){
    for(int c=0;c<9;c++)baseline_projection.q[d][c]+=no[pos].q[d][c];
    baseline_projection.labour[d]+=no[pos].labour[d];baseline_projection.work[d]+=no[pos].work[d];
   }
  }
  for(const auto&t:tasks){
   vector<Projection> options;
   for(const auto&o:t.opts){
    auto p=project(refresh(o.next,day),day+1);
    for(int d=day+1;d<30;d++){
     for(int c=0;c<9;c++)p.q[d][c]-=no[t.pos].q[d][c];
     p.labour[d]-=no[t.pos].labour[d];p.work[d]-=no[t.pos].work[d];
    }
    options.push_back(move(p));
   }
   projections.push_back(move(options));
  }
  initialize_flow_fingerprints();
  projection_flow_fp.resize(projections.size());
  for(int ti=0;ti<(int)projections.size();ti++){
   projection_flow_fp[ti].resize(projections[ti].size());
   for(int oi=0;oi<(int)projections[ti].size();oi++)for(int c=0;c<9;c++){
    uint64_t h=0;for(int d=day+1;d<30;d++)h+=(uint64_t)(int64_t)projections[ti][oi].q[d][c]*flow_salt[d];projection_flow_fp[ti][oi][c]=h;
   }
  }
 }
 // Memoize each commodity's complete scenario trajectory. Keys contain the
 // fingerprints of the rounded flows; full-vector collision checking is not stored.
 // Summation over commodities/scenarios remains in the original order.
 struct FlowKey {uint64_t h=0;int after=0;};
 static constexpr int FLOW_CAP=16384,FLOW_TABLE=32768,FLOW_MASK=FLOW_TABLE-1;
 array<array<double,32>,9> flow_scratch{};
 array<unique_ptr<uint16_t[]>,9> flow_slot;
 array<unique_ptr<uint64_t[]>,9> flow_key_h;
 array<unique_ptr<int[]>,9> flow_key_after;
 array<uint16_t,9> flow_count{};
 array<uint64_t,9> qbase_fp{};array<uint8_t,9> qbase_valid{};int qbase_cache[9][30]{};
 array<unique_ptr<double[]>,9> flow_values;
 // The objective depends only on the projection and these inventories.
 // Route rearrangements often preserve all four inputs.
 bool score_cache_valid=false;uint64_t score_cache_revision=0;double score_cache_value=0.;
 array<int,9> score_cache_after{},score_cache_carry{},score_cache_holding{};
 double future_value(const Plan&p,const int*after,const int*carry){
  return future_value_uncached(p,after,carry);
 }
 // Work is measured in fifth-turns, including per-tile travel and input
 // allowances. 110 fifth-turns is 22 effective turns per worker. The main
 // farmer supplies the same capacity: do not add another worker on top of
 // work/capacity. Estimated capacity was checked against saved legal routes;
 // the actual current-day route feasibility and hiring costs are unchanged.
 double future_value_uncached(const Plan&p,const int*after,const int*carry){
  const Projection& proj=update_plan_projection(p);
  auto&day_cost=labour_day_cost;
  if(!labour_cache_valid){
  for(int d=day+1;d<30;d++){
   double workers=max(1.,proj.work[d]/110.);
   int lo=min(24,(int)workers);double frac=workers-lo; // workers>=1 so truncation==floor, skips a libm call
   day_cost[d]=-proj.labour[d];day_cost[d]-=wage_total[lo]*(1.-frac)+wage_total[lo+1]*frac;
  }
  labour_cache=0.;for(int d=day+1;d<30;d++)labour_cache+=future_df[d]*day_cost[d];labour_cache_valid=true;
  }
  double labour_value=labour_cache;
  if(score_cache_valid&&score_cache_revision==projection_revision&&score_cache_holding==p.holding&&memcmp(score_cache_after.data(),after,9*sizeof(int))==0&&memcmp(score_cache_carry.data(),carry,9*sizeof(int))==0)return score_cache_value;
   array<const double*,9> cached{};
   for(int c=0;c<9;c++){
    int held=clamp(p.holding[c],0,100);FlowKey k;k.after=after[c];k.h=aggregate_flow_fp[c]+holding_flow_fp[c][held]+(uint64_t)(int64_t)carry[c]*flow_salt[day+1];
    uint64_t mixed=k.h^uint64_t(uint32_t(k.after))*0x9e3779b97f4a7c15ULL;uint32_t slot=(uint32_t)mixed&FLOW_MASK;uint16_t idx;
    for(;;slot=(slot+1)&FLOW_MASK){idx=flow_slot[c][slot];if(idx==0xffff)break;if(flow_key_h[c][slot]==k.h&&flow_key_after[c][slot]==k.after){cached[c]=flow_values[c].get()+size_t(idx)*scenario_count;goto flow_done;}}
    {
    if(!qbase_valid[c]||qbase_fp[c]!=aggregate_flow_fp[c]){
     for(int d=day+1;d<30;d++)qbase_cache[c][d]=baseline_projection.q[d][c];
     for(int ti=0;ti<(int)tasks.size();ti++){uint8_t oi=aggregate_choices[ti];if(oi==255)continue;const auto&x=projections[ti][oi];for(int d=day+1;d<30;d++)qbase_cache[c][d]+=x.q[d][c];}
     qbase_fp[c]=aggregate_flow_fp[c];qbase_valid[c]=1;
    }
    int qday[30];int scheduled=0;for(int d=day+1;d<30;d++)scheduled+=hold_schedule[d][c];int prefix=0,assigned=0;
    for(int d=day+1;d<30;d++){int q=qbase_cache[c][d];if(d==day+1)q+=carry[c];if(!scheduled){if(d==day+1)q+=held;}else{prefix+=hold_schedule[d][c];int amount=held*prefix/scheduled;q+=amount-assigned;assigned=amount;}qday[d]=q;}
    double* path=flow_scratch[c].data();
    if(flow_count[c]<FLOW_CAP){idx=flow_count[c]++;flow_slot[c][slot]=idx;flow_key_h[c][slot]=k.h;flow_key_after[c][slot]=k.after;path=flow_values[c].get()+size_t(idx)*scenario_count;}
    for(int sc=0;sc<scenario_count;sc++){
     int inv=after[c];double value=0.;
     for(int d=day+1;d<30;d++){
      int demand=scenario_demand[sc][d][c];inv-=demand/2;
      value+=future_df[d]*trade_difference(c,inv,qday[d],(int)storage_opponent[d][c]);
      inv-=demand-demand/2;
     }
     path[sc]=value;
    }
    cached[c]=path;
    }
flow_done: ;
   }
  // Cached values already include the fixed discount factors. Only nine
  // numbers per scenario are added now, rather than nine numbers per day.
  // The objective remains mean minus one quarter of the standard deviation.
  double total=0.,values[32]{};
  for(int sc=0;sc<scenario_count;sc++){
   double v=labour_value;for(int c=0;c<9;c++)v+=cached[c][sc];
   values[sc]=v;total+=v;
  }
  double mean=total/scenario_count,variance=0.;
  for(int sc=0;sc<scenario_count;sc++){double z=values[sc]-mean;variance+=z*z;}
  double result=mean-.25*sqrt(variance/scenario_count);
  score_cache_revision=projection_revision;score_cache_holding=p.holding;copy(after,after+9,score_cache_after.begin());copy(carry,carry+9,score_cache_carry.begin());score_cache_value=result;score_cache_valid=true;
  return result;
 }

 // Marginal values of forecast supply under the same joint-market objective.
 // Unlike an average sale quote, these include inventory carried across dates
 // and the effect of market prices on the rival's forecast receipts.
 void continuation_prices(const Plan&p,vector<double>&out,array<double,30>&workprice){
  int after[9]{},carry[9]{};
  terminal_score_setup(p,preparation(p),after,carry);
  Projection proj=plan_projection(p,carry);
  out.resize((30-day)*9);
  for(int d=day;d<30;d++)for(int c=0;c<9;c++)out[(d-day)*9+c]=prices[d][c];
  double scenario_base[32]{};
  for(int sc=0;sc<scenario_count;sc++){
   int inv[9];copy(after,after+9,inv);
   for(int d=day+1;d<30;d++)for(int c=0;c<9;c++){
    int demand=scenario_demand[sc][d][c];inv[c]-=demand/2;
    scenario_base[sc]+=future_df[d]*trade_difference(c,inv[c],proj.q[d][c],int(storage_opponent[d][c]));
    inv[c]-=demand-demand/2;
   }
  }
  auto objective=[&](const double*v){
   double mean=0.,variance=0.;for(int sc=0;sc<scenario_count;sc++)mean+=v[sc];mean/=scenario_count;
   for(int sc=0;sc<scenario_count;sc++){double z=v[sc]-mean;variance+=z*z;}
   return mean-.25*sqrt(variance/scenario_count);
  };
  double baseline=objective(scenario_base);
  for(int c=0;c<9;c++){
   double base[32]{};
   auto path=[&](int sc,int changed){
    int inv=after[c];double v=0.;
    for(int d=day+1;d<30;d++){
     int demand=scenario_demand[sc][d][c];inv-=demand/2;
     v+=future_df[d]*trade_difference(c,inv,proj.q[d][c]+(d==changed),int(storage_opponent[d][c]));
     inv-=demand-demand/2;
    }
    return v;
   };
   for(int sc=0;sc<scenario_count;sc++)base[sc]=path(sc,-1);
   for(int d=day+1;d<30;d++){
    double altered[32]{};for(int sc=0;sc<scenario_count;sc++)altered[sc]=scenario_base[sc]+path(sc,d)-base[sc];
    double derivative=(objective(altered)-baseline)/future_df[d];
    // Underrelax each policy-improvement step. No preference for any crop,
    // animal, opponent identity or seed is encoded in this update.
    out[(d-day)*9+c]=.5*prices[d][c]+.5*derivative;
   }
  }
  for(int d=day+1;d<30;d++){
   double workers=max(1.,proj.work[d]/110.);int lo=min(24,(int)workers);
   double a=1.,b=1.;for(int i=1;i<lo;i++){double z=a+b;a=b;b=z;}
   workprice[d]=.5*(use_continuation_work_price?continuation_work_price[d]:unitcost/5.)+.5*a/110.;
  }
 }

 // Keep the original continuation for every current action. Add a second
 // continuation generated with marginal market and labour values, without a
 // second full Solver instance.
 static bool same_actions(const Acts&a,const Acts&b){if(a.size()!=b.size())return false;for(size_t j=0;j<a.size();j++)if(a[j].op!=b[j].op||a[j].item!=b[j].item||a[j].n!=b[j].n)return false;return true;}
 void import_continuations(const vector<double>&revised,const array<double,30>&workprice){
  if(day>=29)return;
  vector<Projection> current_base(tasks.size());for(int ti=0;ti<(int)tasks.size();ti++)current_base[ti]=project(refresh(tiles[tasks[ti].pos],day),day+1);
  double saved_prices[30][9];memcpy(saved_prices,prices,sizeof(prices));bool saved_use=use_continuation_work_price;array<double,30> saved_work{};copy(begin(continuation_work_price),end(continuation_work_price),saved_work.begin());auto saved_memo=move(memo);
  memo.clear();memo.reserve(300000);projection_memo.clear();use_continuation_work_price=true;copy(workprice.begin(),workprice.end(),continuation_work_price);for(int d=day;d<30;d++)for(int c=0;c<9;c++)prices[d][c]=revised[(d-day)*9+c];
  struct Pending{int parent;double value;Projection projection;};vector<vector<Pending>> pending(tasks.size());
  for(int ti=0;ti<(int)tasks.size();ti++){
   const auto&current=tasks[ti];Tile base_tile=tiles[current.pos];if(base_tile.kind==1)base_tile=Tile();auto raw=enumerate(base_tile,day);double base=day<29?time_discount(day)*value(refresh(base_tile,day),day+1):0;vector<Option> alt;
   for(auto&o:raw){if(o.acts.empty())continue;o.value=o.immediate+(day<29?time_discount(day)*value(refresh(o.next,day),day+1):0)-base;alt.push_back(move(o));}
   sort(alt.begin(),alt.end(),[](const auto&a,const auto&b){return a.value>b.value;});vector<Option> kept;vector<uint64_t> kept_keys;
   for(auto&o:alt){bool dominated=false;uint64_t ok=key(o.next,day);for(size_t qi=0;qi<kept.size();qi++){const auto&q=kept[qi];if(q.value>=o.value-1e-8&&q.acts.size()<=o.acts.size()&&q.w<=o.w&&q.f<=o.f&&q.seeds==o.seeds&&q.animals==o.animals&&q.goods==o.goods&&kept_keys[qi]==ok){dominated=true;break;}}if(!dominated){kept_keys.push_back(ok);kept.push_back(move(o));}}
   int original_count=current.opts.size();for(int oi=0;oi<original_count;oi++){const Option*matched=nullptr;for(const auto&o:kept)if(same_actions(o.acts,current.opts[oi].acts)){matched=&o;break;}if(!matched)continue;Projection p=project(refresh(matched->next,day),day+1);for(int d=day+1;d<30;d++){for(int c=0;c<9;c++)p.q[d][c]-=current_base[ti].q[d][c];p.labour[d]-=current_base[ti].labour[d];p.work[d]-=current_base[ti].work[d];}if(memcmp(&p,&projections[ti][oi],sizeof(Projection))==0)continue;pending[ti].push_back({oi,matched->value,move(p)});}
  }
  projection_memo.clear();memcpy(prices,saved_prices,sizeof(prices));use_continuation_work_price=saved_use;copy(saved_work.begin(),saved_work.end(),continuation_work_price);memo=move(saved_memo);
  for(int ti=0;ti<(int)tasks.size();ti++)for(auto&x:pending[ti]){Option copy=tasks[ti].opts[x.parent];copy.future_policy=1;copy.future_parent=x.parent;copy.value=x.value;if(tasks[ti].opts.size()>=255)throw std::runtime_error("too many continuation options");
   tasks[ti].opts.push_back(move(copy));
   array<uint64_t,9> fp{};
   for(int c=0;c<9;c++)for(int d=day+1;d<30;d++)fp[c]+=(uint64_t)(int64_t)x.projection.q[d][c]*flow_salt[d];
   projection_flow_fp[ti].push_back(fp);
   projections[ti].push_back(move(x.projection));}
 }
