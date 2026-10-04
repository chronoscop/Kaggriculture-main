// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
 // Incremental multi-start search with cheap local moves and occasional LNS.
 // A slice ends between proposals; no partially evaluated plan is exported.
 Plan search_best,search_current;
 PlanSummary current_summary{};array<RouteSummary,U> current_route_summaries{};bool current_summary_valid=false;
 void rebuild_current_summary(){current_summary=summarize_plan(search_current,&current_route_summaries);current_summary_valid=true;}
 struct InitialPlan{Plan plan;bool shortlist=false;int preferred=-1;};
 vector<InitialPlan> initial_plans;
 bool strategy_starts_added=false;
 size_t next_initial=0;
 array<int,9> proposed_holds{};
 int minworkers=1,maxworkers=18;
 double liquid_upper=0;
 long long withdrawal_trials=0,withdrawal_wins=0;
 vector<int> all_tasks;
 vector<int> full_order;
 vector<pair<double,pair<int,int>>> full_choices;
 vector<pair<int,int>> full_rank;
 vector<char> full_remove;
 struct Elite{array<int,48> key{};Plan plan;};
 vector<Elite> elites;
 double epoch_temperature=10.,current_temperature=10.;
 array<int,48> production_signature(const Plan&p)const{
  array<int,48> key{};key[0]=p.routes.size()/2;key[1]=p.allow_land;
  for(int c=0;c<9;c++)key[2+c]=p.holding[c]/10;
  for(const auto&r:p.routes)for(auto[ti,oi]:r.tasks){const auto&o=tasks[ti].opts[oi];
   for(int c=0;c<5;c++)key[11+c]+=o.seeds[c];
   for(int c=0;c<3;c++)key[16+c]+=o.animals[c];
   for(int c=0;c<9;c++)key[19+c]+=o.goods[c];
   for(const auto&a:o.acts)key[28+a.op]++;
  }
  // Coarse production niches prevent one-unit variations from occupying
  // the whole archive. Routes with the same niche compete on exact value.
  for(int c=0;c<5;c++)key[11+c]/=4;
  for(int c=0;c<3;c++)key[16+c]/=2;
  for(int c=0;c<9;c++)key[19+c]/=20;
  for(int op=0;op<18;op++)key[28+op]/=8;
  return key;
 }
 void remember_elite(const Plan&p){
  if(p.score<-1e50||p.score<search_best.score-1000.)return;
  auto key=production_signature(p);
  for(auto&e:elites)if(e.key==key){
   if(p.score>e.plan.score+1e-8 || (abs(p.score-e.plan.score)<=1e-8&&route_work(p)<route_work(e.plan)))e.plan=p;
   return;
  }
  if(elites.size()<12){elites.push_back({key,p});return;}
  auto worst=min_element(elites.begin(),elites.end(),[](const Elite&a,const Elite&b){return a.plan.score<b.plan.score;});
  if(p.score>worst->plan.score)*worst={key,p};
 }
 int route_work(const Plan&p)const{int n=0;for(const auto&r:p.routes)n+=route_length(r);return n;}
 void optimize_sale_time(Plan&p,int c){
  if(stock[c]<=p.holding[c])return;
  Plan best=p;
  for(int h:{0,1,5,9,13,17,21}){
   p.sale_after[c]=h;evaluate(p);
   if(p.score>best.score+1e-8)best=p;
  }
  p=move(best);
 }
 void begin_search(){
  tiles_animals=0;for(const auto&t:tiles)if(t.kind==6)tiles_animals++;
  compute_holds();make_tasks();make_projections();
  copy(holds,holds+9,proposed_holds.begin());
  all_tasks.resize(tasks.size());iota(all_tasks.begin(),all_tasks.end(),0);
  minworkers=max(1,existing);maxworkers=min(U,max(existing,18));
  liquid_upper=money;for(int c=0;c<9;c++)liquid_upper+=stock[c]*price(c,mi[c]);
  vector<int> counts;
  for(int k:{4,8,12,16,18,1}){k=clamp(k,minworkers,maxworkers);if(find(counts.begin(),counts.end(),k)==counts.end())counts.push_back(k);}
  for(int mode=0;mode<2;mode++)for(int hm=0;hm<(accumulate(proposed_holds.begin(),proposed_holds.end(),0)>0?2:1);hm++){
   if(mode&&(!landprice||day>=23))continue;
   for(int k:counts){
    if(cost(k)>liquid_upper)continue;
    Plan p;p.allow_land=mode;if(!hm)p.holding=proposed_holds;
    for(int pos:starts(k)){p.routes.push_back({pos,{}});p.routes.back().overnight=day<29;}
    // Retain both the compact shortlist construction and the broader joint
    // construction. A locally superior insertion can lead to a worse whole
    // allocation, so the broader constructor must not erase the old starts.
    initial_plans.push_back({p,true});initial_plans.push_back({move(p),false});
   }
  }
  for(int pos:starts(minworkers))search_best.routes.push_back({pos,{}});
  evaluate(search_best);search_current=search_best;
 }
 void consider(Plan p,int op=-1,bool archive=true){
  if(p.score>search_best.score+1e-8 || (abs(p.score-search_best.score)<=1e-8 && route_work(p)<route_work(search_best))){search_best=p;}
  if(archive)remember_elite(p);
  // Moderate worsening lets coordinated production changes cross small valleys.
  uniform_real_distribution<double> ur(0,1);
  if(p.score>search_current.score||ur(rng)<exp(min(0.,(p.score-search_current.score)/current_temperature)))search_current=move(p);
 }

 vector<int> absent(const Plan&p){
  vector<char> used(tasks.size());for(auto&r:p.routes)for(auto [t,o]:r.tasks)used[t]=true;
  vector<int> out;for(int t=0;t<(int)tasks.size();t++)if(!used[t])out.push_back(t);return out;
 }
 bool withdraw_expansion(Plan&p,bool shrink){
  if(!land_cost(p))return false;
  withdrawal_trials++;double previous=p.score;
  p.allow_land=false;
  for(auto&r:p.routes){
   r.tasks.erase(remove_if(r.tasks.begin(),r.tasks.end(),[&](auto v){return tiles[tasks[v.first].pos].kind==1;}),r.tasks.end());
   r.split=-1;
  }
  // Reconstruct the whole purchased quadrant and its labour together.
  // Merely flipping allow_land only prevented new insertions; it could
  // leave every already-selected locked-tile job (and the purchase) intact.
  if(shrink){
   int n=max(minworkers,int(p.routes.size())-1-int(rng()%3));
   p.routes.resize(n);
  }
  auto order=absent(p);repair(p,order,.4);refine(p);
  withdrawal_wins+=p.score>previous+1e-8;
  return p.score>-1e50;
 }
 bool try_withdrawal(){return withdrawal_trials<8+16*withdrawal_wins || routing_rng()%8==0;}
 void random_extras(vector<int>& order,const Plan&p,int count){
  auto available=absent(p);shuffle(available.begin(),available.end(),rng);
  for(int t:available){if(count<=0)break;if(find(order.begin(),order.end(),t)==order.end()){order.push_back(t);count--;}}
 }
 // Held-Karp shortest path for the selected jobs of one short route.
 // Production options are fixed. The full evaluator checks input reuse and
 // sale timing before any reordered route is accepted by the search.
 bool exact_route_order(Route&r){
  int n=r.tasks.size();if(n<3||n>12||r.split>=0)return false;
  int total=1<<n;
  vector<short> dp(total*n,30000);vector<signed char> parent(total*n,-1);
  int pos[12],travel[12][12];for(int j=0;j<n;j++)pos[j]=tasks[r.tasks[j].first].pos;
  for(int j=0;j<n;j++)for(int k=0;k<n;k++)travel[j][k]=dist(pos[j],pos[k]);
  for(int j=0;j<n;j++)dp[(1<<j)*n+j]=dist(r.start,pos[j]);
  for(int mask=1;mask<total;mask++){
   for(int bits=mask;bits;bits&=bits-1){
    int j=__builtin_ctz((unsigned)bits),base=dp[mask*n+j];
    if(base>=30000)continue;
    for(int rest=(total-1)^mask;rest;rest&=rest-1){
     int k=__builtin_ctz((unsigned)rest),next=(mask|(1<<k))*n+k,v=base+travel[j][k];
     if(v<dp[next]){dp[next]=v;parent[next]=j;}
    }
   }
  }
  auto old=r.tasks,best_order=old;int best_length=route_length(r);
  auto inputs=route_inputs(r);int best_inputs=inputs.w*price(0,mi[0])+inputs.f*price(8,mi[8]);
  // Consider every endpoint. Actual length includes the required PICKUPs
  // and a return only when goods remain, so empty-handed routes stay open.
  for(int end=0;end<n;end++){
   int mask=total-1,last=end;
   for(int k=n-1;k>=0;k--){r.tasks[k]=old[last];int prev=parent[mask*n+last];mask^=1<<last;last=prev;}
   int length=route_length(r);auto need=route_inputs(r);
   int input_cost=need.w*price(0,mi[0])+need.f*price(8,mi[8]);
   if(length<best_length||(length==best_length&&input_cost<best_inputs)){
    best_length=length;best_inputs=input_cost;best_order=r.tasks;
   }
  }
  r.tasks=move(best_order);return r.tasks!=old;
 }
 // Rebuild the entire assignment while preserving every selected production
 // option. Ordinary ruin/repair can change production before a more compact
 // routing of that same production is found, especially when removing a hand.
 bool repack_routes(Plan&p,int workers){
  vector<pair<int,int>> jobs;
  for(const auto&r:p.routes)for(auto v:r.tasks)jobs.push_back(v);
  if(jobs.empty())return false;
  uniform_real_distribution<double> ur(0.,1.);
  vector<pair<double,pair<int,int>>> ranked;
  for(auto v:jobs){const auto&t=tasks[v.first];const auto&o=t.opts[v.second];
   double difficulty=1.5*dshed(t.pos)+o.acts.size()+2*(o.w>0||o.f>0);
   ranked.push_back({-difficulty*(.65+.7*ur(routing_rng)),v});
  }
  sort(ranked.begin(),ranked.end());
  Plan rebuilt=p;rebuilt.routes.clear();
  bool overnight=day<29&&(routing_rng()%4!=0);
  for(int start:starts(workers)){rebuilt.routes.push_back({start,{}});rebuilt.routes.back().overnight=overnight;}
  int capacity=endhour-hour;
  if(day==29)capacity=endhour-hour;
  for(auto [rank,v]:ranked){
   int best_route=-1,best_slot=-1;double best=1e100;
   for(int ri=0;ri<workers;ri++){
    auto&r=rebuilt.routes[ri];int oldlen=route_length(r);
    for(int k=0;k<=(int)r.tasks.size();k++){
     r.tasks.insert(r.tasks.begin()+k,v);int len=route_length(r);
     if(len<=capacity){
      double score=len-oldlen-.08*len;
      if(score<best){best=score;best_route=ri;best_slot=k;}
     }
     r.tasks.erase(r.tasks.begin()+k);
    }
   }
   if(best_route<0){
    // A feasible route can be hidden behind a suboptimal insertion order.
    // Reorder its fixed jobs exactly before rejecting worker compression.
    Route chosen;int shortest=100000;
    for(int ri=0;ri<workers;ri++){
     auto trial=rebuilt.routes[ri];trial.tasks.push_back(v);
     exact_route_order(trial);int len=route_length(trial);
     if(len<=capacity&&len<shortest){shortest=len;chosen=move(trial);best_route=ri;}
    }
    if(best_route<0)return false;
    rebuilt.routes[best_route]=move(chosen);continue;
   }
   auto&r=rebuilt.routes[best_route];r.tasks.insert(r.tasks.begin()+best_slot,v);
  }
  evaluate(rebuilt);if(rebuilt.score<-1e50)return false;
  refine(rebuilt);p=move(rebuilt);return true;
 }
 int local_batch;
 int restart_period;
 void start_iteration(){
  iterations++;
  if(iterations%restart_period==1){
   epoch_temperature=10.;search_current=search_best;current_summary_valid=false;
   if(elites.size()>1&&routing_rng()%3==0){
    vector<int> choices;for(int i=0;i<(int)elites.size();i++)if(elites[i].plan.score>=search_best.score-1000.)choices.push_back(i);
    if(!choices.empty()){
     search_current=elites[choices[routing_rng()%choices.size()]].plan;current_summary_valid=false;
     epoch_temperature=clamp(.35*(search_best.score-search_current.score),10.,150.);
    }
   }
  }
  current_temperature=10.+(epoch_temperature-10.)*(restart_period-1-((iterations-1)%restart_period))/double(restart_period-1);
 }
 void full_search_step(){
  start_iteration();current_summary_valid=false;
  Plan p=search_current;
  int draw=rng()%100,op=draw<20?0:draw<38?1:draw<48?2:draw<63?3:draw<82?4:draw<88?5:draw<94?6:7;

  int ri=rng()%p.routes.size();auto&r=p.routes[ri];
  auto& order=full_order;order.clear();
  if(op==0){
   if(r.tasks.empty())return;
   int at=rng()%r.tasks.size();auto&v=r.tasks[at];
   if(rng()%2==0)v.second=rng()%tasks[v.first].opts.size();
   else{
    int old=v.second,best=old;double score=p.score;int bestwork=route_work(p);
    for(int oi=0;oi<(int)tasks[v.first].opts.size();oi++){
     v.second=oi;double trial=evaluate(p);
     if(trial>score+1e-8 || (abs(trial-score)<=1e-8 && route_work(p)<bestwork)){
      score=trial;best=oi;bestwork=route_work(p);
     }
    }
    v.second=best;p.score=score;
   }
  }else if(op==1){
   if(r.tasks.empty())return;
   int at=rng()%r.tasks.size();auto v=r.tasks[at];r.tasks.erase(r.tasks.begin()+at);r.split=-1;
   if(rng()%4==0){auto&dest=p.routes[rng()%p.routes.size()];dest.split=-1;dest.tasks.insert(dest.tasks.begin()+rng()%(dest.tasks.size()+1),v);}
   else{
    Plan best=p;double bv=-1e100;int bw=100000;
    auto& choices=full_choices;choices.clear();
    for(int target=0;target<(int)p.routes.size();target++){
     auto&dest=p.routes[target];
     int split=dest.split;dest.split=-1;
     int oldlen=route_length(dest);auto oldinputs=route_inputs(dest);
     double best_cost=1e100;int place=-1;
     for(int k=0;k<=(int)dest.tasks.size();k++){
      dest.tasks.insert(dest.tasks.begin()+k,v);
      int len=route_length(dest);
      if(len<=endhour-hour+2){
       auto inputs=route_inputs(dest);
       double cost=.15*(len-oldlen)+(inputs.w-oldinputs.w)*price(0,mi[0])+(inputs.f-oldinputs.f)*price(8,mi[8]);
       if(cost<best_cost){best_cost=cost;place=k;}
      }
      dest.tasks.erase(dest.tasks.begin()+k);
     }
     dest.split=split;
     if(place>=0)choices.push_back({best_cost,{target,place}});
    }
    sort(choices.begin(),choices.end());
    // Most overnight route permutations have identical market cash flow.
    // Use exact travel/input checks to select a few distinct destinations
    // before paying for the whole-farm market evaluation.
    for(int i=0;i<min(8,int(choices.size()));i++){
     auto[target,place]=choices[i].second;auto&dest=p.routes[target];int split=dest.split;dest.split=-1;
     dest.tasks.insert(dest.tasks.begin()+place,v);
     double sc=evaluate(p);int work=route_work(p);
     if(sc>bv+1e-8 || (abs(sc-bv)<=1e-8 && work<bw)){bv=sc;bw=work;best=p;}
     dest.tasks.erase(dest.tasks.begin()+place);dest.split=split;
    }
    p=move(best);
   }
  }else if(op==2){
   if(r.tasks.empty())return;
   auto&dest=p.routes[rng()%p.routes.size()];if(dest.tasks.empty())return;
   swap(r.tasks[rng()%r.tasks.size()],dest.tasks[rng()%dest.tasks.size()]);r.split=dest.split=-1;
  }else if(op==3){
   if(!r.tasks.empty()&&rng()%2){int at=rng()%r.tasks.size();order.push_back(r.tasks[at].first);r.tasks.erase(r.tasks.begin()+at);r.split=-1;}
   random_extras(order,p,2);repair(p,order,.35);
  }else if(op==4||op==5){
   if(op==5&&rng()%6==0){
    // Rebuild capacity and production together. A larger farm can be
    // worthwhile only with several extra hands; single-hand hill climbing
    // can reject each intermediate plan before that combination is reached.
    int workers=clamp(int(p.routes.size())+int(rng()%7)-3,minworkers,maxworkers);
    p.routes.clear();for(int pos:starts(workers)){p.routes.push_back({pos,{}});p.routes.back().overnight=day<29;}
    p.allow_land=landprice&&day<23&&rng()%2;
    p.holding=rng()%2?proposed_holds:array<int,9>{};
   }
   int center=tasks.empty()?44:tasks[rng()%tasks.size()].pos;
   auto& rank=full_rank;rank.clear();
   for(auto&route:p.routes)for(auto [t,o]:route.tasks)rank.push_back({op==4?dist(center,tasks[t].pos)*4+int(rng()%7):int(rng()%1000),t});
   sort(rank.begin(),rank.end());int count=op==4?2+rng()%4:max(1,int(rank.size()*(.15+.30*(rng()%100)/100.)));
   auto& remove=full_remove;remove.assign(tasks.size(),0);for(int k=0;k<min(count,(int)rank.size());k++){remove[rank[k].second]=true;order.push_back(rank[k].second);}
   for(auto&route:p.routes){route.tasks.erase(remove_if(route.tasks.begin(),route.tasks.end(),[&](auto v){return remove[v.first];}),route.tasks.end());route.split=-1;}
   random_extras(order,p,op==4?4:int(tasks.size()));
   // Product-specific reconstruction can change several crop/animal counts
   // together while preserving ordinary maintenance choices. All eight
   // types are sampled, and the common objective decides whether to keep it.
   int preferred=op==5 && rng()%2==0?int(rng()%8):-1;
   repair(p,order,op==4?.65:1.4,preferred);
  }else if(op==6){
   int k=p.routes.size();
   int jump=1+int(rng()%3);
   if(rng()%2&&k<maxworkers){
    int n=min(maxworkers,k+jump);if(cost(n)<liquid_upper){auto ss=starts(n);for(int j=k;j<n;j++){p.routes.push_back({ss[j],{}});p.routes.back().overnight=day<29;}}
   }else if(k>minworkers){
    int n=max(minworkers,k-jump);while((int)p.routes.size()>n){for(auto [t,o]:p.routes.back().tasks)order.push_back(t);p.routes.pop_back();}
   }
   random_extras(order,p,8);repair(p,order,.5);
  }else{
   switch(rng()%6){
    case 0:p.sell_first=!p.sell_first;break;
    case 1:if(day<29)r.overnight=!r.overnight;break;
    case 2:r.split=r.tasks.empty()?-1:int(rng()%(r.tasks.size()+1))-1;break;
    case 3:if(day<29){
     int c=rng()%9,room=max(0,40-accumulate(p.holding.begin(),p.holding.end(),0)+p.holding[c]);
     int choice=rng()%3;p.holding[c]=choice==0?0:choice==1?min(room,proposed_holds[c]):min(room,stock[c]);
    }break;
    case 4:
     if(land_cost(p)){if(try_withdrawal())withdraw_expansion(p,rng()%2);}
     else{p.allow_land=!p.allow_land;random_extras(order,p,6);repair(p,order,.5);}
     break;
    case 5:{int c=rng()%9;int hours[]={0,1,5,9,13,17,21};p.sale_after[c]=hours[rng()%7];}break;
   }
  }
  evaluate(p);
  if(p.score>search_best.score&&iterations%4==0)refine(p);
  consider(move(p),op);
  if(iterations%restart_period==0){
   Plan compact=search_best;int route=(iterations/restart_period-1)%compact.routes.size();
   if(exact_route_order(compact.routes[route])){
    evaluate(compact);
    if(compact.score>=search_best.score-1e-8){
     auto available=absent(compact);repair(compact,available,0.);refine(compact);consider(move(compact),2);
    }
   }
   Plan timed=search_best;optimize_sale_time(timed,(iterations/restart_period-1)%9);
   consider(move(timed),7);search_current=search_best;
  }
  if(iterations%(2*restart_period)==0){
   // An occasional production-preserving intensification has a separate
   // random stream and keeps the ordinary cheap-neighbourhood schedule.
   int n=search_best.routes.size();
   for(int workers:{max(minworkers,n-1),n}){
    Plan compact=search_best;
    if(repack_routes(compact,workers) &&
       (compact.score>search_best.score+1e-8 ||
        (abs(compact.score-search_best.score)<=1e-8 && route_work(compact)<route_work(search_best)))){
     search_best=compact;search_current=move(compact);
    }
   }
   if(land_cost(search_best)&&try_withdrawal()){
    Plan compact=search_best;
    if(withdraw_expansion(compact,true))consider(move(compact),7);
   }
  }
 }

 // One candidate per local proposal. Heavy production/capacity reconstruction
 // remains in full_search_step and is interleaved at a separately set rate.
 void local_search_step(){
  start_iteration();if(!current_summary_valid)rebuild_current_summary();
  Plan&p=search_current;PlanSummary local_summary=current_summary;RouteSummary local_z1{},local_z2{};bool have_z1=false,have_z2=false;
  int ri=rng()%p.routes.size();auto&r=p.routes[ri];
  int dirty1=-1,dirty2=-1;Route backup1,backup2;int backup_i1=-1,backup_i2=-1;
  bool old_early=p.early_pickup,old_sell=p.sell_first;int old_mode=p.preparation_mode;
  double old_score=p.score;int old_ready=p.ready;auto old_preloaded=p.preloaded;
  auto backup_route=[&](int idx){
   if(idx==backup_i1||idx==backup_i2)return;
   if(backup_i1<0){backup_i1=idx;backup1=p.routes[idx];}
   else{backup_i2=idx;backup2=p.routes[idx];}
  };
  auto rollback=[&](){
   if(backup_i1>=0)p.routes[backup_i1]=move(backup1);
   if(backup_i2>=0)p.routes[backup_i2]=move(backup2);
   p.early_pickup=old_early;p.sell_first=old_sell;p.preparation_mode=old_mode;p.score=old_score;p.ready=old_ready;p.preloaded=old_preloaded;
  };
  int draw=rng()%100,op=0;
  if(draw<28){
   if(r.tasks.empty())return;backup_route(ri);
   auto&v=r.tasks[rng()%r.tasks.size()];int count=tasks[v.first].opts.size();
   if(count<2){rollback();return;}int oi=rng()%(count-1);if(oi>=v.second)oi++;v.second=oi;dirty1=ri;
  }else if(draw<48){
   op=1;if(r.tasks.empty())return;backup_route(ri);
   int at=rng()%r.tasks.size();auto job=r.tasks[at];r.tasks.erase(r.tasks.begin()+at);r.split=-1;
   int dest=rng()%p.routes.size();backup_route(dest);auto&rr=p.routes[dest];rr.split=-1;dirty1=ri;dirty2=dest;
   int slot=rng()%(rr.tasks.size()+1);
   if(rng()%2==0){int best=100000;for(int k=0;k<=(int)rr.tasks.size();k++){rr.tasks.insert(rr.tasks.begin()+k,job);int length=route_length(rr);if(length<best){best=length;slot=k;}rr.tasks.erase(rr.tasks.begin()+k);}}
   rr.tasks.insert(rr.tasks.begin()+slot,job);if(dest==ri&&at==slot){rollback();return;}
  }else if(draw<63){
   op=2;if(r.tasks.empty())return;int rri=rng()%p.routes.size();auto&rr=p.routes[rri];if(rr.tasks.empty())return;backup_route(ri);backup_route(rri);dirty1=ri;dirty2=rri;
   int i=rng()%r.tasks.size(),j=rng()%rr.tasks.size();if(&r==&rr&&i==j){rollback();return;}swap(r.tasks[i],rr.tasks[j]);r.split=rr.split=-1;
  }else if(draw<79){
   op=3;backup_route(ri);
   if(!r.tasks.empty()&&rng()%2){r.tasks.erase(r.tasks.begin()+rng()%r.tasks.size());r.split=-1;}
   auto available=absent(p);if(available.empty()){rollback();return;}
   int ti=available[rng()%available.size()],n=tasks[ti].opts.size();if(tiles[tasks[ti].pos].kind==1&&!p.allow_land){rollback();return;}
   int oi=rng()%3==0?rng()%n:rng()%min(n,6);r.tasks.insert(r.tasks.begin()+rng()%(r.tasks.size()+1),{ti,oi});r.split=-1;dirty1=ri;
  }else if(draw<89){
   op=3;if(r.tasks.empty())return;backup_route(ri);r.tasks.erase(r.tasks.begin()+rng()%r.tasks.size());r.split=-1;dirty1=ri;
  }else if(draw<96){
   op=2;if(r.tasks.size()<2)return;backup_route(ri);int a=rng()%r.tasks.size(),b=rng()%r.tasks.size();if(a==b){rollback();return;}if(a>b)swap(a,b);reverse(r.tasks.begin()+a,r.tasks.begin()+b+1);r.split=-1;dirty1=ri;
  }else{
   op=7;switch(rng()%4){
    case 0:if(day<29){backup_route(ri);r.overnight=!r.overnight;dirty1=ri;}else return;break;
    case 1:p.early_pickup=!p.early_pickup;break;
    case 2:p.sell_first=!p.sell_first;break;
    default:p.preparation_mode=(p.preparation_mode+1+rng()%2)%3;
   }
  }

  if(dirty1>=0&&route_length(p.routes[dirty1])>endhour-hour+2){rollback();return;}
  if(dirty2>=0&&dirty2!=dirty1&&route_length(p.routes[dirty2])>endhour-hour+2){rollback();return;}
  auto refresh=[&](int idx,bool second){if(idx<0)return;auto z=summarize_route(p.routes[idx]);add_route_summary(local_summary,current_route_summaries[idx],-1,idx);add_route_summary(local_summary,z,1,idx);if(second){local_z2=z;have_z2=true;}else{local_z1=z;have_z1=true;}};
  refresh(dirty1,false);if(dirty2!=dirty1)refresh(dirty2,true);
  auto ev=evaluate_plan_summary(p,local_summary);if(ev.score<-1e50){rollback();return;}
  bool better=ev.score>search_best.score+1e-8;
  bool archive=better||iterations%64==0;
  uniform_real_distribution<double> ur(0,1);bool accept=ev.score>old_score||ur(rng)<exp(min(0.,(ev.score-old_score)/current_temperature));
  bool tie=abs(ev.score-search_best.score)<=1e-8;
  // Only materialize derived starts/begins for plans that can survive this proposal.
  if(accept||archive||tie){
   array<int,U> old_starts{},old_begins{};for(int u=0;u<(int)p.routes.size();u++){old_starts[u]=p.routes[u].start;old_begins[u]=p.routes[u].begin;}
   apply_evaluation(p,ev);
   if(p.score>search_best.score+1e-8 || (abs(p.score-search_best.score)<=1e-8 && route_work(p)<route_work(search_best))){search_best=p;}
   if(archive)remember_elite(p);
   if(accept){current_summary=local_summary;if(have_z1)current_route_summaries[dirty1]=local_z1;if(have_z2)current_route_summaries[dirty2]=local_z2;current_summary_valid=true;return;}
   for(int u=0;u<(int)p.routes.size();u++){p.routes[u].start=old_starts[u];p.routes[u].begin=old_begins[u];}
  }
  rollback();
 }
 void search_step(){
  if(!local_batch||(iterations+1)%(local_batch+1)==0)full_search_step();
  else local_search_step();
 }
 void advance_search(double seconds){
  auto start=chrono::steady_clock::now();
  const int limit=100000000;
  auto call_elapsed=[&](){return chrono::duration<double>(chrono::steady_clock::now()-start).count();};
  auto time_left=[&](){return call_elapsed()<seconds;};
  while(next_initial<initial_plans.size()&&time_left()){
   auto entry=initial_plans[next_initial++];Plan p=move(entry.plan);repair(p,all_tasks,0,entry.preferred,entry.shortlist);refine(p);consider(move(p));
   if(next_initial==initial_plans.size()&&!strategy_starts_added){
    strategy_starts_added=true;
    // Densify only around the strongest worker count, instead of every count.
    int best_n=search_best.routes.size();
    for(int k:{best_n-1,best_n+1})if(k>=minworkers&&k<=maxworkers&&cost(k)<=liquid_upper){
     Plan q=search_best;q.routes.clear();for(int pos:starts(k)){q.routes.push_back({pos,{}});q.routes.back().overnight=day<29;}
     repair(q,all_tasks,0);refine(q);consider(move(q));
    }
    // Construct each of the eight investment families from an empty daily
    // assignment. Existing production can still be maintained and harvested.
    // This exposes coordinated allocations that small repair neighbourhoods
    // cannot reach through individually profitable one-tile changes.
    if(day<29){
     for(int preferred=0;preferred<8;preferred++){
      Plan q=search_best;q.routes.clear();
      for(int pos:starts(best_n)){q.routes.push_back({pos,{}});q.routes.back().overnight=true;}
      initial_plans.push_back({move(q),false,preferred});
     }
    }
    search_current=search_best;
   }
  }
  while(next_initial==initial_plans.size()&&iterations<limit&&time_left())search_step();
 }
 void import_warm(const Solver&old){
  if(old.day!=day)return;
  Plan p=old.search_best;p.routes.clear();vector<char> used(tasks.size());
  int workers=clamp((int)old.search_best.routes.size(),minworkers,maxworkers);if(workers>(int)old.search_best.routes.size())return;auto ss=starts(workers);
  for(int j=0;j<workers;j++){
   Route r{ss[j],{}};r.overnight=old.search_best.routes[j].overnight;
   int seen=0;
   for(auto [ti,oi]:old.search_best.routes[j].tasks){
    int pos=old.tasks[ti].pos;const auto&acts=old.tasks[ti].opts[oi].acts;
    for(int t=0;t<(int)tasks.size();t++)if(tasks[t].pos==pos&&!used[t]){
     int matched=-1;
     for(int o=0;o<(int)tasks[t].opts.size();o++){
      const auto&a=tasks[t].opts[o].acts;if(a.size()!=acts.size())continue;
      bool same=true;for(size_t k=0;k<a.size();k++)if(a[k].op!=acts[k].op||a[k].item!=acts[k].item||a[k].n!=acts[k].n)same=false;
      if(same){if(matched<0)matched=o;if(tasks[t].opts[o].future_policy==old.tasks[ti].opts[oi].future_policy){matched=o;break;}}
     }
     if(matched>=0){r.tasks.push_back({t,matched});used[t]=true;}
     break;
    }
    if(++seen==old.search_best.routes[j].split)r.split=r.tasks.size();
   }
   p.routes.push_back(move(r));
  }
  for(int c=0;c<9;c++)p.holding[c]=min(p.holding[c],stock[c]);
  evaluate(p);
  // Actual money, weeds and preparation length can invalidate a prediction.
  // Drop the longest route's last job until all real-state checks succeed.
  while(p.score<-1e50){
   auto it=max_element(p.routes.begin(),p.routes.end(),[&](const Route&a,const Route&b){return route_length(a)<route_length(b);});
   if(it!=p.routes.end()&&!it->tasks.empty())it->tasks.pop_back();
   else if((int)p.routes.size()>minworkers)p.routes.pop_back();
   else break;
   evaluate(p);
  }
  consider(move(p));
 }
 double import_seed(const Solver&old){
  // The seed search now uses the same Solver implementation with an independent
  // random stream. No action-signature translation or duplicate solver source.
  Plan p=old.search_best;p.early_pickup=false;double score=evaluate(p);consider(p);
  p.early_pickup=true;double early=evaluate(p);consider(move(p));return max(score,early);
 }
