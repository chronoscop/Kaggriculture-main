// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
// Replay the shared shed in worker order, then the ten market-order slots.
// Routes provide only sparse PICKUP/DROP events; no full game simulation is
// needed for every search proposal. In particular, opening stock remains in
// the shed until its sale and products awaiting an order slot remain overnight.
struct Delivery { array<uint8_t,9> item,qty;uint8_t count=0; };
// Quantities and item order are independent of timing when crops do not
// decay. The key may miss on different unused TaskList bytes, never falsely hit.
struct RouteDeliveryCache { TaskList tasks;int split=-999;bool tail=false;int count=0;Delivery drops[2]; };
mutable array<RouteDeliveryCache,U> delivery_cache;
template<bool Timed> double terminal_score_impl(const Plan&p,const Preparation&setup,int*after,int*carry,const PlanSummary*summary,const TerminalTiming* timing)const {
 if(!setup.valid)return -1e100;
 struct Event { int16_t next;uint8_t kind,key,n; } events[7*U];
 Delivery drops[2*U];
 int16_t head[25],tail[25];int ne=0,nd=0;uint32_t own_hours=uint32_t(1)<<hour;fill(head,head+25,int16_t(-1));
 int required[12]{};
 auto event=[&](int h,int kind,int key,int n) {
  events[ne]={-1,(uint8_t)kind,(uint8_t)key,(uint8_t)n};
  own_hours|=uint32_t(1)<<h;
  if(head[h]>=0)events[tail[h]].next=ne;else head[h]=ne;
  tail[h]=ne++;
 };
 // Appending in worker order preserves same-turn PICKUP/DROP interactions.
 for(int u=0;u<(int)p.routes.size();u++) {
  const auto&r=p.routes[u];
  for(int k=0;k<setup.preloaded[u];k++) {
   int c=setup.pickup_items[u][k],n=setup.pickup_quantities[u][k];required[c]+=n;
   event(setup.preload_hours[u][k],0,c,n);
  }
  if(r.tasks.empty())continue;
  auto&cache=delivery_cache[u];
  if constexpr(Timed){
   if(cache.split==r.split&&cache.tasks.size()==r.tasks.size()&&memcmp(cache.tasks.data,r.tasks.data,sizeof(r.tasks.data))==0){
    // Each task occurs once per plan, so its route offsets are monotone.
    // The route end and split delivery also bound every intermediate action.
    int h=setup.begin[u]+timing->route_off[u];if(h>endhour)return -1e100;
    if(r.split>0&&r.split<=(int)r.tasks.size()){
     int sh=setup.begin[u]+timing->split_drop[u];if(sh>=endhour)return -1e100;
     drops[nd]=cache.drops[0];event(sh,1,nd++,0);
    }
    if(cache.tail){
     if(r.overnight&&day<29)h=24;
     else{h+=timing->tail_dist[u];if(h>=endhour)return -1e100;}
     drops[nd]=cache.drops[cache.count-1];event(h,1,nd++,0);
    }
    continue;
   }
  }
  int delivery_begin=nd;bool has_decay=false,tail_present=false;
  int h=setup.begin[u],at=setup.starts[u],j=0;
  auto inputs=summary?summary->inputs[u]:route_inputs(r);
  array<int,9> bag{},order{};int norder=0;
  auto add=[&](int c,int n){if(n<=0)return;if(bag[c]==0)order[norder++]=c;bag[c]+=n;};
  auto take=[&](int c){if(--bag[c]==0){int k=0;while(k<norder&&order[k]!=c)k++;for(;k+1<norder;k++)order[k]=order[k+1];if(k<norder)norder--;}};
  auto drop=[&](){auto&d=drops[nd];d.count=norder;for(int k=0;k<norder;k++){int c=order[k];d.item[k]=c;d.qty[k]=bag[c];}event(h,1,nd++,0);bag.fill(0);norder=0;};
  add(0,inputs.w);add(8,inputs.f);
  for(auto[ti,oi]:r.tasks) {
   const auto&t=tasks[ti];const auto&o=t.opts[oi];const auto&tt=tiles[t.pos];
   if constexpr(Timed)h=setup.begin[u]+timing->task_start[ti];else{h+=dist(at,t.pos);at=t.pos;}
   has_decay|=tt.kind==3&&tt.decay>=0&&tt.decay<=day*24;
   int na=(int)o.duration;if(h+na>endhour)return -1e100;
   for(int ei=0;ei<o.effect_count;ei++){int eh=h+o.effect_step[ei],op=o.effect_op[ei],item=o.effect_item[ei];
    if(op==FEED)take(0);
    else if(op==FERTILIZE)take(8);
    else if(op==COLLECT_FERTILIZER)add(8,1);
    else if(op==HARVEST){int quantity=o.goods[tt.item];if(tt.kind==3&&tt.decay>=0&&tt.decay<=day*24){if(tt.y-(eh+1)/2<=0)return -1e100;quantity=max(0,quantity-(eh+1)/2);}add(tt.item,quantity);}
   }
   h+=na;
   if(++j==r.split){if constexpr(Timed)h=setup.begin[u]+timing->split_drop[u];else{h+=dshed(at);at=shed(at);}if(h>=endhour)return -1e100;drop();h++;}
  }
  if constexpr(Timed)h=setup.begin[u]+timing->route_off[u];
  tail_present=accumulate(bag.begin(),bag.end(),0)>0;
  if(tail_present) {
   if(r.overnight&&day<29){if(h>endhour)return -1e100;h=24;}
   else {if constexpr(Timed)h+=timing->tail_dist[u];else h+=dshed(at);if(h>=endhour)return -1e100;}
   drop();
  } else if(h>endhour)return -1e100;
  if constexpr(Timed){if(!has_decay){cache.tasks=r.tasks;cache.split=r.split;cache.tail=tail_present;cache.count=nd-delivery_begin;for(int k=0;k<cache.count;k++)cache.drops[k]=drops[delivery_begin+k];}}
 }
 int current[12],inv[9];copy(stock.begin(),stock.end(),current);copy(mi,mi+9,inv);
 int occupied=accumulate(current,current+12,0);
 double self=-cost(p.routes.size())-(summary?(summary->land?landprice:0):land_cost(p)),other=0;
 // Opening-stock order exposure is deliberately a conservative forecast.
 for(int c=0;c<9;c++) {
  int rival=opponent[hour][c];if(!rival)continue;
  int immediate=setup.early_sales[c],later=setup.deferred[c]+p.holding[c]+setup.late_sales[c];
  if(!later)continue;
  auto advantage=[&](int qty) {
   if(qty>=0&&qty<=100)return opening_exposure[c][qty];
   int early=mi[c],late=mi[c];
   long long first=sell_value(c,early,qty);first-=sell_value(c,early,rival);
   long long last=-sell_value(c,late,rival);last+=sell_value(c,late,qty);
   return first-last;
  };
  self-=max(0LL,advantage(immediate+later)-advantage(immediate));
 }
 auto farm_events=[&](int h) {
  for(int e=head[h];e>=0;e=events[e].next) {
   const auto&ev=events[e];
   if(ev.kind==0) {
    if(current[ev.key]<ev.n)return false;
    current[ev.key]-=ev.n;occupied-=ev.n;required[ev.key]-=ev.n;
   } else {
    const auto&d=drops[ev.key];
    for(int k=0;k<d.count;k++){int c=d.item[k],n=min<int>(d.qty[k],max(0,100-occupied));current[c]+=n;occupied+=n;}
   }
  }
  return true;
 };

 for(int c=0;c<9;c++)if(setup.deferred[c])own_hours|=uint32_t(1)<<setup.release[c];
 for(int h=hour;h<hour+(int(setup.orders.size())+9)/10;h++)own_hours|=uint32_t(1)<<h;
 int previous_hour=hour;
 auto advance_town=[&](int h){
  int ticks=(h+3)/4-(previous_hour+3)/4;
  bool center=previous_hour==0&&h>0;
  if(ticks||center)for(int c=0;c<9;c++)inv[c]-=ticks*shop_demand[c]+int(center&&c<8);
  previous_hour=h;
 };
 uint32_t unprocessed=((uint32_t(1)<<endhour)-1)&~((uint32_t(1)<<hour)-1);
 while(uint32_t pending=(own_hours|opponent_hours)&unprocessed) {
  int h=__builtin_ctz(pending);unprocessed&=~((uint32_t(1)<<(h+1))-1);
  advance_town(h);
  unsigned traded=opponent_items[h];int sold[9]{},bought[9]{};
  if(own_hours&(uint32_t(1)<<h)){
  if(!farm_events(h))return -1e100;
  int reserve[9],view[9];
  for(int c=0;c<9;c++){reserve[c]=p.holding[c]+required[c]+(h<setup.release[c]?setup.deferred[c]:0);view[c]=current[c];}
  int offset=(h-hour)*10,stop=min(offset+10,int(setup.orders.size())),slots=10-max(0,stop-offset);
  for(int j=offset;j<stop;j++) {
   const auto&o=setup.orders[j];int c=o.item;
   if(o.op==0) {
    int n=min<int>(o.n,max(0,view[c]-reserve[c]));view[c]-=n;current[c]-=n;occupied-=n;sold[c]+=n;traded|=1u<<c;
   } else if(o.op==2||o.op==3) {
    // A partial BUY would make a later PICKUP/PLACE fail. Reject that plan.
    if(occupied+o.n>100)return -1e100;
    current[c]+=o.n;occupied+=o.n;if(o.op==2){bought[c]+=o.n;traded|=1u<<c;}own_hours|=uint32_t(1)<<(h+1);
   }
  }
  if(slots) {
   int rank[9],quantity[9],count=0;double priority[9];
   for(int c=0;c<9;c++){quantity[c]=max(0,view[c]-reserve[c]);if(quantity[c])rank[count++]=c;}
   if(count>slots){for(int i=0;i<count;i++){int c=rank[i];priority[c]=sale_priority(c,inv[c],quantity[c],opponent[h][c]);}}
   if(count>slots)for(int i=1;i<count;i++){int x=rank[i],j=i;while(j>0&&priority[rank[j-1]]<priority[x]){rank[j]=rank[j-1];j--;}rank[j]=x;}
   if(count>slots)own_hours|=uint32_t(1)<<(h+1);
   for(int k=0;k<min(slots,count);k++){int c=rank[k],n=quantity[c];current[c]-=n;occupied-=n;sold[c]+=n;traded|=1u<<c;}
  }
  else { own_hours|=uint32_t(1)<<(h+1); }
  }
  // Retain the existing public opponent-flow model and simultaneous unitwise
  // quotes, but feed it the sales that can actually leave our shed this turn.
  // CTZ keeps the commodity summation order while skipping inactive items.
  for(unsigned bits=traded;bits;bits&=bits-1) {
   int c=__builtin_ctz(bits),a=sold[c],b=opponent[h][c],buy=bought[c];
   if(buy){int simultaneous=min(b,buy);self+=trade_difference(c,inv[c],-buy,simultaneous);b-=simultaneous;}
   if(a||b)self+=trade_difference(c,inv[c],a,b);
  }
 }
 advance_town(endhour);
 if(!farm_events(24))return -1e100;
 if(after)copy(inv,inv+9,after);
 if(carry)for(int c=0;c<9;c++)carry[c]=max(0,current[c]-p.holding[c]);
 return self-other;
}

inline double terminal_score_setup(const Plan&p,const Preparation&setup,int*after=nullptr,int*carry=nullptr,const PlanSummary*summary=nullptr)const {return terminal_score_impl<false>(p,setup,after,carry,summary,nullptr);}
inline double terminal_score_timed(const Plan&p,const Preparation&setup,const TerminalTiming&timing,int*after=nullptr,int*carry=nullptr,const PlanSummary*summary=nullptr)const {return terminal_score_impl<true>(p,setup,after,carry,summary,&timing);}
