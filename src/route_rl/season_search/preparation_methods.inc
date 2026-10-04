// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
// Type: 0 SELL, 1 BUY_SEED, 2 BUY_PRODUCT, 3 BUY_ANIMAL, 4 HIRE, 5 BUY_LAND.
struct MarketOrder { uint8_t op,item,n;MarketOrder()=default;MarketOrder(int o,int i,int q):op(o),item(i),n(q){} };
template<int CAP> struct FixedOrders {
 array<MarketOrder,CAP> data; int count=0;
 inline int size()const{return count;}
 inline void push_back(MarketOrder x){data[count++]=x;}
 inline MarketOrder& operator[](int i){return data[i];}
 inline const MarketOrder& operator[](int i)const{return data[i];}
 template<int C2> inline void append(const FixedOrders<C2>&o){for(int i=0;i<o.size();i++)push_back(o[i]);}
};
struct TerminalTiming {array<uint16_t,100> task_start;array<uint16_t,U> route_off,split_drop,tail_dist;};
struct Preparation {
 FixedOrders<48> orders;
 int ready=0,land_turn=-1;
 bool valid=true;double purchase_cost=0.;
 array<uint8_t,9> deferred{},release{},origin{},early_sales{},late_sales{};
 array<uint8_t,U> preloaded{};
 array<uint8_t,U> begin,unit_turn,starts;
 array<array<uint8_t,5>,U> preload_hours,pickup_items,pickup_quantities;
};
void preparation_uncached_into(const Plan&p,const PlanSummary&s,Preparation&out,TerminalTiming* timing=nullptr)const {
 int need[12]{},ns[5]{};
 const auto&inputs=s.inputs;int nr=s.nr;need[0]=s.w;need[8]=s.f;for(int c=0;c<3;c++)need[9+c]=s.animals[c];for(int c=0;c<5;c++)ns[c]=s.seeds[c];
 int base_need_total=accumulate(need,need+12,0);
 for(int c=0;c<9;c++)need[c]+=p.holding[c];
 double conservative=cost(nr)+(s.land?landprice:0);
 for(int c=0;c<9;c++){int missing=need[c]-stock[c];if(missing)conservative+=missing*price(c,mi[c]-missing);}
 for(int c=0;c<3;c++)conservative+=animalcost[c]*max(0,need[9+c]-stock[9+c]);
 for(int c=0;c<5;c++)conservative+=seedcost[c]*max(0,ns[c]-seeds[c]);
 if(base_need_total+accumulate(holds,holds+9,0)>100)conservative=1e100;out.purchase_cost=conservative;
 FixedOrders<2> products;FixedOrders<3> animals;FixedOrders<5> seed_buys;
 double cash=cost(nr)+(s.land?landprice:0);
 int room=accumulate(stock.begin(),stock.end(),0)-100;
 for(int c:{0,8}) {
  int n=max(0,need[c]-stock[c]);if(!n)continue;
  products.push_back({2,c,n});int inv=mi[c];cash+=buy_value(c,inv,n);room+=n;
 }
 for(int c=0;c<3;c++) {
  int n=max(0,need[9+c]-stock[9+c]);
  if(n){animals.push_back({3,9+c,n});cash+=animalcost[c]*n;room+=n;}
 }
 for(int c=0;c<5;c++) {
  int n=max(0,ns[c]-seeds[c]);if(n){seed_buys.push_back({1,c,n});cash+=n*seedcost[c];}
 }
 cash-=money;
 int surplus[9],rank[9],nrank=0;double priority[9];
 for(int c=0;c<9;c++){surplus[c]=max(0,stock[c]-need[c]);if(surplus[c]){rank[nrank++]=c;priority[c]=opening_exposure[c][surplus[c]]+1e-6*surplus[c]*table[c][3000];}}
 for(int i=1;i<nrank;i++){int x=rank[i],j=i;while(j>0&&priority[rank[j-1]]<priority[x]){rank[j]=rank[j-1];j--;}rank[j]=x;}
 for(int ri=0;ri<nrank;ri++){int c=rank[ri];if(!(surplus[c]&&(p.sell_first||room>0||cash>0)))continue;
  int n=surplus[c],inv=mi[c];out.orders.push_back({0,c,n});room-=n;cash-=sell_value(c,inv,n);surplus[c]=0;
 }
 auto hires=[&](){for(int n=existing;n<nr;n++)out.orders.push_back({4,0,1});};
 auto land=[&](){if(s.land)out.orders.push_back({5,0,1});};
 auto goods=[&](){out.orders.append(products);out.orders.append(animals);};
 auto sowing=[&](){out.orders.append(seed_buys);};
 if(p.preparation_mode==1){land();goods();sowing();hires();}
 else if(p.preparation_mode==2){goods();hires();land();sowing();}
 else {hires();land();goods();sowing();}
 int last=-1;
 for(int j=0;j<(int)out.orders.size();j++)if(out.orders[j].op!=0)last=j;
 out.ready=hour+(last+10)/10;
 int deferred_room=max(0,40-accumulate(p.holding.begin(),p.holding.end(),0));
 for(int ri=0;ri<nrank;ri++){int c=rank[ri];if(!surplus[c])continue;
  int earliest=hour+int(out.orders.size())/10;
  int release=clamp(p.sale_after[c],0,endhour-1);
  if(release>max(earliest,out.ready)) {
   int n=min(surplus[c],deferred_room);deferred_room-=n;
   out.deferred[c]=n;out.release[c]=release;out.origin[c]=earliest;surplus[c]-=n;
  }
  if(surplus[c])out.orders.push_back({0,c,surplus[c]});
 }
 for(int j=0;j<out.orders.size();j++){const auto&o=out.orders[j];if(o.op==0){if(j<10)out.early_sales[o.item]+=o.n;else out.late_sales[o.item]+=o.n;}}
 // Quantities available before each farm phase, after earlier reservations.
 const int input_item[5]={0,8,9,10,11};
 int input_event_time[5];for(int q=0;q<5;q++)input_event_time[q]=-1;
 int input_event_delta[5]{};
 int reserved_total[5]{},reserved_before_event[5]{};
 int seed_ready[5];for(int c=0;c<5;c++)seed_ready[c]=(ns[c]>seeds[c]?25:-1);
 int early_plants[5]{};
 out.unit_turn.fill(hour);
 int hired=existing;
 for(int j=0;j<(int)out.orders.size();j++) {
  const auto&o=out.orders[j];int h=hour+j/10;
  if(h>=endhour){out.valid=false;return;}
  if(o.op==4)out.unit_turn[hired++]=h+1;
  else if(o.op==5)out.land_turn=h;
  else if(o.op==1){seed_ready[o.item]=h+1;}
  else {
   int q=o.item==0?0:o.item==8?1:o.item==9?2:o.item==10?3:o.item==11?4:-1;
   if(q>=0){input_event_time[q]=h+1;input_event_delta[q]=o.op==0?-int(o.n):int(o.n);}
  }
 }
 const int access[4]={44,45,54,55};
 int spawn_t=-1,spawn_counts[4]{};
 for(int u=0;u<nr;u++) {
  const auto&r=p.routes[u];
  // HIREs are resolved in actual order. Hires sharing one market turn see
  // the same positions for all already-active workers; only the newly chosen
  // spawn tile changes between those HIRE orders.
  if(u<existing)out.starts[u]=positions[u];
  else {
   int t=out.unit_turn[u]-1;
   if(t!=spawn_t){spawn_t=t;fill(spawn_counts,spawn_counts+4,0);for(int v=0;v<u;v++){int pos=position_at(p.routes[v],out.begin[v],t,out.starts[v]);for(int k=0;k<4;k++)spawn_counts[k]+=pos==access[k];}}
   int best=0;for(int k=1;k<4;k++)if(spawn_counts[k]<spawn_counts[best])best=k;
   out.starts[u]=access[best];spawn_counts[best]++;
  }
  int t=out.unit_turn[u];if(!p.early_pickup)t=max(t,out.ready);
  auto pickup=[&](int q,int c,int n) {
   if(!n)return;
   if(dshed(out.starts[u])){out.valid=false;return;}
   int et=input_event_time[q],final_available=stock[c]+input_event_delta[q]-reserved_total[q];
   if(final_available<n){out.valid=false;return;}
   if(et>=0&&input_event_delta[q]>0&&t<et&&stock[c]-reserved_before_event[q]<n)t=et;
   if(t>=endhour){out.valid=false;return;}
   int k=out.preloaded[u]++;
   out.preload_hours[u][k]=t;out.pickup_items[u][k]=c;out.pickup_quantities[u][k]=n;
   reserved_total[q]+=n;if(et>=0&&t<et)reserved_before_event[q]+=n;
   t++;
  };
  pickup(0,0,inputs[u].w);pickup(1,8,inputs[u].f);
  for(int c=0;c<3;c++)pickup(2+c,9+c,inputs[u].animals[c]);
  if(!out.valid)return;
  int begin=t,off=0,at=out.starts[u],j=0;
  uint16_t plants[100];int nplants=0;
  for(auto[ti,oi]:r.tasks) {
   const auto&o=tasks[ti].opts[oi];off+=dist(at,tasks[ti].pos);at=tasks[ti].pos;if(timing)timing->task_start[ti]=(uint16_t)off;
   if(tiles[at].kind==1)begin=max(begin,out.land_turn+1-off);
   if(o.plant_item>=0)plants[nplants++]=uint16_t(((off+o.plant_step)<<3)|o.plant_item);off+=(int)o.duration;
   if(++j==r.split){if(timing)timing->split_drop[u]=(uint16_t)(off+dshed(at));off+=dshed(at)+1;at=shed(at);}
  }
  if(timing){timing->route_off[u]=(uint16_t)off;timing->tail_dist[u]=(uint16_t)dshed(at);}
  // Seed purchases are one bulk order per crop. After that order becomes
  // available, total stock is sufficient for all planned plants. Therefore
  // only plants before seed_ready[c] compete for the opening seed stock.
  // A seed deficit cannot improve before its bulk purchase arrives. Jump
  // directly to that start time, while preserving the old route-end cutoff.
  if(nplants)for(;;) {
   int next_begin=begin+1;int trial[5]{};bool enough=true;
   for(int k=0;k<nplants;k++){
    int packed=plants[k],offset=packed>>3,c=packed&7;int h=begin+offset;
    if(h>=endhour){enough=false;break;}
    if(seed_ready[c]>=0&&h<seed_ready[c]&&early_plants[c]+(++trial[c])>seeds[c]){enough=false;next_begin=max(next_begin,seed_ready[c]-offset);break;}
   }
   if(enough){for(int c=0;c<5;c++)early_plants[c]+=trial[c];break;}
   if(begin+off>=endhour||next_begin+off>endhour){out.valid=false;return;}
   begin=next_begin;
  }
  out.begin[u]=begin;
 }
 return;
}

// The variable-length cache key cost exceeds recomputing this schedule.
Preparation preparation(const Plan&p,const PlanSummary&s)const { Preparation out;preparation_uncached_into(p,s,out);return out; }
Preparation preparation(const Plan&p)const { auto s=summarize_plan(p);Preparation out;preparation_uncached_into(p,s,out);return out; }
