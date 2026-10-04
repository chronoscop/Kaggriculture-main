// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
// Multi-day tile dynamic programming, route LNS, and inventory holding search.
// Extended and corrected from the supplied final-day search prototype.
// No hidden opponent inventory, random seed, or future shop sequence is accepted.
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
using namespace std;
namespace kg {
// Expected win + half draw under explicit, uncalibrated logistic margin noise.
// Cash is a forecast input; no cash reward is supplied to PPO.
double expected_terminal_score(double margin,double scale){
 scale=max(1.,scale);
 auto cdf=[&](double threshold){double z=clamp((threshold-margin)/scale,-60.,60.);return 1./(1.+exp(-z));};
 return clamp(1.-.5*(cdf(.5)+cdf(-.5)),0.,1.);
}
constexpr int DAYS=30,N=100,U=20;
enum Op {PASS,NORTH,SOUTH,EAST,WEST,PICKUP,DROP,PLACE,PLANT,WATER,HARVEST,FERTILIZE,BUILD_COOP,BUILD_PASTURE,DIG,FEED,COLLECT_FERTILIZER,CARE};
struct Act{uint8_t op=PASS,item=0,n=0;Act()=default;Act(int o,int i=0,int q=0):op(o),item(i),n(q){}};
// Generated tile options contain at most a few actions. Keep them with the
// option so constructing and copying options needs no separate allocation.
struct Acts{
 Act data[8]{};uint8_t count=0;
 void push_back(Act a){if(count>=8)throw runtime_error("too many tile actions");data[count++]=a;}
 size_t size()const{return count;}bool empty()const{return count==0;}
 Act& operator[](size_t i){return data[i];}const Act& operator[](size_t i)const{return data[i];}
 Act* begin(){return data;}Act* end(){return data+count;}
 const Act* begin()const{return data;}const Act* end()const{return data+count;}
};
// kind 0 empty, 1 locked, 2 weed, 3 crop, 4 empty coop, 5 empty pasture, 6 animal
struct Tile {int16_t kind=0,item=0,age=0,y=0,dry=0,water=0,fert=-1,care=0,fa=0,pending=0,decay=-1;};
struct Option{Acts acts;Tile next;double immediate=0,value=0,investment=0;int w=0,f=0,future_policy=0,future_parent=-1;int need_w=0,need_f=0,net_w=0,net_f=0,route_class=0,goods_total=0,net_goods=0,seeds_total=0,animals_total=0;int8_t plant_item=-1,plant_step=-1;uint8_t effect_count=0,duration=0;array<uint8_t,4> effect_step{},effect_op{},effect_item{};array<int,5> seeds{};array<int,3> animals{};array<int,9> goods{};};
struct Task{int pos;vector<Option> opts;};
struct TaskRef{uint8_t first=0,second=0;TaskRef()=default;TaskRef(int a,int b):first((uint8_t)a),second((uint8_t)b){}TaskRef(const pair<int,int>&p):TaskRef(p.first,p.second){}operator pair<int,int>()const{return{first,second};}bool operator==(const TaskRef&o)const{return first==o.first&&second==o.second;}bool operator!=(const TaskRef&o)const{return!(*this==o);}};
// A route has at most 24 working hours. Allow extra temporary insertions while
// avoiding allocations in local search and route copies.
struct TaskList {
 static constexpr size_t capacity=32;
 TaskRef data[capacity]{};uint8_t count=0;
 size_t size()const{return count;}bool empty()const{return count==0;}
 TaskRef* begin(){return data;}TaskRef* end(){return data+count;}
 const TaskRef* begin()const{return data;}const TaskRef* end()const{return data+count;}
 TaskRef& operator[](size_t i){return data[i];}const TaskRef& operator[](size_t i)const{return data[i];}
 TaskRef& back(){return data[count-1];}const TaskRef& back()const{return data[count-1];}
 void clear(){count=0;}
 void push_back(TaskRef v){if(count>=capacity)throw runtime_error("too many route tasks");data[count++]=v;}
 void pop_back(){--count;}
 TaskRef* insert(TaskRef* pos,TaskRef v){size_t i=pos-data;if(count>=capacity)throw runtime_error("too many route tasks");
  for(size_t k=count;k>i;k--)data[k]=data[k-1];data[i]=v;++count;return data+i;}
 TaskRef* erase(TaskRef* pos){return erase(pos,pos+1);}
 TaskRef* erase(TaskRef* first,TaskRef* last){size_t i=first-data,n=last-first;
  for(size_t k=i;k+n<count;k++)data[k]=data[k+n];count-=n;return data+i;}
 void resize(size_t n){if(n>capacity)throw runtime_error("too many route tasks");count=n;}
 bool operator==(const TaskList&o)const{return count==o.count&&equal(begin(),end(),o.begin());}
 bool operator!=(const TaskList&o)const{return !(*this==o);}
};
struct Route{int start=44;TaskList tasks;int split=-1;bool overnight=false;int begin=-1;};
struct Plan{int ready=0;bool sell_first=false;vector<Route> routes;double score=-1e100;bool allow_land=false;array<int,9> holding{},sale_after{};bool early_pickup=true;array<int,U> preloaded{};int preparation_mode=0;};
int first[8]={2,2,8,10,10,4,8,6};
int maxage[5]={4,3,8,10,12};
int maxyield[8]={6,4,4,4,6,4,6,6};
int interval_[8]={0,0,1,2,0,1,2,3};
int seedcost[5]={10,20,50,100,80};
int animalcost[3]={300,400,500};
inline bool ongoing(int c){return c==2||c==3;}
struct GeometryTables {
 array<array<unsigned char,100>,100> d{};array<unsigned char,100> sh{},ds{};
 GeometryTables(){for(int a=0;a<100;a++){int x=a%10,y=a/10;sh[a]=(x<=4?4:5)+10*(y<=4?4:5);for(int b=0;b<100;b++)d[a][b]=abs(x-b%10)+abs(y-b/10);}for(int a=0;a<100;a++)ds[a]=d[a][sh[a]];}
};
static const GeometryTables geom;
inline int dist(int a,int b){return geom.d[a][b];}
inline int shed(int a){return geom.sh[a];}
inline int dshed(int a){return geom.ds[a];}

struct Solver{
 // Commodity masks for sparse market replay, in the original item order.
 array<uint16_t,24> opponent_items{};
 int day,hour,remaining,endhour,existing;double money,unitcost=2.5,visitcost=2.5;
 double opponent_money=0.,terminal_uncertainty=2000.;
 double prices[30][9]{};double future_df[30]{},wage_total[26]{};
 double continuation_work_price[30]{};bool use_continuation_work_price=false;
 double future_action_cost(const Option&o,int d)const{
  if(use_continuation_work_price && d>day)
   return continuation_work_price[d]*(5*o.acts.size()+(o.acts.empty()?0:7)+(o.w>0)+(o.f>0));
  return unitcost*o.acts.size()+(o.acts.empty()?0:visitcost);
 }

 array<int,12> stock{};array<int,5> seeds{};array<int,3> animals{};array<int,9> goods{};vector<int> positions;vector<array<int,12>> inventory;
 vector<Tile> tiles;vector<Task> tasks;unordered_map<uint64_t,double> memo;
 mt19937 rng,routing_rng;
 int iterations=0;int scenario_demand[32][30][9]{},scenario_count=8;
 bool investments=true;int landprice=0;int tiles_animals=0;
 long long cumulative[9][6002]{}; int floor_inventory[9]{};
 long long opening_exposure[9][101]{};
 uint32_t opponent_hours=0;int mi[9]{};int opponent[24][9]{};int table[9][6001]{};
 Solver(const int* in,const double* fc,const Solver*cached=nullptr){
  const int* p=in;day=*p++;local_batch=day<6?15:7;restart_period=day<6?8192:512;hour=*p++;existing=*p++;money=*p++;unitcost=(*p++)/1000.;visitcost=(*p++)/1000.;int rseed=*p++;rng.seed(rseed);routing_rng.seed(rseed^0x47c9b63aU);
  remaining=30-day;endhour=day==29?23:24;
  {double df=1.;for(int d=day+1;d<30;d++){int q=d-1;df*=q<8?.90:(q<14?.97:1.);future_df[d]=df;}}
  {double a=1,b=1,total=0;wage_total[1]=0;for(int n=2;n<=25;n++){total+=a;wage_total[n]=total;double z=a+b;a=b;b=z;}}
  for(int i=0;i<12;i++)stock[i]=*p++;
  for(int i=0;i<5;i++)seeds[i]=*p++;
  for(int i=0;i<existing;i++)positions.push_back(*p++);
  for(int i=0;i<existing;i++){array<int,12> a;for(int j=0;j<12;j++)a[j]=*p++;inventory.push_back(a);}
  for(int z=0;z<100;z++){Tile t;t.kind=*p++;t.item=*p++;t.age=*p++;t.y=*p++;t.dry=*p++;t.water=*p++;t.fert=*p++;t.care=*p++;t.fa=*p++;t.pending=*p++;t.decay=*p++;tiles.push_back(t);}
  for(int d=day;d<30;d++)for(int j=0;j<9;j++)prices[d][j]=fc[(d-day)*9+j];
  for(int j=0;j<9;j++)mi[j]=*p++;
  for(int j=0;j<9;j++)shop_demand[j]=*p++;
  for(int h=0;h<24;h++)for(int j=0;j<9;j++){opponent[h][j]=*p++;if(opponent[h][j]){opponent_hours|=uint32_t(1)<<h;opponent_items[h]|=uint16_t(1)<<j;}}
  // The seed solver receives the same input and uses the same immutable
  // market lookup tables as the main solver.
  if(cached){
   memcpy(table,cached->table,sizeof(table));
   memcpy(cumulative,cached->cumulative,sizeof(cumulative));
   memcpy(floor_inventory,cached->floor_inventory,sizeof(floor_inventory));
   memcpy(opening_exposure,cached->opening_exposure,sizeof(opening_exposure));
  }else{
  for(int j=0;j<9;j++)for(int k=0;k<=6000;k++)table[j][k]=market_price(j,mi[j]+k-3000);
  for(int c=0;c<9;c++){
   floor_inventory[c]=2147483647;
   for(int k=0;k<=6000;k++){
    cumulative[c][k+1]=cumulative[c][k]+table[c][k];
    if(table[c][k]==1 && floor_inventory[c]==2147483647)floor_inventory[c]=mi[c]+k-3000;
   }
  }
  // Existing stock and the rival's opening-stock estimate are constant
  // during a search slice. Cache the exact order ambiguity for all feasible
  // shed quantities; both sale sorting and holding evaluation reuse it.
  for(int c=0;c<9;c++)for(int q=0;q<=100;q++){
   int early=mi[c],late=mi[c],rival=opponent[hour][c];
   long long first=sell_value(c,early,q);first-=sell_value(c,early,rival);
   long long last=-sell_value(c,late,rival);last+=sell_value(c,late,q);
   opening_exposure[c][q]=first-last;
  }
  }
  for(int d=day;d<30;d++)for(int c=0;c<9;c++)storage_opponent[d][c]=*p++;
  for(int d=day;d<30;d++)for(int c=0;c<9;c++)storage_flow[d][c]=*p++;
  int observed_shops=clamp(*p++,0,8);
  opponent_money=*p++;terminal_uncertainty=max(1.,double(*p++));
  scenario_count=observed_shops==8?1:observed_shops>=6?8:observed_shops>=4?16:32;
  for(int c=0;c<9;c++){flow_slot[c]=unique_ptr<uint16_t[]>(new uint16_t[FLOW_TABLE]);memset(flow_slot[c].get(),0xff,FLOW_TABLE*sizeof(uint16_t));flow_key_h[c]=unique_ptr<uint64_t[]>(new uint64_t[FLOW_TABLE]);flow_key_after[c]=unique_ptr<int[]>(new int[FLOW_TABLE]);flow_values[c]=unique_ptr<double[]>(new double[size_t(FLOW_CAP)*scenario_count]);}

  mt19937 climate_rng(rseed^0x73a581deU);
  const int shop_daily[8][9]={{6,0,0,0,0,6,0,0,0},{6,0,0,6,0,6,0,0,0},{6,6,6,6,0,0,0,0,0},{6,0,0,6,0,0,6,0,0},{0,12,0,0,0,0,0,0,0},{6,0,6,0,0,0,6,0,0},{0,0,0,6,0,0,6,0,0},{0,0,0,0,0,0,0,12,0}};
  for(int sc=0;sc<scenario_count;sc++)for(int d=day+1;d<30;d++)for(int c=0;c<9;c++)scenario_demand[sc][d][c]=6*shop_demand[c]+(c<8);
  // Speculative dawns may precede the observation of a newly unlocked shop.
  // Sample every unobserved slot, including one whose unlock date is today.
  for(int slot=observed_shops;slot<8;slot++){
   for(int block=0;block<scenario_count;block+=8){
    array<int,8> order; iota(order.begin(),order.end(),0);shuffle(order.begin(),order.end(),climate_rng);
    for(int k=0;k<8&&block+k<scenario_count;k++)for(int d=max(day+1,3*(slot+1));d<30;d++)for(int c=0;c<9;c++)scenario_demand[block+k][d][c]+=shop_daily[order[k]][c];
   }
  }
  memo.reserve(300000);
 }

 int market_price(int item,int inv)const{
  const double base[]={25,35,60,120,250,50,160,200,100};
  const double scale[]={400,450,200,100,300,332,122,105,200};
  const double below[]={.8,1.,.4,.7,.2,.4,.6,.2,.4};
  const double above[]={.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};
  double x=abs(inv-10000),t=scale[item];int shape=0;
  // Match Python's arithmetic order, not just the algebraic expression:
  // base +/- (target * base / f(T)) * f(x), then ties-to-even rounding.
  // Rearranging to base*(1-target*f(x)/f(T)) changes some half-dollar cases.
  if(inv<10000){
   if(item==0||item==3||item==6)shape=1;
   else if(item==1||item==2||item==5)shape=2;
   else if(item==4||item==7)shape=3;
  }else{
   if(item==0||item==5)shape=3;
   else if(item==1||item==2)shape=1;
   else if(item==4||item==7)shape=4;
  }
  auto f=[&](double y){
   if(shape==1)return sqrt(y);
   if(shape==2){double u=y/t;double z=max(0.,u-1.);return u+8.*(z*z);}
   if(shape==3)return log(1.+y);
   if(shape==4)return y*y;
   return y;
  };
  double amp=(inv<10000?below[item]:above[item])*base[item]/f(t);
  double p=inv<10000?base[item]+amp*f(x):base[item]-amp*f(x);
  return max(1,(int)nearbyint(p));
 }
 int price(int item,int inv)const{int idx=inv-mi[item]+3000;return idx>=0&&idx<=6000?table[item][idx]:market_price(item,inv);}
 // Exact aggregation of this evaluator's unit-lockstep market model.
 // Equal same-direction flows cancel from the cash difference, but their
 // inventory movement, including the $1 floor, must still be applied.
 long long sell_value(int c,int&inv,int n)const{
  if(n<=0)return 0;
  int k=inv-mi[c]+3000;
  if(k>=0 && k<=6000 && n<=6000-k){
   int sold=min(n,max(0,floor_inventory[c]-inv));
   long long v=cumulative[c][k+sold]-cumulative[c][k]+n-sold;
   inv+=sold;return v;
  }
  long long v=0;while(n--){int p=price(c,inv);v+=p;if(p>1)inv++;}return v;
 }
 long long buy_value(int c,int&inv,int n)const{
  if(n<=0)return 0;
  int k=inv-mi[c]+3000;
  if(k>=n&&k<=6001){long long v=cumulative[c][k]-cumulative[c][k-n];inv-=n;return v;}
  long long v=0;while(n--){inv--;v+=price(c,inv);}return v;
 }
 long long trade_difference(int c,int&inv,int own,int opp)const{
  long long v=0;
  if(own>0&&opp>0){
   int n=min(own,opp),k=inv-mi[c]+3000;
   if(k>=0&&k+2*n<=6000){
    int active=floor_inventory[c]<=inv?0:min(n,(floor_inventory[c]-inv+1)/2);
    inv+=2*active;
   }else{for(int j=0;j<n;j++)if(price(c,inv)>1)inv+=2;}
   own-=n;opp-=n;
  }else if(own<0&&opp<0){int n=min(-own,-opp);inv-=2*n;own+=n;opp+=n;}
  // Opposite-direction trades quote simultaneously, not sequentially.
  while(own&&opp){
   int ps=price(c,inv),pb=price(c,inv-1),delta=0;
   // Opposite orders cancel their inventory movement above the price floor.
   // Every matched unit therefore sees the same two quotes.
   if(ps>1){int n=min(abs(own),abs(opp)),sign=own>0?1:-1;v+=(long long)sign*n*((long long)ps+pb);own-=sign*n;opp+=sign*n;continue;}
   if(own>0){v+=ps;if(ps>1)delta++;own--;}else{v-=pb;delta--;own++;}
   if(opp>0){v-=ps;if(ps>1)delta++;opp--;}else{v+=pb;delta--;opp++;}
   inv+=delta;
  }
  if(own>0)v+=sell_value(c,inv,own);else if(own<0)v-=buy_value(c,inv,-own);
  if(opp>0)v-=sell_value(c,inv,opp);else if(opp<0)v+=buy_value(c,inv,-opp);
  return v;
 }
 double sale_priority(int c,int inv,int own,int opp)const{
  if(inv==mi[c]&&opp==opponent[hour][c]&&own>=0&&own<=100)return opening_exposure[c][own]+1e-6*own*price(c,inv);
  int early_inv=inv,late_inv=inv;
  double early=sell_value(c,early_inv,own);early-=sell_value(c,early_inv,opp);
  double late=-sell_value(c,late_inv,opp);late+=sell_value(c,late_inv,own);
  // Total money-difference exposure to trading first versus last. The tiny
  // tie-breaker prefers larger cash releases when no rival stock is inferred.
  return early-late+1e-6*own*price(c,inv);
 }
 struct Inputs {int w=0,f=0;array<int,3> animals{};};
 struct RouteSummary{Inputs input{};array<int,3> animals{};array<int,5> seeds{};int locked=0;bool invalid_split=false;};
 struct PlanSummary{array<Inputs,U> inputs{};int nr=0,w=0,f=0,locked=0,invalid_split_routes=0;array<int,3> animals{};array<int,5> seeds{};bool land=false,invest=false,invalid_split=false;};
 struct RepairScratch {
  vector<pair<double,int>> rank;
  vector<int> c_len,c_goods,c_w,c_f,c_start;
  vector<array<int,3>> c_anim;
  vector<char> c_hasdecay,c_valid;
  vector<Inputs> route_need;
  vector<double> choice_score;
  vector<int> choice_route,choice_position;
  // Prefix deficits and suffix resources are valid until this route changes.
  array<array<array<int,TaskList::capacity+1>,8>,U> route_work{};array<int,U> route_travel{};
  vector<int> sc_nwDW,sc_nwDF,sc_nwMW,sc_nwMF;
 } repair_scratch;
 RouteSummary summarize_route(const Route&r)const{
  RouteSummary z;int w=0,f=0,k=0;
  for(auto [ti,oi]:r.tasks){const auto&o=tasks[ti].opts[oi];
   for(int c=0;c<3;c++){z.input.animals[c]+=o.animals[c];z.animals[c]+=o.animals[c];}
   for(int c=0;c<5;c++)z.seeds[c]+=o.seeds[c];
   z.input.w=max(z.input.w,o.need_w-w);z.input.f=max(z.input.f,o.need_f-f);w+=o.net_w;f+=o.net_f;
   z.locked+=tiles[tasks[ti].pos].kind==1;
   if(r.split>0&&k>=r.split&&(o.w||o.f||o.animals[0]||o.animals[1]||o.animals[2]))z.invalid_split=true;k++;
  }
  return z;
 }
 void add_route_summary(PlanSummary&s,const RouteSummary&z,int sign,int u)const{
  s.inputs[u]=z.input;s.w+=sign*z.input.w;s.f+=sign*z.input.f;for(int c=0;c<3;c++)s.animals[c]+=sign*z.animals[c];for(int c=0;c<5;c++)s.seeds[c]+=sign*z.seeds[c];
  s.locked+=sign*z.locked;s.invalid_split_routes+=sign*int(z.invalid_split);s.land=s.locked>0;s.invalid_split=s.invalid_split_routes>0;
  s.invest=(s.animals[0]+s.animals[1]+s.animals[2]+s.seeds[0]+s.seeds[1]+s.seeds[2]+s.seeds[3]+s.seeds[4])>0;
 }
 PlanSummary summarize_plan(const Plan&p,array<RouteSummary,U>*routes_out=nullptr)const{
  PlanSummary s;s.nr=p.routes.size();for(int u=0;u<s.nr;u++){auto z=summarize_route(p.routes[u]);if(routes_out)(*routes_out)[u]=z;add_route_summary(s,z,1,u);}return s;
 }
 // Compose precomputed prefix deficits. Integer-identical to replaying every
 // action, including the conservative exclusion of decaying wheat harvests.
 Inputs route_inputs(const Route&r)const{
  Inputs need;int w=0,f=0;
  for(auto [ti,oi]:r.tasks){const auto&o=tasks[ti].opts[oi];
   for(int c=0;c<3;c++)need.animals[c]+=o.animals[c];
   need.w=max(need.w,o.need_w-w);need.f=max(need.f,o.need_f-f);
   w+=o.net_w;f+=o.net_f;
  }
  return need;
 }
 #include "preparation_methods.inc"
  double terminal_score(const Plan&p,int*after=nullptr,int*carry=nullptr)const{
   return terminal_score_setup(p,preparation(p),after,carry);
  }
 #include "market_timeline.inc"
 int shop_demand[9]{};
 #include "storage_methods.inc"

 void refine(Plan&p){
  evaluate(p);
  {Plan best=p;for(int mode=0;mode<3;mode++)if(mode!=p.preparation_mode){Plan q=p;q.preparation_mode=mode;evaluate(q);if(q.score>best.score+1e-8)best=move(q);}p=move(best);}
  {Plan alternate=p;alternate.early_pickup=!p.early_pickup;evaluate(alternate);if(alternate.score>p.score+1e-8)p=move(alternate);}
  if(day<29){for(auto&r:p.routes){bool old=r.overnight;double before=p.score;r.overnight=!old;evaluate(p);if(p.score<=before){r.overnight=old;p.score=before;}}}
  for(auto&r:p.routes){int old=r.split,best=old;double score=p.score;
   for(int sp=1;sp<(int)r.tasks.size();sp++){
    bool still_needs_inputs=false;
    for(int k=sp;k<(int)r.tasks.size();k++){auto [ti,oi]=r.tasks[k];const auto&o=tasks[ti].opts[oi];if(o.w||o.f||accumulate(o.animals.begin(),o.animals.end(),0))still_needs_inputs=true;}
    if(still_needs_inputs)continue; // DROP would discard inputs needed later.
    r.split=sp;if(route_length(r)>endhour-hour+2)continue;
    Plan tmp=p; double s=evaluate(tmp);if(s>score){score=s;best=sp;}
   }
   r.split=best;p.score=score;
  }
  evaluate(p);
 }
 uint64_t key(const Tile&t,int d){
  uint64_t k=d;k=k*8+t.kind;k=k*12+t.item;k=k*64+min(63,max(0,int(t.age)));k=k*8+min(7,int(t.y));k=k*3+min(2,int(t.dry));k=k*2+t.water;k=k*5+min(4,max(0,t.fert-d+1));k=k*2+t.care;k=k*2+t.fa;k=k*32+min(31,int(t.pending));return k;
 }
 Tile refresh(Tile t,int d){
  if(t.kind==3){
   bool watered=t.water;t.dry=watered?0:t.dry+1;t.water=0;
   if(t.dry>=2){t=Tile();t.kind=2;return t;}
   int c=t.item;
   if(ongoing(c)){
    int ds=t.age+1-first[c];
    if(ds>=0&&ds%interval_[c]==0){int pc=ds/interval_[c]+1;
     if(pc<=4)t.y=min(4,t.y+((watered&&t.fert>=d)?2:1));
    }
    // At the beginning of a decay day the crop still exists, but cannot be
    // treated as freely available all day. Future DP conservatively clears it.
    if(t.age+1>first[c]+3*interval_[c]){t=Tile();t.kind=2;return t;}
   }else if(t.age+1>maxage[c]){t=Tile();t.kind=2;return t;}
   t.age++;
  }else if(t.kind==6){
   t.dry=t.water?0:t.dry+1;
   if(t.dry>=2){int c=t.item;t=Tile();t.kind=c==5?4:5;return t;}
   int ds=t.age+1-first[t.item];
   if(ds>=0&&ds%interval_[t.item]==0){t.y=min(maxyield[t.item],t.y+1+(t.water?t.pending:0));t.pending=0;}
   if(t.water&&t.care)t.pending++;
   t.fa=1;t.water=t.care=0;t.age++;
  }
  return t;
 }
 void add_plant_variants(vector<Option>&out,const Option&prefix,int d){
  if(d!=day)return;
  for(int c=0;c<5;c++){
   if(d+first[c]>=30)continue;
   Option o=prefix;
   if(o.next.kind==1||o.next.kind==6||o.next.kind==3)continue;
   if(o.next.kind!=0)o.acts.push_back({DIG});
   o.acts.push_back({PLANT,c,1});o.acts.push_back({WATER});o.seeds[c]++;
   o.immediate-=seedcost[c];o.next=Tile();o.next.kind=3;o.next.item=c;o.next.y=ongoing(c)?0:1;o.next.water=1;o.next.dry=1;o.next.age=0;
   if(!ongoing(c))o.next.decay=(d+maxage[c]+1)*24;
   out.push_back(move(o));
  }
  if(d>=24 || !investments)return;
  for(int c=0;c<3;c++){
   Option o=prefix;
   if(o.next.kind==1||o.next.kind==6||o.next.kind==3)continue;
   int structure=c==0?4:5;
   if(o.next.kind!=structure){
    if(o.next.kind!=0)o.acts.push_back({DIG});
    o.acts.push_back({c==0?BUILD_COOP:BUILD_PASTURE});
   }
   o.acts.push_back({PLACE,9+c,1});o.animals[c]++;
   o.immediate-=animalcost[c];o.next=Tile();o.next.kind=6;o.next.item=5+c;
   // New livestock may also be fed and cared for on placement day. Keeping
   // these legal branches lets the DP price the earlier banked care bonus.
   Option fed=o;fed.acts.push_back({FEED});fed.w++;fed.immediate-=prices[d][0];fed.next.water=1;
   Option cared=fed;cared.acts.push_back({CARE});cared.next.care=1;
   out.push_back(move(o));out.push_back(move(fed));out.push_back(move(cared));
  }
 }
 vector<Option> enumerate(const Tile&t,int d){
  vector<Option> out;Option no;no.next=t;out.push_back(no);
  if(t.kind==1)return out;
  if(t.kind==0||t.kind==2||t.kind==4||t.kind==5){add_plant_variants(out,no,d);return out;}
  if(t.kind==6){
   for(int mask=1;mask<16;mask++){
    bool fd=mask&1,ca=mask&2,ha=mask&4,fe=mask&8;
    if(fd&&t.water||ca&&t.care||ha&&t.y<=0||fe&&!t.fa)continue;
    if(ca&&!(fd||t.water))continue;
    Option o;o.next=t;
    if(fd){o.acts.push_back({FEED});o.w++;o.immediate-=prices[d][0];o.next.water=1;}
    if(ca){o.acts.push_back({CARE});o.next.care=1;}
    if(ha){o.acts.push_back({HARVEST});o.immediate+=prices[d][t.item]*t.y;o.goods[t.item]+=t.y;o.next.y=0;}
    if(fe){o.acts.push_back({COLLECT_FERTILIZER});o.immediate+=prices[d][8];o.goods[8]++;o.next.fa=0;}
    out.push_back(move(o));
   }
   return out;
  }
  int c=t.item;
  for(int mode=0;mode<3;mode++){
   if(mode&&t.water)continue;
   if(mode==2&&(t.fert>=d||d==29&&ongoing(c)))continue;
   Option base;base.next=t;
   if(mode==2){base.acts.push_back({FERTILIZE});base.f++;base.immediate-=prices[d][8];base.next.fert=d+2;}
   if(mode){base.acts.push_back({WATER});base.next.water=1;
    if(!ongoing(c)&&t.age>=(maxage[c]+1)/2&&t.age<=maxage[c])base.next.y=min(maxyield[c],base.next.y+(base.next.fert>=d?2:1));
   }
   if(mode)out.push_back(base);
   if(t.age>=first[c]&&base.next.y>0){
    Option h=base;h.acts.push_back({HARVEST});h.immediate+=base.next.y*prices[d][c];h.goods[c]+=base.next.y;h.next.y=0;
    if(!ongoing(c))h.next=Tile();
    out.push_back(h);
    if(!ongoing(c)){
     add_plant_variants(out,h,d);
     if((d>day) && c<2 && d+first[c]<30){
      Option again=h;again.acts.push_back({PLANT,c,1});again.acts.push_back({WATER});
      again.seeds[c]++;again.immediate-=seedcost[c];again.next=Tile();
      again.next.kind=3;again.next.item=c;again.next.y=1;again.next.water=1;
      again.next.dry=1;again.next.decay=(d+maxage[c]+1)*24;
      out.push_back(move(again));
     }
    }
    else if(t.age>=first[c]+3*interval_[c]){
     h.acts.push_back({DIG});h.next=Tile();out.push_back(h);add_plant_variants(out,h,d);
    }
   }
  }
  // Replace an exhausted ongoing crop without waiting for weeds.
  if(ongoing(c)&&t.age>=first[c]+3*interval_[c]&&t.y==0){
   Option o;o.next=Tile();o.acts.push_back({DIG});out.push_back(o);add_plant_variants(out,o,d);
  }
  return out;
 }
 double time_discount(int d)const{return d<8?.90:(d<14?.97:1.);}
 double value(Tile t,int d){
  if(d>=30||t.kind==0||t.kind==1||t.kind==2||t.kind==4||t.kind==5)return 0;
  uint64_t k=key(t,d);auto it=memo.find(k);if(it!=memo.end())return it->second;
  auto os=enumerate(t,d);double best=-1e100;
  for(auto&o:os){
   double score=o.immediate-future_action_cost(o,d);
   if(d<29)score+=time_discount(d)*value(refresh(o.next,d),d+1);
   best=max(best,score);
  }
  memo[k]=best;return best;
 }
 void make_tasks(){
  int nextq=-1;int checks[]={45,54,55};int lc[]={1000,2000,4000};
  for(int q=0;q<3;q++)if(tiles[checks[q]].kind==1){nextq=q;landprice=lc[q];break;}
  for(int pos=0;pos<100;pos++){
   Tile t=tiles[pos];if(t.kind==1){
   int x=pos%10,y=pos/10;int quadrant=x>=5?(y>=5?2:0):1;
   if(day>=23||nextq!=quadrant||money<landprice+100)continue;
   t=Tile();
  }
   auto opts=enumerate(t,day);
   double base=day<29?time_discount(day)*value(refresh(t,day),day+1):0;
   Task task;task.pos=pos;
   for(auto&o:opts){
    if(o.acts.empty())continue;
     o.value=o.immediate+(day<29?time_discount(day)*value(refresh(o.next,day),day+1):0)-base;
     // Standalone value orders proposals. Keep every legal action sequence
     // for the common market and alternative continuation-policy comparison.
     task.opts.push_back(move(o));
   }
   if(task.opts.empty())continue;
   sort(task.opts.begin(),task.opts.end(),[](const auto&a,const auto&b){return a.value>b.value;});
    vector<Option> kept;vector<uint64_t> kept_keys;
    for(auto&o:task.opts){
     bool dominated=false;uint64_t ok=key(o.next,day);
     for(size_t qi=0;qi<kept.size();qi++){auto&q=kept[qi];
      if(q.value>=o.value-1e-8&&q.acts.size()<=o.acts.size()&&q.w<=o.w&&q.f<=o.f&&q.seeds==o.seeds&&q.animals==o.animals&&q.goods==o.goods&&kept_keys[qi]==ok){dominated=true;break;}}
     if(!dominated){kept_keys.push_back(ok);kept.push_back(move(o));}
    }
    task.opts=move(kept);
    vector<array<int,4>> route_signatures;

    for(auto&o:task.opts){
     o.duration=(uint8_t)o.acts.size();int w=0,f=0;
     for(int j=0;j<(int)o.duration;j++){
      const auto&a=o.acts[j];
      if(a.op==FEED){--w;o.need_w=max(o.need_w,-w);}
      else if(a.op==FERTILIZE){--f;o.need_f=max(o.need_f,-f);}
      else if(a.op==COLLECT_FERTILIZER)++f;
      else if(a.op==HARVEST&&tiles[pos].item==0){
       const auto&tt=tiles[pos];
       if(!(tt.kind==3&&tt.decay>=0&&tt.decay<=day*24))w+=o.goods[0];
      }
      if(a.op==PLANT){o.plant_item=a.item;o.plant_step=j;}
      if(a.op==FEED||a.op==FERTILIZE||a.op==COLLECT_FERTILIZER||a.op==HARVEST){int e=o.effect_count++;o.effect_step[e]=j;o.effect_op[e]=a.op;o.effect_item[e]=a.item;}
     }
     o.net_w=w;o.net_f=f;o.goods_total=accumulate(o.goods.begin(),o.goods.end(),0);o.net_goods=o.goods_total-o.w-o.f;o.seeds_total=accumulate(o.seeds.begin(),o.seeds.end(),0);o.animals_total=accumulate(o.animals.begin(),o.animals.end(),0);o.investment=0;for(int c=0;c<5;c++)o.investment+=seedcost[c]*o.seeds[c];for(int c=0;c<3;c++)o.investment+=animalcost[c]*o.animals[c];
     array<int,4> sig{o.need_w,o.need_f,o.net_w,o.net_f};
     auto it=find(route_signatures.begin(),route_signatures.end(),sig);
     o.route_class=it-route_signatures.begin();if(it==route_signatures.end())route_signatures.push_back(sig);
    }
    if(task.opts.size()>255)throw std::runtime_error("too many task options");
    tasks.push_back(move(task));
  }
 }
 #include "future_methods.inc"
  int tail_goods(const Route&r)const{return tail_goods_inputs(r,route_inputs(r));}
  int tail_goods_inputs(const Route&r,const Inputs&inputs)const{
   int bag=inputs.w+inputs.f,j=0;
   for(auto [ti,oi]:r.tasks){bag+=tasks[ti].opts[oi].net_goods;if(++j==r.split)bag=0;}
   return bag;
  }
  int route_length(const Route&r)const{
   if(r.tasks.empty())return 0;
   int needw=0,needf=0,w=0,f=0,bagnet=0,at=r.start,time=0,j=0,animalmask=0;bool split_seen=false;
   for(auto [ti,oi]:r.tasks){const auto&t=tasks[ti];const auto&o=t.opts[oi];
    needw=max(needw,o.need_w-w);needf=max(needf,o.need_f-f);w+=o.net_w;f+=o.net_f;
    for(int c=0;c<3;c++)if(o.animals[c])animalmask|=1<<c;
    bagnet+=o.net_goods;time+=dist(at,t.pos)+(int)o.duration;at=t.pos;
    if(++j==r.split){time+=dshed(at)+1;at=shed(at);bagnet=0;split_seen=true;}
   }
   int tg=bagnet+(split_seen?0:needw+needf);if(tg>0&&!r.overnight)time+=dshed(at)+1;
   time+=(needw>0)+(needf>0)+__builtin_popcount((unsigned)animalmask);return time;
  }
  int route_length_inputs(const Route&r,const Inputs&inputs,int tg)const{
   if(r.tasks.empty())return 0;
   int at=r.start,time=0,j=0;
   for(auto [ti,oi]:r.tasks){const auto&t=tasks[ti];const auto&o=t.opts[oi];time+=dist(at,t.pos)+o.duration;at=t.pos;if(++j==r.split){time+=dshed(at)+1;at=shed(at);}}
   if(tg>0&&!r.overnight)time+=dshed(at)+1;
   time+=(inputs.w>0)+(inputs.f>0);for(int c=0;c<3;c++)time+=(inputs.animals[c]>0);return time;
  }
  // Fused tail-goods + length single task pass (integer-only, so the result
  // is bit-identical to tail_goods_inputs + route_length_inputs).
  int route_length_goods_inputs(const Route&r,const Inputs&inputs,int&tg,int start_override=-1)const{
   if(r.tasks.empty()){tg=0;return 0;}
   int bag=inputs.w+inputs.f;
   int at=start_override>=0?start_override:r.start,time=0,j=0;
   for(auto [ti,oi]:r.tasks){
    const auto&t=tasks[ti];const auto&o=t.opts[oi];
    bag+=o.net_goods;time+=dist(at,t.pos)+(int)o.duration;at=t.pos;
    if(++j==r.split){time+=dshed(at)+1;at=shed(at);bag=0;}
   }
   tg=bag;if(tg>0&&!r.overnight)time+=dshed(at)+1;
   time+=(inputs.w>0)+(inputs.f>0);for(int c=0;c<3;c++)time+=(inputs.animals[c]>0);return time;
  }
  bool decay_ok(const Route&r,const Plan&p)const{
   bool any=false;for(auto [ti,oi]:r.tasks){const auto&t=tiles[tasks[ti].pos];if(t.kind==3&&t.decay>=0&&t.decay<=day*24)any=true;}
   if(!any)return true;
   return decay_ok_inputs(r,p,route_inputs(r));
  }
  bool decay_ok_resolved(const Route&r,const Inputs&inputs,int start,int begin)const{
   bool any=false;for(auto [ti,oi]:r.tasks){const auto&t=tiles[tasks[ti].pos];if(t.kind==3&&t.decay>=0&&t.decay<=day*24)any=true;}
   if(!any)return true;
   int at=start,h=begin,j=0;
   for(auto [ti,oi]:r.tasks){const auto&task=tasks[ti];const auto&t=tiles[task.pos];const auto&o=task.opts[oi];h+=dist(at,task.pos);at=task.pos;
    for(const auto&a:o.acts){if(a.op==HARVEST&&t.kind==3&&t.decay>=0&&t.decay<=day*24&&t.y-(h+1)/2<=0)return false;h++;}
    if(++j==r.split){h+=dshed(at)+1;at=shed(at);}
   }
   return true;
  }
  bool decay_ok_inputs(const Route&r,const Plan&p,const Inputs&inputs)const{
   if(r.begin>=0)return decay_ok_resolved(r,inputs,r.start,r.begin);
   int h=p.ready+(inputs.w>0)+(inputs.f>0);for(int c=0;c<3;c++)h+=(inputs.animals[c]>0);
   return decay_ok_resolved(r,inputs,r.start,h);
  }
 inline double cost(int workers)const{return wage_total[workers]-wage_total[max(1,existing)];}
 struct EvalResult{double score=-1e100,forecast_margin=-1e100;Preparation setup;bool resolved=false;};
 double match_score(double forecast_margin)const{return 10000.*expected_terminal_score(forecast_margin,terminal_uncertainty);}
 EvalResult evaluate_plan_summary(const Plan&p,const PlanSummary&summary){
  EvalResult result;
  if(!p.allow_land&&summary.land)return result;if(summary.invalid_split)return result;
  TerminalTiming timing;preparation_uncached_into(p,summary,result.setup,&timing);if(!result.setup.valid)return result;result.resolved=true;
  if(result.setup.purchase_cost+operating_reserve(summary)>money)return result;
  int after[9]{},carry[9]{};double score=terminal_score_timed(p,result.setup,timing,after,carry,&summary);if(score<-1e50){result.score=score;return result;}
  if(day<29){
   score+=future_value(p,after,carry);
   for(int c=0;c<5;c++)score-=seedcost[c]*max(0,summary.seeds[c]-seeds[c]);for(int c=0;c<3;c++)score-=animalcost[c]*max(0,summary.animals[c]-stock[9+c]);
  }
  // Preserve affordability and route feasibility, then rank final candidates
  // by expected terminal match score. 10000 is numerical search scaling only.
  result.forecast_margin=money-opponent_money+score;
  result.score=match_score(result.forecast_margin);return result;
 }
 EvalResult evaluate_plan(const Plan&p){auto summary=summarize_plan(p);return evaluate_plan_summary(p,summary);}
 void apply_evaluation(Plan&p,const EvalResult&r){p.score=r.score;if(!r.resolved)return;p.ready=r.setup.ready;for(int u=0;u<U;u++)p.preloaded[u]=r.setup.preloaded[u];for(int u=0;u<(int)p.routes.size();u++){p.routes[u].begin=r.setup.begin[u];p.routes[u].start=r.setup.starts[u];}}
 double evaluate(Plan&p){auto r=evaluate_plan(p);apply_evaluation(p,r);return r.score;}
 vector<int> starts(int workers)const{
  // Access tiles in the engine's NWSE order ((4,4),(5,4),(4,5),(5,5)).
  vector<int>s=positions;int locs[]={44,45,54,55};
  while((int)s.size()<workers){int best=44,bc=999;for(int z:locs){int c=count(s.begin(),s.end(),z);if(c<bc){best=z;bc=c;}}s.push_back(best);}return s;
 }
 // Walk `steps` tiles from `at` toward `target`; emit/engine both move on X
 // first and then Y, one action per turn.
 inline int move_along(int at,int target,int steps)const{
  if(steps<=0)return at;
  int x=at%10,y=at/10,tx=target%10,ty=target/10;
  int dx=tx-x,dy=ty-y,ax=abs(dx);
  if(steps<=ax)return x+(dx>0?steps:-steps)+10*y;
  steps-=ax;x=tx;
  int ay=abs(dy);
  if(steps>=ay)return tx+10*ty;
  return x+10*(y+(dy>0?steps:-steps));
 }
 // Position after the farm phase of turn t for one route. The action of turn
 // t has already happened when the market phase resolves HIREs.
 int position_at(const Route&r,int begin,int t,int start=-1)const{
  if(start<0)start=r.start;
  if(t<begin||r.tasks.empty())return start;
  int at=start,h=begin,j=0;
  for(auto [ti,oi]:r.tasks){
   const auto&task=tasks[ti];const auto&o=task.opts[oi];
   int d=dist(at,task.pos);
   if(t<h+d)return move_along(at,task.pos,t-h+1);
   h+=d;at=task.pos;
   for(const auto&a:o.acts){if(t==h)return at;h++;}
   if(++j==r.split){d=dshed(at);if(t<h+d)return move_along(at,shed(at),t-h+1);h+=d;at=shed(at);if(t==h)return at;h++;}
  }
  if(tail_goods(r)>0&&!r.overnight){int d=dshed(at);if(t<h+d)return move_along(at,shed(at),t-h+1);h+=d;at=shed(at);if(t==h)return at;}
  return at;
 }
 // HIRE spawns at the least-occupied access tile (NWSE order) during the
 // market phase of its hire turn, after that turn's farm actions. Resolve each
 // hired route's start from the positions of the units that already exist.
 void resolve_spawns(Plan&p,const Preparation&setup){
  for(int u=0;u<(int)p.routes.size();u++)p.routes[u].start=setup.starts[u];
 }
 int land_cost(const Plan&p,const Option*extra=nullptr,int pos=-1)const{
  if(pos>=0&&tiles[pos].kind==1)return landprice;
  for(const auto&r:p.routes)for(auto [ti,oi]:r.tasks)if(tiles[tasks[ti].pos].kind==1)return landprice;
  return 0;
 }
 double operating_reserve(const PlanSummary&s)const{if(day>=24||!s.invest)return 0.;int n=tiles_animals+s.animals[0]+s.animals[1]+s.animals[2];return min(money*.5,max(75.,n*price(0,mi[0]-n)+35.));}
 double operating_reserve(const Plan&p,const Option*extra=nullptr)const{
  auto s=summarize_plan(p);if(extra){for(int c=0;c<3;c++)s.animals[c]+=extra->animals[c];for(int c=0;c<5;c++)s.seeds[c]+=extra->seeds[c];s.invest=s.invest||extra->animals[0]||extra->animals[1]||extra->animals[2]||extra->seeds[0]||extra->seeds[1]||extra->seeds[2]||extra->seeds[3]||extra->seeds[4];}return operating_reserve(s);
 }
 double purchase_cost(const Plan&p,const PlanSummary&s)const{
  int need[12]{};int ns[5]{};need[0]=s.w;need[8]=s.f;for(int c=0;c<3;c++)need[9+c]=s.animals[c];for(int c=0;c<5;c++)ns[c]=s.seeds[c];
  double cost_=cost(s.nr)+(s.land?landprice:0);
  for(int c=0;c<9;c++){int reserved=p.holding[c];int missing=need[c]-stock[c]+reserved;if(missing)cost_+=missing*price(c,mi[c]-missing);}
  for(int c=0;c<3;c++)cost_+=animalcost[c]*max(0,need[9+c]-stock[9+c]);
  for(int c=0;c<5;c++)cost_+=seedcost[c]*max(0,ns[c]-seeds[c]);
  if(accumulate(need,need+12,0)+accumulate(holds,holds+9,0)>100)return 1e100;return cost_;
 }
 double purchase_cost(const Plan&p,const Option* extra=nullptr,int pos=-1)const{
  int need[12]{};int ns[5]{};
  auto add=[&](const Option&o){need[0]+=o.w;need[8]+=o.f;for(int c=0;c<3;c++)need[9+c]+=o.animals[c];for(int c=0;c<5;c++)ns[c]+=o.seeds[c];};
  for(const auto&r:p.routes)for(auto [ti,oi]:r.tasks)add(tasks[ti].opts[oi]);
  need[0]=need[8]=0;for(const auto&r:p.routes){auto x=route_inputs(r);need[0]+=x.w;need[8]+=x.f;}
  if(extra)add(*extra);
  double cost_=cost(p.routes.size())+land_cost(p,extra,pos);
  for(int c=0;c<9;c++){
   int reserved=p.holding[c];
   int missing=need[c]-stock[c]+reserved;
   if(missing>0)cost_+=missing*price(c,mi[c]-missing);
   else if(missing<0)cost_+=missing*price(c,mi[c]-missing); // conservative sale quote
  }
  for(int c=0;c<3;c++)cost_+=animalcost[c]*max(0,need[9+c]-stock[9+c]);
  for(int c=0;c<5;c++)cost_+=seedcost[c]*max(0,ns[c]-seeds[c]);
  if(accumulate(need,need+12,0)+accumulate(holds,holds+9,0)>100)return 1e100;
  return cost_;
 }
 void repair(Plan&p,vector<int> order,double noise,int preferred=-1,bool shortlist=false){
  auto& rank=repair_scratch.rank;rank.clear();
  uniform_real_distribution<double> ur(0,1);
  double capital_bias=noise>0 ? (ur(rng)<.3?0.:18.*ur(rng)) : 0.;
  for(int ti:order){auto&t=tasks[ti];double density=0;
   for(auto&o:t.opts)density=max(density,(o.value-capital_bias*o.investment)/(o.acts.size()+1.6+dshed(t.pos)*.25));
   rank.push_back({-density*exp(noise*(ur(rng)-.5)),ti});
  }
  sort(rank.begin(),rank.end());
  evaluate(p);
  // Any hired worker can move between shed access tiles when preparation
  // changes. Allow their maximum two-step start displacement in cheap pruning.
  int maxlen=endhour-hour+2;
  // R2-3b: cache per-route old stats across ti (routes only change on accept).
  // Values are reused bit-exactly; rng untouched.
  int nroutes=p.routes.size();
  auto& c_len=repair_scratch.c_len;auto& c_goods=repair_scratch.c_goods;auto& c_w=repair_scratch.c_w;auto& c_f=repair_scratch.c_f;
  auto& c_start=repair_scratch.c_start;c_start.resize(nroutes);
  c_len.assign(nroutes,-1);c_goods.assign(nroutes,0);c_w.assign(nroutes,0);c_f.assign(nroutes,0);
  auto& c_anim=repair_scratch.c_anim;auto& c_hasdecay=repair_scratch.c_hasdecay;auto& c_valid=repair_scratch.c_valid;
  c_anim.resize(nroutes);c_hasdecay.assign(nroutes,0);c_valid.assign(nroutes,0);
  struct PlanNeeds{int w=0,f=0;array<int,3> anim{};array<int,5> seeds{};int land=0;};
  PlanNeeds plan_need;auto& route_need=repair_scratch.route_need;route_need.resize(nroutes);
  for(int ri=0;ri<nroutes;ri++){const auto&r0=p.routes[ri];auto x=route_inputs(r0);route_need[ri]=x;plan_need.w+=x.w;plan_need.f+=x.f;for(const auto&v:r0.tasks){const auto&o0=tasks[v.first].opts[v.second];for(int c=0;c<3;c++)plan_need.anim[c]+=o0.animals[c];for(int c=0;c<5;c++)plan_need.seeds[c]+=o0.seeds[c];}}
  plan_need.land=land_cost(p);const double repair_workers_cost=cost(p.routes.size());const int holding_total=accumulate(p.holding.begin(),p.holding.end(),0);const bool reserve_disabled=day>=24;
  auto accepted_insertion=[&](int ri,int oi,int task_index){auto now=route_inputs(p.routes[ri]);plan_need.w+=now.w-route_need[ri].w;plan_need.f+=now.f-route_need[ri].f;route_need[ri]=now;const auto&o=tasks[task_index].opts[oi];for(int c=0;c<3;c++)plan_need.anim[c]+=o.animals[c];for(int c=0;c<5;c++)plan_need.seeds[c]+=o.seeds[c];if(tiles[tasks[task_index].pos].kind==1)plan_need.land=landprice;};
  auto candidate_summary=[&](int ri,int oi,int task_index){
   PlanSummary s;s.nr=nroutes;for(int u=0;u<nroutes;u++)s.inputs[u]=route_need[u];
   auto now=route_inputs(p.routes[ri]);s.inputs[ri]=now;s.w=plan_need.w+now.w-route_need[ri].w;s.f=plan_need.f+now.f-route_need[ri].f;
   s.animals=plan_need.anim;s.seeds=plan_need.seeds;const auto&o=tasks[task_index].opts[oi];for(int c=0;c<3;c++)s.animals[c]+=o.animals[c];for(int c=0;c<5;c++)s.seeds[c]+=o.seeds[c];
   s.land=plan_need.land||tiles[tasks[task_index].pos].kind==1;s.invest=(s.animals[0]+s.animals[1]+s.animals[2]+s.seeds[0]+s.seeds[1]+s.seeds[2]+s.seeds[3]+s.seeds[4])>0;
   const auto&r=p.routes[ri];if(r.split>0){int at=0;for(auto [xt,xo]:r.tasks){if(at++>=r.split){const auto&z=tasks[xt].opts[xo];if(z.w||z.f||z.animals[0]||z.animals[1]||z.animals[2]){s.invalid_split=true;break;}}}}
   return s;
  };
  auto& choice_score=repair_scratch.choice_score;auto& choice_route=repair_scratch.choice_route;
  auto& choice_position=repair_scratch.choice_position;
  auto& sc_nwDW=repair_scratch.sc_nwDW;auto& sc_nwDF=repair_scratch.sc_nwDF;
  auto& sc_nwMW=repair_scratch.sc_nwMW;auto& sc_nwMF=repair_scratch.sc_nwMF;
   for(auto [rr,ti]:rank){
    auto&t=tasks[ti];if(tiles[t.pos].kind==1&&!p.allow_land)continue;double best=-1e100;int br=-1,bp=-1,bo=-1;
    choice_score.assign(t.opts.size(),-1e100);
    choice_route.assign(t.opts.size(),-1);choice_position.assign(t.opts.size(),-1);
    uint64_t eligible[4]{};int eligible_words=(t.opts.size()+63)/64;
    int base_w=plan_need.w,base_f=plan_need.f;int base_anim[3]={plan_need.anim[0],plan_need.anim[1],plan_need.anim[2]};int base_ns[5]={plan_need.seeds[0],plan_need.seeds[1],plan_need.seeds[2],plan_need.seeds[3],plan_need.seeds[4]};
    int base_anim_total=base_anim[0]+base_anim[1]+base_anim[2];int base_ns_total=base_ns[0]+base_ns[1]+base_ns[2]+base_ns[3]+base_ns[4];bool base_invest=base_anim_total>0||base_ns_total>0;int base_n=tiles_animals+base_anim_total;int extra_land=(tiles[t.pos].kind==1)?landprice:plan_need.land;
    for(int oi=0;oi<(int)t.opts.size();oi++){
     // Continuation alternatives inherit their parent's current-day action.
     // Their insertion choices are copied below; only parents need this test.
     const auto&eo=t.opts[oi];if(eo.future_parent>=0)continue;
     int need0=base_w+eo.w,need8=base_f+eo.f;
     int need9[3]={base_anim[0]+eo.animals[0],base_anim[1]+eo.animals[1],base_anim[2]+eo.animals[2]};
     int nst[5]={base_ns[0]+eo.seeds[0],base_ns[1]+eo.seeds[1],base_ns[2]+eo.seeds[2],base_ns[3]+eo.seeds[3],base_ns[4]+eo.seeds[4]};
     double pc=repair_workers_cost+extra_land;
     for(int c=0;c<9;c++){
      int needc=0;
      if(c==0)needc=need0;else if(c==8)needc=need8;
      int reserved=p.holding[c];
      int missing=needc-stock[c]+reserved;
      if(missing>0)pc+=missing*price(c,mi[c]-missing);
      else if(missing<0)pc+=missing*price(c,mi[c]-missing);
     }
     for(int c=0;c<3;c++)pc+=animalcost[c]*max(0,need9[c]-stock[9+c]);
     for(int c=0;c<5;c++)pc+=seedcost[c]*max(0,nst[c]-seeds[c]);
     int needsum=need0+need8+need9[0]+need9[1]+need9[2]+accumulate(holds,holds+9,0);
     if(needsum>100)pc=1e100;
     double rv=0;
     if(!reserve_disabled){
      int extra_anim_total=eo.animals_total;
      int extra_ns_total=eo.seeds_total;
      bool investt=base_invest||extra_anim_total>0||extra_ns_total>0;
      if(investt){int nt=base_n+extra_anim_total;rv=min(money*.5,max(75.,nt*price(0,mi[0]-nt)+35.));}
     }
     // Harvested wheat and collected fertilizer can fund later actions on
     // the same route. Gross input needs are not a valid rejection bound.
     // The full evaluator checks the exact prefix deficit after insertion.
     bool affordable=pc+rv<=money || eo.w>0 || eo.f>0;
     if(preferred>=0){
      for(int c=0;c<5;c++)if(eo.seeds[c]&&c!=preferred)affordable=false;
      for(int c=0;c<3;c++)if(eo.animals[c]&&c+5!=preferred)affordable=false;
     }
     if(affordable&&eo.future_parent<0)eligible[oi>>6]|=uint64_t(1)<<(oi&63);
    }
    // CTZ enumeration preserves option order and skips whole ineligible tiles.
    if(!(eligible[0]|eligible[1]|eligible[2]|eligible[3]))continue;
    // O(1) exact try-length inputs: per-option invariants (route/position
    // independent). Integer-only; same accumulate/investment_cost expressions.
    int nopts=t.opts.size();
    sc_nwDW.assign(nopts,0);sc_nwDF.assign(nopts,0);sc_nwMW.assign(nopts,0);sc_nwMF.assign(nopts,0);
    {const auto&ttt0=tiles[t.pos];
    for(int oi=0;oi<nopts;oi++){
     const auto&eo=t.opts[oi];
     sc_nwDW[oi]=eo.net_w;sc_nwDF[oi]=eo.net_f;sc_nwMW[oi]=eo.need_w;sc_nwMF[oi]=eo.need_f;
    }}
    bool tileDecays=(tiles[t.pos].decay>=0&&tiles[t.pos].decay<=day*24);
   for(int ri=0;ri<(int)p.routes.size();ri++){
    auto&r=p.routes[ri];
    auto& sc_preW=repair_scratch.route_work[ri][0];auto& sc_preF=repair_scratch.route_work[ri][1];
    auto& sc_preNW=repair_scratch.route_work[ri][2];auto& sc_preNF=repair_scratch.route_work[ri][3];
    auto& sc_suMW=repair_scratch.route_work[ri][4];auto& sc_suMF=repair_scratch.route_work[ri][5];
    auto& sc_gsuf=repair_scratch.route_work[ri][6];auto& sc_pos=repair_scratch.route_work[ri][7];
    auto& route_travel=repair_scratch.route_travel[ri];
    // Preparation can relocate unchanged workers. Only their first leg changes.
    bool route_changed=!c_valid[ri];int start_delta=0;
    int oldlen,oldw,oldf,oldgoods;array<int,3> oldanimals{};bool route_has_decay;
    if(c_valid[ri]){oldlen=c_len[ri];if(!r.tasks.empty()){int first=tasks[r.tasks[0].first].pos;start_delta=dist(r.start,first)-dist(c_start[ri],first);oldlen+=start_delta;}oldgoods=c_goods[ri];oldw=c_w[ri];oldf=c_f[ri];oldanimals=c_anim[ri];route_has_decay=c_hasdecay[ri];}
     else{
     auto oldinputs=route_inputs(r);oldlen=route_length_goods_inputs(r,oldinputs,oldgoods);oldw=0;oldf=0;oldanimals={};route_has_decay=false;
     for(auto [xx,yy]:r.tasks){auto&tt=tiles[tasks[xx].pos];if(tt.kind==3&&tt.decay>=0&&tt.decay<=day*24)route_has_decay=true;}
     for(auto [x,y]:r.tasks){const auto&o=tasks[x].opts[y];oldw+=o.w;oldf+=o.f;for(int c=0;c<3;c++)oldanimals[c]+=o.animals[c];}
     c_start[ri]=r.start;c_len[ri]=oldlen;c_goods[ri]=oldgoods;c_w[ri]=oldw;c_f[ri]=oldf;c_anim[ri]=oldanimals;c_hasdecay[ri]=route_has_decay;c_valid[ri]=1;
    }
    int L=(int)r.tasks.size();
    if(r.split>L)throw std::runtime_error("invalid route split");
    if(route_changed){
     sc_preW[0]=0;sc_preF[0]=0;sc_preNW[0]=0;sc_preNF[0]=0;route_travel=0;
     int prevp=r.start;
     for(int k=0;k<L;k++){
      int ak=r.tasks[k].first,bk=r.tasks[k].second;
      const auto&oo=tasks[ak].opts[bk];
      sc_pos[k]=tasks[ak].pos;
      sc_preNW[k+1]=max(sc_preNW[k],oo.need_w-sc_preW[k]);
      sc_preNF[k+1]=max(sc_preNF[k],oo.need_f-sc_preF[k]);
      sc_preW[k+1]=sc_preW[k]+oo.net_w;sc_preF[k+1]=sc_preF[k]+oo.net_f;
      route_travel+=dist(prevp,sc_pos[k])+(int)oo.duration;prevp=sc_pos[k];
     }
     sc_suMW[L]=0;sc_suMF[L]=0;sc_gsuf[L]=0;
     for(int k=L-1;k>=0;k--){
      const auto&oo=tasks[r.tasks[k].first].opts[r.tasks[k].second];
      sc_suMW[k]=max(oo.need_w,-oo.net_w+sc_suMW[k+1]);
      sc_suMF[k]=max(oo.need_f,-oo.net_f+sc_suMF[k+1]);
      sc_gsuf[k]=oo.net_goods+sc_gsuf[k+1];
     }
    }
    bool check_decay=(tiles[t.pos].kind==3&&tiles[t.pos].decay>=0&&tiles[t.pos].decay<=day*24)||route_has_decay;
    // Specialize once per route so the inner scan drops unused return/split work.
    auto scan=[&](auto overnight_tag,auto split_tag){
     constexpr bool Overnight=decltype(overnight_tag)::value,Split=decltype(split_tag)::value;
    for(int word=0;word<eligible_words;word++)for(uint64_t bits=eligible[word];bits;bits&=bits-1){
     int oi=word*64+__builtin_ctzll(bits);auto&o=t.opts[oi];
     for(int j=0;j<=L;j++){
      int prev=j==0?r.start:sc_pos[j-1];
      int after=j==L?-1:sc_pos[j];
      int newgoods=oldgoods+o.goods_total;
      int gap=dist(prev,t.pos)+(after<0?0:dist(t.pos,after)-dist(prev,after));
      int delta=gap+(after<0?(newgoods?dshed(t.pos):0)-(oldgoods?dshed(prev):0):0)+(int)o.duration;
      if(!oldgoods&&newgoods){delta++;if(after>=0)delta+=dshed(tasks[r.tasks.back().first].pos);}

      if(!oldw&&o.w)delta++;
      if(!oldf&&o.f)delta++;for(int c=0;c<3;c++)if(!oldanimals[c]&&o.animals[c])delta++;
      if(!Overnight&&oldlen+delta>maxlen+2)continue;
      int real_length;
      {
       // Exact O(1) length from prefix/suffix aggregates: same integer value
       // as insert + route_length(r), with no mutation. No FP/RNG involved.
       int Mw_=sc_preW[j]+sc_nwDW[oi],Mf_=sc_preF[j]+sc_nwDF[oi];
       int MnW=max(sc_preNW[j],-sc_preW[j]+sc_nwMW[oi]);
       int MnF=max(sc_preNF[j],-sc_preF[j]+sc_nwMF[oi]);
       int netW=max(MnW,-Mw_+sc_suMW[j]);
       int netF=max(MnF,-Mf_+sc_suMF[j]);
       int tg;
       if(!Split)tg=netW+netF+sc_gsuf[0]+o.net_goods;
       else tg=netW+netF+((j>=r.split)?sc_gsuf[r.split-1]:sc_gsuf[r.split])+o.net_goods;
       int rl=route_travel+start_delta+(int)o.acts.size()+gap;
       if(Split){
        int hi=r.split-1;
        int hopos=(hi<j)?sc_pos[hi]:((hi==j)?t.pos:sc_pos[hi-1]);
        rl+=dshed(hopos)+1;
       }
       if(tg>0&&!Overnight)rl+=dshed((L==j)?t.pos:sc_pos[L-1])+1;
       rl+=(netW>0)+(netF>0);
       int an0=oldanimals[0]+o.animals[0],an1=oldanimals[1]+o.animals[1],an2=oldanimals[2]+o.animals[2];
       rl+=(an0>0)+(an1>0)+(an2>0);
       real_length=rl;if(real_length>maxlen)continue;
       delta=real_length-oldlen;
       // Expiring crops cannot wait past their exact every-other-turn decay.
       if(tileDecays){
        int arrival=hour+dshed(t.pos);
        if(arrival>=2*tiles[t.pos].y)continue;
       }
      }
      double sc=o.value-capital_bias*o.investment-0.12*delta;
      // Prefer compact routes and early delivery when scores tie.
      sc-=0.00001*(oldlen+delta);
      if(sc>best){best=sc;br=ri;bp=j;bo=oi;}
      if(sc>choice_score[oi]){choice_score[oi]=sc;choice_route[oi]=ri;choice_position[oi]=j;}
     }
    }
    };
    // Overnight routes without an intermediate delivery need only the best
    // travel bucket and the next two: wheat/fertilizer pickup adds at most two.
    auto scan_positions=[&](){
     if(tileDecays&&hour+dshed(t.pos)>=2*tiles[t.pos].y)return;
     int travel[TaskList::capacity+1],min_travel=100000;
     for(int j=0;j<=L;j++){
      int prev=j==0?r.start:sc_pos[j-1],after=j==L?-1:sc_pos[j];
      travel[j]=route_travel+start_delta+dist(prev,t.pos)+(after<0?0:dist(t.pos,after))-(after<0?0:dist(prev,after));
      min_travel=min(min_travel,travel[j]);
     }
     uint64_t buckets[3]{};
     for(int j=0;j<=L;j++){int extra=travel[j]-min_travel;if(extra<=2)buckets[extra]|=uint64_t(1)<<j;}
     // Options sharing prefix deficits and net resources share the best slot.
     uint64_t relevant=buckets[0]|buckets[1]|buckets[2],done[4]{};
     int class_length[256],class_position[256];
     for(int word=0;word<eligible_words;word++)for(uint64_t option_bits=eligible[word];option_bits;option_bits&=option_bits-1){
      int oi=word*64+__builtin_ctzll(option_bits);const auto&o=t.opts[oi];
      int fixed_length=(int)o.duration+(oldanimals[0]+o.animals[0]>0)+(oldanimals[1]+o.animals[1]>0)+(oldanimals[2]+o.animals[2]>0);
      if(min_travel+fixed_length>maxlen)continue;
      int cl=o.route_class;uint64_t bit=uint64_t(1)<<(cl&63);
      if(!(done[cl>>6]&bit)){
       uint64_t wheat=0,fert=0;
       for(uint64_t bits=relevant;bits;bits&=bits-1){
        int j=__builtin_ctzll(bits);uint64_t posbit=uint64_t(1)<<j;
        if(sc_preNW[j]>0||o.need_w>sc_preW[j]||sc_suMW[j]>sc_preW[j]+o.net_w)wheat|=posbit;
        if(sc_preNF[j]>0||o.need_f>sc_preF[j]||sc_suMF[j]>sc_preF[j]+o.net_f)fert|=posbit;
       }
       uint64_t zero=~(wheat|fert),one=wheat^fert,two=wheat&fert;
       // Lowest set bit keeps the original first-position tie break.
       uint64_t positions=buckets[0]&zero;int penalty=0;
       if(!positions){penalty=1;positions=(buckets[0]&one)|(buckets[1]&zero);}
       if(!positions){penalty=2;positions=(buckets[0]&two)|(buckets[1]&one)|(buckets[2]&zero);}
       class_length[cl]=min_travel+penalty;class_position[cl]=__builtin_ctzll(positions);done[cl>>6]|=bit;
      }
      int length=class_length[cl]+fixed_length;
      if(length>maxlen)continue;
      int position=class_position[cl],delta=length-oldlen;
      double sc=o.value-capital_bias*o.investment-0.12*delta;sc-=0.00001*(oldlen+delta);
      if(sc>best){best=sc;br=ri;bp=position;bo=oi;}
      if(sc>choice_score[oi]){choice_score[oi]=sc;choice_route[oi]=ri;choice_position[oi]=position;}
     }
    };
    if(r.overnight){if(r.split>0)scan(true_type{},true_type{});else scan_positions();}
    else{if(r.split>0)scan(false_type{},true_type{});else scan(false_type{},false_type{});}
   }
   for(int oi=0;oi<(int)t.opts.size();oi++)if(t.opts[oi].future_parent>=0){
    int source=t.opts[oi].future_parent;choice_score[oi]=choice_score[source];choice_route[oi]=choice_route[source];choice_position[oi]=choice_position[source];
   }
   if(br>=0){
    double before=p.score;int oldland=plan_need.land;int newland=oldland||(tiles[t.pos].kind==1)?landprice:0;
    if(shortlist){
     int second=-1;double second_score=-1e100;
     for(int oi=0;oi<(int)t.opts.size();oi++)if(oi!=bo&&choice_route[oi]>=0&&choice_score[oi]>second_score){second_score=choice_score[oi];second=oi;}
     double score=before;int chosen=-1;EvalResult chosen_eval;
     for(int cand:{bo,second}){
      if(cand<0||choice_route[cand]<0)continue;
      auto&route=p.routes[choice_route[cand]];route.tasks.insert(route.tasks.begin()+choice_position[cand],{ti,cand});
      auto cs=candidate_summary(choice_route[cand],cand,ti);auto ev=evaluate_plan_summary(p,cs);double sc=ev.score;
      if(ev.score>-1e50&&match_score(ev.forecast_margin+newland-oldland)>=before-1e-9&&sc>score){score=sc;chosen=cand;chosen_eval=move(ev);}
      route.tasks.erase(route.tasks.begin()+choice_position[cand]);
     }
     if(chosen>=0){int cri=choice_route[chosen];auto&route=p.routes[cri];route.tasks.insert(route.tasks.begin()+choice_position[chosen],{ti,chosen});apply_evaluation(p,chosen_eval);accepted_insertion(cri,chosen,ti);c_valid[cri]=0;}
     else p.score=before;
    }else{
     // Choose between the feasible production alternatives using the JOINT
     // farm-and-market objective, not just the standalone tile value. The
     // original repair rejected a tile when its first option was unprofitable
     // under current supply, without ever trying that tile's other crops.
     double improvement=-1e100;int selected=-1;EvalResult selected_eval;
     array<int,3> exact{};array<double,3> exact_key;exact.fill(-1);exact_key.fill(-1e100);
     for(int oi=0;oi<(int)t.opts.size();oi++)if(choice_route[oi]>=0){
      double key=choice_score[oi]-capital_bias*t.opts[oi].investment;
      for(int q=0;q<3;q++)if(key>exact_key[q]){for(int z=3-1;z>q;z--){exact_key[z]=exact_key[z-1];exact[z]=exact[z-1];}exact_key[q]=key;exact[q]=oi;break;}
     }
     for(int q=0;q<3;q++){int oi=exact[q];if(oi<0)continue;
      auto&route=p.routes[choice_route[oi]];route.tasks.insert(route.tasks.begin()+choice_position[oi],{ti,oi});
      auto cs=candidate_summary(choice_route[oi],oi,ti);auto ev=evaluate_plan_summary(p,cs);
      if(ev.score<-1e50){route.tasks.erase(route.tasks.begin()+choice_position[oi]);continue;}
      // Land and investment are cash forecast adjustments, so transform them
      // before comparing with the expected-match-score objective.
      double delta=match_score(ev.forecast_margin+newland-oldland)-before;
      double priority=match_score(ev.forecast_margin+newland-oldland-capital_bias*t.opts[oi].investment)-before;
      if(delta>=-1e-9 && priority>improvement){improvement=priority;selected=oi;selected_eval=move(ev);}
      route.tasks.erase(route.tasks.begin()+choice_position[oi]);
     }
     if(selected>=0){int sri=choice_route[selected];auto&route=p.routes[sri];route.tasks.insert(route.tasks.begin()+choice_position[selected],{ti,selected});apply_evaluation(p,selected_eval);accepted_insertion(sri,selected,ti);c_valid[sri]=0;}
     else p.score=before;
    }
   }
  }
 }
 #include "search_methods.inc"
 void emit(const Plan&p,int*out){
  // out: header[32], actions[24][20][3]. Python executes market orders.
  memset(out,0,sizeof(int)*(32+24*U*3));
  const auto& setup=preparation(p);
  out[23]=int(p.sell_first);out[24]=setup.orders.size();out[25]=1;
  out[22]=land_cost(p)>0;out[0]=p.routes.size();out[1]=p.ready;for(int c=0;c<9;c++)out[10+c]=p.holding[c];
  out[9]=int(round(p.score*100.));
  auto emitact=[&](int h,int u,Act a){if(h>=0&&h<24){int*z=out+32+(h*U+u)*3;z[0]=a.op;z[1]=a.item;z[2]=a.n;}};
  for(int u=0;u<(int)p.routes.size();u++){
   const Route&r=p.routes[u];if(r.tasks.empty())continue;
   int w=0,f=0;array<int,3> animals{};for(auto [ti,oi]:r.tasks){auto&o=tasks[ti].opts[oi];w+=o.w;f+=o.f;for(int c=0;c<5;c++)out[4+c]+=o.seeds[c];for(int c=0;c<3;c++){animals[c]+=o.animals[c];out[19+c]+=o.animals[c];}}
   auto inputs=route_inputs(r);w=inputs.w;f=inputs.f;out[2]+=w;out[3]+=f;
   int h=setup.begin[u],at=setup.starts[u];
   auto move_to=[&](int target){
    while(at%10<target%10){emitact(h++,u,{EAST});at++;}
    while(at%10>target%10){emitact(h++,u,{WEST});at--;}
    while(at/10<target/10){emitact(h++,u,{SOUTH});at+=10;}
    while(at/10>target/10){emitact(h++,u,{NORTH});at-=10;}
   };
   int pickup_index=0;
   auto pickup=[&](int item,int quantity){if(!quantity)return;
    if(pickup_index<setup.preloaded[u])emitact(setup.preload_hours[u][pickup_index],u,{PICKUP,item,quantity});
    else emitact(h++,u,{PICKUP,item,quantity});
    pickup_index++;
   };
   pickup(0,w);pickup(8,f);for(int c=0;c<3;c++)pickup(9+c,animals[c]);
   int j=0;for(auto [ti,oi]:r.tasks){auto&t=tasks[ti];auto&o=t.opts[oi];move_to(t.pos);for(auto a:o.acts)emitact(h++,u,a);if(++j==r.split){move_to(shed(at));emitact(h++,u,{DROP});}}
   if(tail_goods(r)>0&&!r.overnight){move_to(shed(at));emitact(h++,u,{DROP});}
  }
  // Keep the established 32-int header / action-table ABI. Nine release
  // hours use two packed words; nine quantities (0..100) use three.
  for(int c=0;c<9;c++){
   out[26+c/6]|=setup.release[c]<<(5*(c%6));
   out[28+c/4]|=setup.deferred[c]<<(7*(c%4));
  }
  out[25]=7;out[31]=accumulate(setup.preloaded.begin(),setup.preloaded.end(),0)+(p.early_pickup?256:0)+(p.preparation_mode<<9);
  int*market_out=out+32+24*U*3;
  market_out[0]=setup.orders.size();
  for(int k=0;k<(int)setup.orders.size();k++){
   market_out[1+3*k]=setup.orders[k].op;market_out[2+3*k]=setup.orders[k].item;market_out[3+3*k]=setup.orders[k].n;
  }
 }
 double best_score(const Plan&p){return p.score;}
};
}

// Persistent speculative search is confined to a single observation snapshot.
// At dawn only its action signatures are imported into a fresh actual solver.
extern "C" double expected_terminal_score(double margin,double scale){return kg::expected_terminal_score(margin,scale);}
extern "C" void* search_create(const int*input,const double*forecast){
 try{auto s=make_unique<kg::Solver>(input,forecast);s->begin_search();return s.release();}catch(...){return nullptr;}
}
extern "C" int search_advance(void*handle,double seconds){
 try{static_cast<kg::Solver*>(handle)->advance_search(seconds);return 0;}catch(...){return -1;}
}
extern "C" int search_import(void*handle,void*previous){
 try{if(handle&&previous)static_cast<kg::Solver*>(handle)->import_warm(*static_cast<kg::Solver*>(previous));return 0;}catch(...){return -1;}
}
extern "C" void search_destroy(void*handle){delete static_cast<kg::Solver*>(handle);}
extern "C" int plan_day_warm(const int*input,const double*forecast,double seconds,void*warm,int*output){
 try{
  auto start=chrono::steady_clock::now();
  auto elapsed=[&](){return chrono::duration<double>(chrono::steady_clock::now()-start).count();};
  auto s=make_unique<kg::Solver>(input,forecast);s->begin_search();
  if(warm)s->import_warm(*static_cast<kg::Solver*>(warm));
  if(seconds>=.15){
   auto seed_start=chrono::steady_clock::now();
   auto seed=make_unique<kg::Solver>(input,forecast,s.get());seed->begin_search();seed->rng.seed(uint32_t(input[6])^0x9e3779b9U);seed->routing_rng.seed(uint32_t(input[6])^0x243f6a88U);
   double seed_limit=min(.30,seconds*.12);
   seed->advance_search(max(0.,seed_limit-chrono::duration<double>(chrono::steady_clock::now()-seed_start).count()));
   s->import_seed(*seed);
  }
  bool improvement=seconds>=.20&&s->day<29;
  s->advance_search(max(0.,(improvement?seconds*.45:seconds)-elapsed()-.008));
  if(improvement&&seconds-elapsed()>.04){
   vector<double> revised;array<double,30> workprice{};
   s->continuation_prices(s->search_best,revised,workprice);
   s->import_continuations(revised,workprice);
   s->advance_search(max(0.,seconds-elapsed()-.008));
  }
  s->evaluate(s->search_best);s->emit(s->search_best,output);
  return 0;
 }catch(...){return -1;}
}
