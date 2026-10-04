// Adapted from msdsm/kaggriculture-solution 84057a0fda4238ccdebc46f9bf5496c6c4b2e00d; see docs/action_ppo_sources.md.
 int holds[9]{};int hold_schedule[30][9]{};
 double storage_flow[30][9]{}, storage_opponent[30][9]{};
 double storage_score(int c)const{
  double sum=0,sq=0;int scenarios=scenario_count;
  for(int sc=0;sc<scenarios;sc++){
   int inv=mi[c];double v=0;
   for(int d=day;d<30;d++){
    double df=d==day?1.:future_df[d];
    int dem=d==day?6*shop_demand[c]+(c<8):scenario_demand[sc][d][c];
    inv-=dem/2;
    v+=df*trade_difference(c,inv,int(storage_flow[d][c]),int(storage_opponent[d][c]));
    inv-=dem-dem/2;
   }
   sum+=v;sq+=v*v;
  }
  double mean=sum/scenarios;
  return mean-.25*sqrt(max(0.,sq/scenarios-mean*mean));
 }
 void compute_holds(){
  if(day>=29)return;
  double scores[9];for(int c=0;c<9;c++)scores[c]=storage_score(c);
  // The shared storage budget is deliberately below 100, reserving capacity
  // for deliveries and tomorrow's inputs. No invisible opponent data is used.
  for(int n=0;n<40;n++){
   double best=1.;int bc=-1,bd=-1;
   for(int c=0;c<9;c++){
    if(holds[c]>=stock[c])continue;
    for(int d=day+1;d<30;d++){
     storage_flow[day][c]--;storage_flow[d][c]++;
     double gain=storage_score(c)-scores[c];
     storage_flow[day][c]++;storage_flow[d][c]--;
     if(gain>best){best=gain;bc=c;bd=d;}
    }
   }
   if(bc<0)break;
   storage_flow[day][bc]--;storage_flow[bd][bc]++;hold_schedule[bd][bc]++;holds[bc]++;scores[bc]+=best;
  }
 }
