// Generic C ABI over the installed LibTorch; no policy logic and no Python API.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/Parallel.h>
#include <torch/csrc/autograd/autograd.h>
#include <string>
#include <vector>
#include <cstring>
using Tensor=at::Tensor;
static thread_local std::string error;
#define BEGIN try { error.clear();
#define END } catch(const std::exception& e) {error=e.what();return nullptr;}
extern "C" {
const char* ft_error(){return error.c_str();}
void ft_threads(int n){at::set_num_threads(n);at::set_num_interop_threads(1);}
void ft_worker_threads(int n){at::set_num_threads(n);at::init_num_threads();}
int ft_thread_count(){return at::get_num_threads();}
int ft_grad_mode(int enabled){bool old=at::GradMode::is_enabled();at::GradMode::set_enabled(enabled!=0);return old;}
void ft_free(Tensor* t){delete t;}
int64_t ft_numel(const Tensor* t){return t->numel();}
Tensor* ft_new(const void* values,const int64_t* dims,int ndims,int kind,int device,int grad){
 BEGIN
 auto options=at::TensorOptions().dtype(kind==0?at::kFloat:at::kLong).device(at::kCPU);
 auto t=at::from_blob(const_cast<void*>(values),at::IntArrayRef(dims,ndims),options).clone();
 if(kind==2)t=t.to(at::kBool);
 t=t.to(device<0?at::Device(at::kCPU):at::Device(at::kCUDA,device));
 t.set_requires_grad(grad!=0);return new Tensor(t);
 END
}
int ft_data(const Tensor* t,float* output,int64_t count){
 try {error.clear();if(t->numel()!=count)throw std::runtime_error("tensor output length mismatch");
 auto cpu=t->detach().to(at::kCPU).to(at::kFloat).contiguous();std::memcpy(output,cpu.data_ptr<float>(),count*sizeof(float));return 0;
 }catch(const std::exception& e){error=e.what();return -1;}
}
int ft_copy(Tensor* dst,const Tensor* src){try{error.clear();at::NoGradGuard guard;dst->copy_(*src);return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int ft_backward(const Tensor* loss){try{error.clear();torch::autograd::backward({*loss});return 0;}catch(const std::exception& e){error=e.what();return -1;}}
int ft_has_grad(const Tensor* t){return t->grad().defined();}
void ft_zero_grad(Tensor* t){if(t->grad().defined())t->mutable_grad()=Tensor();}
Tensor* ft_op(int op,const Tensor* const* in,int count,const int64_t* ints,int ni,const double* scalars,int ns){
 BEGIN
 const auto& a=*in[0];
 switch(op){
 case 0:return new Tensor(at::linear(a,*in[1],*in[2]));
 case 1:return new Tensor(a.tanh());
 case 2:{std::vector<Tensor> xs;for(int i=0;i<count;i++)xs.push_back(*in[i]);return new Tensor(at::cat(xs,ints[0]));}
 case 3:return new Tensor(a.unsqueeze(ints[0]));
 case 4:return new Tensor(a.squeeze(ints[0]));
 case 5:return new Tensor(a.expand(at::IntArrayRef(ints,ni)));
 case 6:return new Tensor(at::log_sigmoid(a));
 case 42:return new Tensor(a.softmax(ints[0],at::kFloat));
 case 7:return new Tensor(a.log_softmax(ints[0],at::kFloat));
 case 8:return new Tensor(a.logical_not());
 case 9:return new Tensor(a.any(ints[0]));
 case 10:return new Tensor(a.masked_fill(*in[1],scalars[0]));
 case 11:return new Tensor(at::where(*in[1],a,*in[2]));
 case 12:return new Tensor(a+*in[1]);
 case 13:return new Tensor(a-*in[1]);
 case 14:return new Tensor(a* *in[1]);
 case 15:return new Tensor(a/ *in[1]);
 case 16:return new Tensor(a.exp());
 case 17:return new Tensor(ni?a.sum(at::IntArrayRef(ints,ni)):a.sum());
 case 18:return new Tensor(a.mean());
 case 19:return new Tensor(a.clamp(scalars[0],scalars[1]));
 case 20:return new Tensor(at::minimum(a,*in[1]));
 case 21:return new Tensor(a.gather(ints[0],*in[1]));
 case 22:return new Tensor(a.index_select(ints[0],*in[1]));
 case 23:return new Tensor(a.std(ints[0]!=0));
 case 24:return new Tensor(a.square());
 case 25:return new Tensor(a.detach());
 case 26:return new Tensor(at::zeros_like(a));
 case 27:return new Tensor(a.narrow(ints[0],ints[1],ints[2]));
 case 28:return new Tensor(a.sqrt());
 case 29:return new Tensor(a.clamp_min(scalars[0]));
 case 43:return new Tensor(a/scalars[0]);
 case 30:return new Tensor(a+scalars[0]);
 case 31:return new Tensor(a*scalars[0]);
 case 32:return new Tensor(a.neg());
 case 33:return new Tensor(a.grad().defined()?a.grad():at::zeros_like(a));
 case 34:return new Tensor(a.clone());
 case 35:return new Tensor(a.logsumexp(at::IntArrayRef(ints,ni),true));
 case 36:return new Tensor(a.reshape(at::IntArrayRef(ints,ni)));
 case 37:return new Tensor(a.norm());
 case 38:return new Tensor(a.reciprocal());
 case 39:return new Tensor(at::addcdiv(a,*in[1],*in[2],scalars[0]));
 case 40:return new Tensor(a.lerp(*in[1],scalars[0]));
 // Hierarchical categorical distribution: category head and conditional candidates.
 case 44:{
 auto groups=*in[2];auto mask=*in[3];auto head=*in[1];
 auto ids=at::arange(head.size(1),groups.options());
 auto membership=groups.unsqueeze(-1).eq(ids).logical_and(mask.unsqueeze(-1));
 auto legal=membership.any(1);
 auto glp=head.masked_fill(legal.logical_not(),-1e9).log_softmax(-1);
 auto expanded=a.unsqueeze(-1).expand_as(membership);
 auto denom=expanded.masked_fill(membership.logical_not(),-1e9).logsumexp(1);
 auto lp=a-denom.gather(1,groups)+glp.gather(1,groups);
 return new Tensor(lp.masked_fill(mask.logical_not(),-1e9));
 }
 // Actual exploration behavior distribution; used identically at sampling and PPO update.
 case 45:{auto q=*in[1];auto eps=*in[2];auto mask=*in[3];
 return new Tensor(((1.-eps)*a.exp()+eps*q).clamp_min(1e-30).log().masked_fill(mask.logical_not(),-1e9));}
 // Plan policies have a single legal category: avoid the N x width x 19 expansion.
 case 46:return new Tensor(a.masked_fill(in[1]->logical_not(),-1e9).log_softmax(-1));
 case 41:return new Tensor(at::addcmul(a,*in[1],*in[2],scalars[0]));
 default:throw std::runtime_error("unknown tensor operation");
 }
 END
}
}
