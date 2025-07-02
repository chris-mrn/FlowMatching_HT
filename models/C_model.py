

import numpy as np
import torch
# from marginalTailAdaptiveFlow.utils.flows import experiment
import flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from .utils.scaler_grad import NativeScalerWithGradNormCount as NativeScaler
from matplotlib import cm




class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)



print_every = 10
min_loss=9999999999




from typing import Callable
from torch import nn, Tensor

def jvp(f,x, v) -> tuple[Tensor, ...]:
    return torch.autograd.functional.jvp(
        f, x, v, 
        create_graph=torch.is_grad_enabled()
    )


def t_dir(f, t) -> tuple[Tensor, ...]:
    return jvp(f, t, torch.ones_like(t))


def get_t_dir(dimension,noise2data,Model, x: Tensor, t: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    def flow(Xz,timez):
        param_tail_eps=Model(timez.unsqueeze(1))
        dummy_tail_param2=param_tail_eps.reshape(param_tail_eps.shape[0],4,dimension)
        _unc_pos_tail2,_unc_neg_tail2,shift2,_unc_scale2, = dummy_tail_param2[:,0,:],dummy_tail_param2[:,1,:],dummy_tail_param2[:,2,:],dummy_tail_param2[:,3,:]

        D1=noise2data.pos_tail(_unc_pos_tail2)
        D2=noise2data.neg_tail(_unc_neg_tail2)
        D3=shift2
        D4=noise2data.scale(_unc_scale2)   

        valuez=torch.cat([D1,D2,D3,D4],1)

        return(valuez)









    def f(x_in):
        def f_(t_in):
            return flow(x_in, t_in)
        return f_

    return t_dir(f(x), t)


# In[ ]:


class HeavyTailedModel(nn.Module):
    def __init__(
        self,tail_param_net,PreVFnet,NOISE2DATA,dimension,simulation=False
    ):
        # self.features = features
        super(HeavyTailedModel, self).__init__()
        self.tail_param_net=tail_param_net
        self.PreVFnet=PreVFnet
        self.NOISE2DATA=NOISE2DATA
        self.dimension=dimension
        self.sim=simulation
    def forward(self, x_t,time_t):
        #during ode simulation time_t is a single number while when sim=False time_t is a array of different times
        if self.sim:
            
            time_t=time_t.reshape(-1).expand(x_t.shape[0])
        else:
            time_t=time_t.unsqueeze(1)

        dim=self.dimension

        time_t=time_t.reshape(-1).expand(x_t.shape[0])
    

        prefinal_vf=self.PreVFnet(x_t,time_t)

        prefinal_vf=prefinal_vf.reshape(prefinal_vf.shape[0],-1)
            
        param_tail=self.tail_param_net(time_t.unsqueeze(1)) #BX80

        True_timederiv=get_t_dir(dim,self.NOISE2DATA,self.tail_param_net,x_t,time_t)
        param_grad=True_timederiv[1] #time derive



        param_tail=param_tail.reshape(param_tail.shape[0],-1)
        param_tail_pre_eps=param_tail

        phi_t=self.NOISE2DATA.inverse(x_t,param_tail,False,None,None)

        jacobian_phi=self.NOISE2DATA.fwd_dTTF_dz(phi_t, param_tail)
        jacobian_param_tail=self.NOISE2DATA.dTTF_dtailparam(phi_t, param_tail)



        first_part=param_grad[:,0:dim]*jacobian_param_tail[0]+param_grad[:,dim:2*dim]*jacobian_param_tail[1]+param_grad[:,2*dim:3*dim]*jacobian_param_tail[2]+param_grad[:,3*dim:4*dim]*jacobian_param_tail[3]
        second_part=torch.bmm(jacobian_phi,prefinal_vf.unsqueeze(2)).squeeze(2)

        velocity_field=first_part+second_part

        return(velocity_field)                    



class LightTailedModel(nn.Module):
    def __init__(
        self,PreVFnet,dimension,simulation=False
    ):
        # self.features = features
        super(LightTailedModel, self).__init__()
        self.PreVFnet=PreVFnet
        self.dimension=dimension
        self.sim=simulation
    def forward(self, x_t,time_t):
        dim=self.dimension
        if self.sim:
            
            time_t=time_t.reshape(-1).expand(x_t.shape[0])
        else:
            time_t=time_t.unsqueeze(1)
    

        prefinal_vf=self.PreVFnet(x_t,time_t)

        prefinal_vf=prefinal_vf.reshape(prefinal_vf.shape[0],-1)

        velocity_field=prefinal_vf

        return(velocity_field)                    

 

class heavy_tail_FM:
    def __init__(self,Tail_paramNet,model,TTF,dimension,iterations,device,steps):
        self.model=model
        self.Tail_paramNet=Tail_paramNet
        self.TTF=TTF
        self.device=device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.dim=dimension
        self.iterations=iterations
        self.flow_model=HeavyTailedModel(Tail_paramNet,model,TTF,dimension)
        self.steps=steps
        self.count=0
    def train_epoch(self,optim1,optim2,train_loader,noise_loader,curr_iter):
        loss_scaler = NativeScaler()
        iI=curr_iter
        for data,noise in zip(train_loader,noise_loader):
            self.count=self.count+1
            if self.count==self.steps:
                break


            optim1.zero_grad()
            optim2.zero_grad()

            x_1=data[0].float().to(self.device)   #batch x 20
            x_0 = noise.float().to(self.device) #batch x 20


            if iI<(3*self.iterations)//4:
                t = 1-torch.sqrt(1-torch.rand(x_1.shape[0])).to(self.device) #best
            else:
                t= -torch.log(1 - torch.rand(x_1.shape[0]) * (1 - torch.exp(torch.tensor(-1)))).to(self.device)

            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t,time_t,dx_t=path_sample.x_t,path_sample.t,path_sample.dx_t  #x_t- B X 20



            velocity_field=self.flow_model(x_t,time_t)

            
            loss = torch.pow( velocity_field - dx_t, 2).mean() 
            print("LOSS-",loss)

            loss_scaler(
                loss,
                optim1,
                optim2,
                parameters=self.model.parameters(),
                parameters2=self.Tail_paramNet.parameters(),
                update_grad=True,
                approach=2
                )      

   
       
    def flow(self,x_t,t):
        return self.flow_model(x_t, t)
    

    def generate(self,num_samples):
        # step size for ode solver
        self.flow_model.sim=True
        wrapped_vf = WrappedModel(self.flow_model)
        step_size = 0.05


        batch_size = num_samples  # batch size
        T = torch.linspace(0,1,10)  # sample times
        T = T.to(self.device)
 
        x_init = torch.randn((batch_size, self.dim), dtype=torch.float32, device=self.device)
        solver = ODESolver(velocity_model=wrapped_vf)  
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  
        sol = sol.cpu().numpy()
        generated_data=sol[9]
        self.flow_model.sim=False
        return(generated_data)
    
    


# In[ ]:





# In[ ]:





# In[ ]:


class Light_tail_FM:
    def __init__(self,model,dimension,iterations,device,steps):
        self.model=model
        self.device=device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.dim=dimension
        self.flow_model=LightTailedModel(model,dimension,simulation=False).to(device)
        self.iterations=iterations
        self.count=0
        self.steps=steps
    def train_epoch(self,optim1,train_loader,noise_loader,curr_iter):
        loss_scaler = NativeScaler()
        iterations=self.iterations
        iI=curr_iter

        for data,noise in zip(train_loader,noise_loader):
            self.count=self.count+1
            if self.count==self.steps:
                break
            optim1.zero_grad()

            x_1=data[0].float().to(self.device)   #batch x 20
            x_0 = noise.float().to(self.device)#torch.randn_like(x_1).float().to(self.device) #batch x 20


            if iI<(3*iterations)//4:
                t = 1-torch.sqrt(1-torch.rand(x_1.shape[0])).to(self.device) #best
            else:
                t= -torch.log(1 - torch.rand(x_1.shape[0]) * (1 - torch.exp(torch.tensor(-1)))).to(self.device)

            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            x_t,time_t,dx_t=path_sample.x_t,path_sample.t,path_sample.dx_t  #x_t- B X 20



            velocity_field=self.flow_model(x_t,time_t)

            
            loss = torch.pow( velocity_field - dx_t, 2).mean() 
            print("LOSS-",loss)

            loss_scaler(
                loss,
                optim1,
                None,
                parameters=self.model.parameters(),
                parameters2=None,
                update_grad=True,
                approach=1
                )      


 
    def flow(self,x_t,t):
        return self.flow_model(x_t, t)
    def generate(self,num_samples):
        # step size for ode solver
        self.flow_model.sim=True
        wrapped_vf = WrappedModel(self.flow_model)
        step_size = 0.05


        batch_size = num_samples  # batch size
        T = torch.linspace(0,1,10)  # sample times
        T = T.to(self.device)
 
        x_init = torch.randn((batch_size, self.dim), dtype=torch.float32, device=self.device)
        solver = ODESolver(velocity_model=wrapped_vf)  
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  
        sol = sol.cpu().numpy()
        generated_data=sol[9]
        self.flow_model.sim=False
        return(generated_data)

 



class heavy_tail_input:
    def __init__(self,Tail_paramNet,model,TTF,dimension,iterations,device,steps):
        self.model=model
        self.Tail_paramNet=Tail_paramNet
        self.TTF=TTF
        self.device=device
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.dim=dimension
        self.iterations=iterations
        self.flow_model=LightTailedModel(model,dimension,simulation=False).to(device)
        self.steps=steps
        self.count=0
    def train_epoch(self,optim1,optim2,train_loader,noise_loader,curr_iter):
        loss_scaler = NativeScaler()
        iI=curr_iter
        for data,noise in zip(train_loader,noise_loader):
            self.count=self.count+1
            if self.count==self.steps:
                break


            optim1.zero_grad()
            optim2.zero_grad()
            const=-1
            param_tail_pre=self.Tail_paramNet(1+torch.zeros(noise.shape[0],1).to(self.device)) #BX80
            dummy_tail_param=param_tail_pre.reshape(param_tail_pre.shape[0],4,self.dim)
            _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:]**2+const,dummy_tail_param[:,1,:]**2+const,dummy_tail_param[:,2,:]*0,dummy_tail_param[:,3,:]*0      # i am keeping shift 0 and var softplus(0)     
            param_tail=torch.cat([_unc_pos_tail,_unc_neg_tail,shift,_unc_scale],1)

            x_1=data[0].float().to(self.device)   #batch x 20
            x_0 = noise.float().to(self.device)#torch.randn_like(x_1).float().to(self.device) #batch x 20
            x_0=self.TTF(x_0,param_tail)


            if iI<(3*self.iterations)//4:
                t = 1-torch.sqrt(1-torch.rand(x_1.shape[0])).to(self.device) #best
            else:
                t= -torch.log(1 - torch.rand(x_1.shape[0]) * (1 - torch.exp(torch.tensor(-1)))).to(self.device)

            path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
            
            x_t,time_t,dx_t=path_sample.x_t,path_sample.t,path_sample.dx_t  #x_t- B X 20
            # print("CHECK GRAD",x_1.shape,x_t.requires_grad,dx_t.requires_grad)




            velocity_field=self.flow_model(x_t,time_t)

            
            loss = torch.pow( velocity_field - dx_t, 2).mean() 
            print("LOSS-",loss)

            loss_scaler(
                loss,
                optim1,
                optim2,
                parameters=self.model.parameters(),
                parameters2=self.Tail_paramNet.parameters(),
                update_grad=True,
                approach=2
                )      

   
       
    def flow(self,x_t,t):
        return self.flow_model(x_t, t)
    

    def generate(self,num_samples):
        # step size for ode solver
        self.flow_model.sim=True
        wrapped_vf = WrappedModel(self.flow_model)
        step_size = 0.01
        const=-1
        param_tail_pre=self.Tail_paramNet(1+torch.zeros(num_samples,1).to(self.device)) #BX80
        dummy_tail_param=param_tail_pre.reshape(param_tail_pre.shape[0],4,self.dim)
        _unc_pos_tail,_unc_neg_tail,shift,_unc_scale =dummy_tail_param[:,0,:]**2+const,dummy_tail_param[:,1,:]**2+const,dummy_tail_param[:,2,:]*0,dummy_tail_param[:,3,:]*0      # i am keeping shift 0 and var softplus(0)     
        param_tail=torch.cat([_unc_pos_tail,_unc_neg_tail,shift,_unc_scale],1)
        
        batch_size = num_samples  # batch size
        T = torch.linspace(0,1,10)  # sample times
        T = T.to(self.device)
 
        x_init = torch.randn((batch_size, self.dim), dtype=torch.float32, device=self.device)
        x_init=self.TTF(x_init,param_tail)
        solver = ODESolver(velocity_model=wrapped_vf)  
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  
        sol = sol.cpu().numpy()
        generated_data=sol[9]

        self.flow_model.sim=False
        
        
        return(generated_data)
    
    